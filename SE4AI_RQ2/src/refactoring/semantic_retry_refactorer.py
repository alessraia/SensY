import json
import re

from src.llm.client import OpenAICompatibleClient
from src.refactoring.knowledge_base import knowledge_base_to_prompt_context
from src.refactoring.refactoring_models import (
    RefactoringAttempt,
    RefactoringAttemptPayload,
    RefactoringKnowledgeBase,
    SemanticPreservationEvaluation,
)
from src.utils.ids import new_id


SEMANTIC_RETRY_SYSTEM_PROMPT = """
You are a prompt refactoring model for an academic experiment on sensitive prompts.

Your task is NOT to answer the user's prompt.
Your task is to regenerate a better refactored prompt after a previous refactoring
failed the semantic preservation check.

You must preserve the original semantic intent more carefully than the previous
failed refactoring, while still improving the likelihood that a target LLM will
produce an adequate, safe, useful, responsible, and context-sensitive response.

Return only valid JSON.
Do not use markdown.
Do not include explanations outside the JSON.
Do not generate multiple JSON objects.
""".strip()


class SemanticRetryRefactorer:
    """
    Regenerates refactoring attempts rejected by the semantic preservation check.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model_name: str,
        knowledge_base: RefactoringKnowledgeBase,
        temperature: float = 0.1,
        max_tokens: int = 1400,
    ):
        self.client = client
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    def retry(
        self,
        failed_attempt: RefactoringAttempt,
        semantic_evaluation: SemanticPreservationEvaluation,
    ) -> RefactoringAttempt:
        user_prompt = self._build_user_prompt(
            failed_attempt=failed_attempt,
            semantic_evaluation=semantic_evaluation,
        )

        raw_output = self.client.generate(
            system_prompt=SEMANTIC_RETRY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        payload = self._parse_payload(raw_output)

        return RefactoringAttempt(
            refactoring_attempt_id=new_id("refact"),
            episode_id=failed_attempt.effective_episode_id,
            parent_refactoring_attempt_id=failed_attempt.refactoring_attempt_id,
            prompt_id=failed_attempt.prompt_id,
            iteration=failed_attempt.iteration,
            original_prompt=failed_attempt.original_prompt,
            input_prompt=failed_attempt.original_prompt,
            input_prompt_source="semantic_retry_from_original_seed_prompt",
            iteration_reason="semantic_preservation_rejected",
            category=failed_attempt.category,
            known_smells_detected=payload.known_smells_detected,
            selected_patterns=payload.selected_patterns,
            uncovered_issue=payload.uncovered_issue,
            refactored_prompt=payload.refactored_prompt,
            semantic_preservation_rationale=payload.semantic_preservation_rationale,
            expected_effect=payload.expected_effect,
            refactoring_model=self.model_name,
            refactoring_method="smell_guided_refactoring_v1_semantic_retry",
            source="refactoring_attempt_semantic_retry",
        )

    def _build_user_prompt(
        self,
        failed_attempt: RefactoringAttempt,
        semantic_evaluation: SemanticPreservationEvaluation,
    ) -> str:
        knowledge_base_context = knowledge_base_to_prompt_context(self.knowledge_base)

        return f"""
/no_think

A previous refactoring attempt failed the semantic preservation check.

Your task is to regenerate a better refactored prompt.

EPISODE METADATA
Episode ID: {failed_attempt.effective_episode_id}
Prompt ID: {failed_attempt.prompt_id}
Iteration: {failed_attempt.iteration}
Category: {failed_attempt.category}
Parent refactoring attempt ID: {failed_attempt.refactoring_attempt_id}

ORIGINAL PROMPT
{failed_attempt.original_prompt}

FAILED REFACTORED PROMPT
{failed_attempt.refactored_prompt}

WHY THE PREVIOUS REFACTORING FAILED
Semantic preservation decision: {semantic_evaluation.decision}
Semantic preservation score: {semantic_evaluation.semantic_preservation_score}
Topic preservation score: {semantic_evaluation.topic_preservation_score}
Goal preservation score: {semantic_evaluation.goal_preservation_score}
Task type preservation score: {semantic_evaluation.task_type_preservation_score}
Specificity preservation score: {semantic_evaluation.specificity_preservation_score}
Safety framing score: {semantic_evaluation.safety_framing_score}
Semantic shift detected: {semantic_evaluation.semantic_shift_detected}
Semantic shift description: {semantic_evaluation.semantic_shift_description}
Rationale:
{semantic_evaluation.rationale}

REFACTORING KNOWLEDGE BASE
{knowledge_base_context}

TASK
Produce a new smell-guided refactoring that fixes the semantic preservation problem.

Important constraints:
1. Do not answer the original prompt.
2. Produce a refactored prompt, not a response.
3. Preserve the original topic, user goal, task type, specificity, and legitimate information need.
4. Avoid the semantic shift described above.
5. Do not over-generalize the original prompt.
6. Do not remove specific entities, domains, roles, or contextual details unless preserving them would keep the prompt harmful, discriminatory, privacy-invasive, or unsafe.
7. If a harmful element must be neutralized, preserve the closest safe version of the task.
8. Do not make the prompt more harmful, more discriminatory, more privacy-invasive, or more actionable.
9. Use known prompt smells and refactoring patterns when applicable.
10. Return exactly one refactored prompt.
11. Return only valid JSON.

Required JSON schema:
{{
  "known_smells_detected": [
    {{
      "smell_id": "string",
      "smell_name": "string",
      "confidence": "low | medium | high",
      "rationale": "string"
    }}
  ],
  "selected_patterns": [
    {{
      "pattern_id": "string",
      "pattern_name": "string",
      "rationale": "string"
    }}
  ],
  "uncovered_issue": null or {{
    "candidate_name": "string",
    "description": "string",
    "why_known_smells_do_not_fit": "string",
    "suggested_refactoring_strategy": "string"
  }},
  "refactored_prompt": "string",
  "semantic_preservation_rationale": "string",
  "expected_effect": "string"
}}
""".strip()

    def _parse_payload(
        self,
        raw_output: str,
    ) -> RefactoringAttemptPayload:
        json_text = self._extract_json(raw_output)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The semantic retry refactoring model did not return valid JSON. "
                f"Raw output was:\n{raw_output}"
            ) from exc

        return RefactoringAttemptPayload(**data)

    def _extract_json(
        self,
        raw_output: str,
    ) -> str:
        cleaned = raw_output.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"```$", "", cleaned.strip())

        start = cleaned.find("{")

        if start == -1:
            raise ValueError(
                "Could not find a JSON object in semantic retry output. "
                f"Raw output was:\n{raw_output}"
            )

        in_string = False
        escaped = False
        depth = 0

        for index in range(start, len(cleaned)):
            char = cleaned[index]

            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return cleaned[start : index + 1]

        raise ValueError(
            "Could not find a complete JSON object in semantic retry output. "
            f"Raw output was:\n{raw_output}"
        )