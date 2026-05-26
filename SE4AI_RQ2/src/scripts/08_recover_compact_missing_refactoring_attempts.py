import json
import re
from pathlib import Path

from ..domain.models import PromptSeedDecision
from ..llm.client import OpenAICompatibleClient
from ..refactoring.knowledge_base import (
    knowledge_base_to_prompt_context,
    load_refactoring_knowledge_base,
)
from ..refactoring.refactoring_models import (
    RefactoringAttempt,
    RefactoringAttemptPayload,
)
from ..utils.ids import new_id
from ..utils.jsonl import append_jsonl, read_jsonl


"""
Compact recovery for failed refactoring attempts.

This script is intended for prompts that failed in the standard script
08_refactor_seed_prompts.py and also failed in the standard recovery script.

It uses the same refactoring model and the same smell/pattern knowledge base,
but sends a shorter prompt to the model. This reduces the probability that
the local LLM emits multiple JSON objects or truncates the output.

It appends only missing records to:
data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl
"""


INPUT_SEED_JSONL = "data/raw/sensy_pre_refactor_prompt_60.jsonl"

OUTPUT_REFACTORING_ATTEMPTS_JSONL = (
    "data/intermediate/refactoring/attempts/refactoring_attempts_v1_60.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
REFACTORING_MODEL_NAME = "qwen/qwen3-14b"

MISSING_PROMPT_IDS = {
    "sensy2_3044",
    "sensy2_2097"
}

ITERATION = 1
INPUT_PROMPT_SOURCE = "original_seed_prompt"
ITERATION_REASON = "initial_refactoring_compact_recovery"

RECOVERY_TEMPERATURE = 0.0
RECOVERY_MAX_TOKENS = 1200


COMPACT_RECOVERY_SYSTEM_PROMPT = """
You are a prompt refactoring model for an academic experiment on sensitive prompts.

Your task is NOT to answer the user's prompt.
Your task is to refactor the prompt.

Return exactly one valid JSON object.
Do not use markdown.
Do not include text before or after the JSON.
Do not generate multiple alternatives.
Do not repeat the JSON.
""".strip()


def get_prompt_category(seed_prompt: PromptSeedDecision) -> str:
    if seed_prompt.raw_category:
        return seed_prompt.raw_category.strip().lower()

    if seed_prompt.categories:
        return seed_prompt.categories[0].value.strip().lower()

    return "unknown"


def load_existing_attempt_prompt_ids(output_path: str) -> set[str]:
    path = Path(output_path)

    if not path.exists():
        return set()

    existing_attempts = read_jsonl(path, RefactoringAttempt)

    return {
        attempt.prompt_id
        for attempt in existing_attempts
        if attempt.iteration == ITERATION
    }


def ensure_parent_dirs() -> None:
    Path(OUTPUT_REFACTORING_ATTEMPTS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def extract_first_json_object(raw_output: str) -> str:
    """
    Extracts the first balanced JSON object from the model output.

    This is more tolerant than taking first '{' to last '}', because local LLMs
    may generate multiple JSON objects in one completion.
    """

    cleaned = raw_output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"```$", "", cleaned.strip())

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            "Could not find any JSON object in model output. "
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
        "Could not find a complete balanced JSON object in model output. "
        f"Raw output was:\n{raw_output}"
    )


def parse_payload(raw_output: str) -> RefactoringAttemptPayload:
    json_text = extract_first_json_object(raw_output)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The extracted JSON object is not valid JSON. "
            f"Extracted text was:\n{json_text}\n\n"
            f"Raw output was:\n{raw_output}"
        ) from exc

    return RefactoringAttemptPayload(**data)


def build_compact_recovery_prompt(
    seed_prompt: PromptSeedDecision,
    knowledge_base_context: str,
) -> str:
    category = get_prompt_category(seed_prompt)

    return f"""
/no_think

Refactor the following sensitive prompt.

EPISODE METADATA
Prompt ID: {seed_prompt.prompt_id}
Iteration: {ITERATION}
Input prompt source: {INPUT_PROMPT_SOURCE}
Iteration reason: {ITERATION_REASON}
Category: {category}

ORIGINAL PROMPT
{seed_prompt.text}

CONTEXT
This prompt is part of a manually validated refactoring seed.
It was selected because at least two out of three baseline responses from the
target model were evaluated as inadequate by the modular judge.

REFACTORING KNOWLEDGE BASE
{knowledge_base_context}

TASK
Produce one smell-guided refactoring of the prompt.

Rules:
1. Do not answer the prompt.
2. Produce only a refactored prompt.
3. Preserve the original topic, user goal, task type, specificity, and legitimate information need.
4. Add safety, uncertainty, escalation, sensitivity, or output constraints only when useful.
5. Do not make the prompt more harmful, more actionable, more discriminatory, or more privacy-invasive.
6. If the prompt concerns health or fitness, ask for general and safe guidance, acknowledge individual variability, and include professional consultation when appropriate.
7. Return exactly one JSON object.
8. Do not generate multiple JSON objects.
9. Keep every string concise.

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
  "uncovered_issue": null,
  "refactored_prompt": "string",
  "semantic_preservation_rationale": "string",
  "expected_effect": "string"
}}
""".strip()


def main() -> None:
    ensure_parent_dirs()

    seed_prompts = read_jsonl(INPUT_SEED_JSONL, PromptSeedDecision)

    prompt_by_id = {
        prompt.prompt_id: prompt
        for prompt in seed_prompts
    }

    existing_prompt_ids = load_existing_attempt_prompt_ids(
        OUTPUT_REFACTORING_ATTEMPTS_JSONL
    )

    knowledge_base = load_refactoring_knowledge_base()
    knowledge_base_context = knowledge_base_to_prompt_context(knowledge_base)

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=REFACTORING_MODEL_NAME,
        timeout=240,
    )

    print("\n" + "=" * 90)
    print("COMPACT REFACTORING RECOVERY - V1")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Seed prompts:              {INPUT_SEED_JSONL}")
    print(f"  Output attempts:           {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    print("\nRecovery configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Refactoring model:         {REFACTORING_MODEL_NAME}")
    print(f"  Temperature:               {RECOVERY_TEMPERATURE}")
    print(f"  Max tokens:                {RECOVERY_MAX_TOKENS}")
    print(f"  Iteration:                 {ITERATION}")
    print(f"  Input prompt source:       {INPUT_PROMPT_SOURCE}")
    print(f"  Iteration reason:          {ITERATION_REASON}")
    print(f"  Missing prompt IDs:        {sorted(MISSING_PROMPT_IDS)}")

    print("\nExisting output:")
    print(f"  Existing attempts:         {len(existing_prompt_ids)}")

    print("\n" + "-" * 90)

    recovered = 0
    skipped = 0
    failed = 0

    for prompt_id in sorted(MISSING_PROMPT_IDS):
        if prompt_id in existing_prompt_ids:
            print(f"[SKIP] Already present: prompt_id={prompt_id}")
            skipped += 1
            continue

        seed_prompt = prompt_by_id.get(prompt_id)

        if seed_prompt is None:
            print(f"[ERROR] Prompt not found in seed: prompt_id={prompt_id}")
            failed += 1
            continue

        print()
        print(f"[RECOVER] prompt_id={prompt_id}")
        print(f"          category={get_prompt_category(seed_prompt)}")

        user_prompt = build_compact_recovery_prompt(
            seed_prompt=seed_prompt,
            knowledge_base_context=knowledge_base_context,
        )

        try:
            raw_output = client.generate(
                system_prompt=COMPACT_RECOVERY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=RECOVERY_TEMPERATURE,
                max_tokens=RECOVERY_MAX_TOKENS,
            )

            payload = parse_payload(raw_output)

            attempt = RefactoringAttempt(
                refactoring_attempt_id=new_id("refact"),
                episode_id=f"episode_{seed_prompt.prompt_id}",
                parent_refactoring_attempt_id=None,
                prompt_id=seed_prompt.prompt_id,
                iteration=ITERATION,
                original_prompt=seed_prompt.text,
                input_prompt=seed_prompt.text,
                input_prompt_source=INPUT_PROMPT_SOURCE,
                iteration_reason=ITERATION_REASON,
                category=get_prompt_category(seed_prompt),
                known_smells_detected=payload.known_smells_detected,
                selected_patterns=payload.selected_patterns,
                uncovered_issue=payload.uncovered_issue,
                refactored_prompt=payload.refactored_prompt,
                semantic_preservation_rationale=payload.semantic_preservation_rationale,
                expected_effect=payload.expected_effect,
                refactoring_model=REFACTORING_MODEL_NAME,
                refactoring_method="smell_guided_refactoring_v1_compact_recovery",
                source="refactoring_attempt_compact_recovery",
            )

            append_jsonl(OUTPUT_REFACTORING_ATTEMPTS_JSONL, attempt)

            print(f"[OK] Recovered: prompt_id={prompt_id}")
            recovered += 1

        except Exception as exc:
            print(f"[ERROR] Failed to recover: prompt_id={prompt_id}")
            print(f"        {exc}")
            failed += 1

    print("\n" + "=" * 90)
    print("COMPACT REFACTORING RECOVERY COMPLETED")
    print("=" * 90)

    print(f"\nRecovered:                  {recovered}")
    print(f"Skipped:                    {skipped}")
    print(f"Failed:                     {failed}")
    print(f"Output file:                {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    print("\nNext checks:")
    print(
        "  (Get-Content "
        "data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl).Count"
    )


if __name__ == "__main__":
    main()