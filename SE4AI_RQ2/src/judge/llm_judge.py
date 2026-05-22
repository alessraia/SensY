from ..domain.models import (
    JudgeResponseEvaluationPayload,
    ResponseEvaluation,
    TargetResponse,
)
from ..llm.client import OpenAICompatibleClient
from ..utils.ids import new_id
from .parsers import parse_model_output
from .rubrics import BASELINE_RESPONSE_EVALUATION_RUBRIC


class LLMJudge:
    """
    LLM-as-a-Judge per valutare le risposte baseline.

    Riceve una TargetResponse e restituisce una ResponseEvaluation.
    """

    def __init__(self, client: OpenAICompatibleClient, model_name: str):
        self.client = client
        self.model_name = model_name

    def evaluate_baseline_response(
        self,
        response: TargetResponse,
        categories: list[str],
        raw_category: str | None = None,
    ) -> ResponseEvaluation:
        categories_text = ", ".join(categories) if categories else "Unknown"

        user_prompt = f"""
Sensitive prompt categories:
{categories_text}

Raw category:
{raw_category}

Original user prompt:
{response.prompt_text}

Target model:
{response.target_model}

Target model response:
{response.response_text}
""".strip()

        raw_output = self.client.generate(
            system_prompt=BASELINE_RESPONSE_EVALUATION_RUBRIC,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=768,
        )

        parsed = parse_model_output(raw_output, JudgeResponseEvaluationPayload)

        return ResponseEvaluation(
            evaluation_id=new_id("eval"),
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            target_model=response.target_model,
            judge_model=self.model_name,
            repetition=response.repetition,
            adequacy_score=parsed.adequacy_score,
            safety_score=parsed.safety_score,
            helpfulness_score=parsed.helpfulness_score,
            sensitivity_awareness_score=parsed.sensitivity_awareness_score,
            over_refusal=parsed.over_refusal,
            under_refusal=parsed.under_refusal,
            label=parsed.label,
            rationale=parsed.rationale,
            failure_modes=parsed.failure_modes,
            source="baseline_evaluation",
        )