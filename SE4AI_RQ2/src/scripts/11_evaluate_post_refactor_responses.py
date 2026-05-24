import json
from pathlib import Path

from tqdm import tqdm

from ..domain.models import (
    SensitivityCategory,
    SensitivePrompt,
    TargetResponse,
)
from ..judge.llm_judge import LLMJudge
from ..llm.client import OpenAICompatibleClient


INPUT_POST_REFACTOR_RESPONSES_JSONL = (
    "data/intermediate/refactoring/post_refactor_responses/"
    "post_refactor_responses_v1.jsonl"
)

OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL = (
    "data/intermediate/refactoring/post_refactor_evaluations/"
    "post_refactor_evaluations_v1.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

# Deve essere lo stesso judge model usato nello script 05.
JUDGE_MODEL_NAME = "deepseek-r1-distill-qwen-32b"

OVERWRITE_OUTPUT = True

# Prima prova con 3. Se funziona, rimetti None.
LIMIT_RESPONSES = None


CATEGORY_CANONICAL_MAP = {
    "religion and philosophy": SensitivityCategory.RELIGION_PHILOSOPHY,
    "politics and society": SensitivityCategory.POLITICS_SOCIETY,
    "relationships and sentiments": SensitivityCategory.RELATIONSHIPS_SENTIMENTS,
    "health and mental well-being": SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    "identity and diversity": SensitivityCategory.IDENTITY_DIVERSITY,
    "sexual": SensitivityCategory.SEXUAL,
    "security": SensitivityCategory.SECURITY,
    "other": SensitivityCategory.OTHER,
}


def ensure_parent_dirs() -> None:
    Path(OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_output_if_needed() -> None:
    output_path = Path(OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()


def read_dict_jsonl(path: str) -> list[dict]:
    records: list[dict] = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number} in {path}"
                ) from exc

    return records


def normalize_category_to_enum(record: dict) -> SensitivityCategory:
    raw_category = record.get("category")

    if raw_category is None or not str(raw_category).strip():
        return SensitivityCategory.OTHER

    normalized = str(raw_category).strip().lower()

    return CATEGORY_CANONICAL_MAP.get(normalized, SensitivityCategory.OTHER)


def get_refactored_prompt_text(record: dict) -> str:
    """
    Returns the prompt that was actually submitted to the target model.

    Current post-refactor response records should contain refactored_prompt.
    The fallback to prompt_text makes the script more robust.
    """

    refactored_prompt = record.get("refactored_prompt")

    if refactored_prompt is not None and str(refactored_prompt).strip():
        return str(refactored_prompt)

    prompt_text = record.get("prompt_text")

    if prompt_text is not None and str(prompt_text).strip():
        return str(prompt_text)

    raise ValueError(
        "Post-refactor response record does not contain either "
        "'refactored_prompt' or 'prompt_text'."
    )


def get_target_model_name(record: dict) -> str:
    """
    Current post-refactor response records use model_name.
    Baseline TargetResponse uses target_model.
    This function supports both.
    """

    target_model = record.get("target_model")

    if target_model is not None and str(target_model).strip():
        return str(target_model)

    model_name = record.get("model_name")

    if model_name is not None and str(model_name).strip():
        return str(model_name)

    raise ValueError(
        "Post-refactor response record does not contain either "
        "'target_model' or 'model_name'."
    )


def build_sensitive_prompt_from_post_response(record: dict) -> SensitivePrompt:
    """
    Builds the SensitivePrompt object expected by the existing judge.

    The judge must evaluate the response with respect to the actual
    post-refactoring prompt submitted to the target model.
    """

    category = normalize_category_to_enum(record)
    refactored_prompt = get_refactored_prompt_text(record)

    return SensitivePrompt(
        prompt_id=record["prompt_id"],
        text=refactored_prompt,
        sensitive=1,
        categories=[category],
        raw_category=category.value,
        category_present=True,
        source="post_refactor_prompt_v1",
    )


def build_target_response_from_post_response(record: dict) -> TargetResponse:
    """
    Builds the TargetResponse object expected by the existing judge.

    TargetResponse requires:
    - target_model
    - prompt_text
    - response_text
    """

    refactored_prompt = get_refactored_prompt_text(record)
    target_model = get_target_model_name(record)

    return TargetResponse(
        response_id=record["response_id"],
        prompt_id=record["prompt_id"],
        target_model=target_model,
        repetition=int(record["repetition"]),
        prompt_text=refactored_prompt,
        response_text=record["response_text"],
        source=record.get("source", "post_refactor_response_v1"),
    )


def enrich_evaluation_with_refactoring_metadata(
    evaluation,
    response_record: dict,
) -> dict:
    """
    Converts the judge ResponseEvaluation into a JSON-serializable dictionary
    and adds refactoring metadata needed to reconstruct the full episode later.
    """

    if hasattr(evaluation, "model_dump"):
        evaluation_dict = evaluation.model_dump(mode="json")
    else:
        evaluation_dict = evaluation.dict()

    evaluation_dict["source"] = "post_refactor_evaluation_modular_v4"

    evaluation_dict["episode_id"] = response_record.get("episode_id")
    evaluation_dict["refactoring_attempt_id"] = response_record.get(
        "refactoring_attempt_id"
    )
    evaluation_dict["parent_refactoring_attempt_id"] = response_record.get(
        "parent_refactoring_attempt_id"
    )
    evaluation_dict["iteration"] = response_record.get("iteration")
    evaluation_dict["original_prompt"] = response_record.get("original_prompt")
    evaluation_dict["refactored_prompt"] = get_refactored_prompt_text(response_record)
    evaluation_dict["refactoring_method"] = response_record.get("refactoring_method")
    evaluation_dict["input_prompt_source"] = response_record.get("input_prompt_source")
    evaluation_dict["iteration_reason"] = response_record.get("iteration_reason")
    evaluation_dict["category"] = response_record.get("category")

    return evaluation_dict


def append_dict_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def preflight_validate_records(records: list[dict]) -> None:
    """
    Validates that all records can be converted to SensitivePrompt and
    TargetResponse before calling the judge model.

    This prevents wasting LLM calls when there is a schema mismatch.
    """

    failed = []

    for index, record in enumerate(records, start=1):
        try:
            build_sensitive_prompt_from_post_response(record)
            build_target_response_from_post_response(record)
        except Exception as exc:
            failed.append(
                {
                    "index": index,
                    "prompt_id": record.get("prompt_id"),
                    "response_id": record.get("response_id"),
                    "error": str(exc),
                }
            )

    if failed:
        print("\n[ERROR] Preflight validation failed.")
        print("The judge was not called because some records are malformed.")

        for item in failed[:20]:
            print(
                f"  index={item['index']} | "
                f"prompt_id={item['prompt_id']} | "
                f"response_id={item['response_id']} | "
                f"error={item['error']}"
            )

        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more failures.")

        raise ValueError(
            f"Preflight validation failed for {len(failed)} records."
        )


def main() -> None:
    ensure_parent_dirs()
    remove_existing_output_if_needed()

    response_records = read_dict_jsonl(INPUT_POST_REFACTOR_RESPONSES_JSONL)

    total_available_responses = len(response_records)

    if LIMIT_RESPONSES is not None:
        selected_response_records = response_records[:LIMIT_RESPONSES]
    else:
        selected_response_records = response_records

    preflight_validate_records(selected_response_records)

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=JUDGE_MODEL_NAME,
        timeout=240,
    )

    judge = LLMJudge(
        client=client,
        model_name=JUDGE_MODEL_NAME,
    )

    print("\n" + "=" * 90)
    print("POST-REFACTOR RESPONSE EVALUATION - MODULAR JUDGE V4")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Post-refactor responses:   {INPUT_POST_REFACTOR_RESPONSES_JSONL}")
    print(f"  Output evaluations:        {OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL}")

    print("\nJudge configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Judge model:               {JUDGE_MODEL_NAME}")
    print(f"  Overwrite output:          {OVERWRITE_OUTPUT}")
    print(f"  Limit responses:           {LIMIT_RESPONSES}")

    print("\nDataset:")
    print(f"  Total responses:           {total_available_responses}")
    print(f"  Responses selected:        {len(selected_response_records)}")

    print("\n" + "-" * 90)

    generated = 0
    failed = 0
    failed_items: list[dict] = []

    for response_record in tqdm(
        selected_response_records,
        desc="Evaluating post-refactor responses",
    ):
        try:
            prompt = build_sensitive_prompt_from_post_response(response_record)
            response = build_target_response_from_post_response(response_record)

            evaluation = judge.evaluate_baseline_response(
                prompt=prompt,
                response=response,
            )

            enriched_evaluation = enrich_evaluation_with_refactoring_metadata(
                evaluation=evaluation,
                response_record=response_record,
            )

            append_dict_jsonl(
                OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL,
                enriched_evaluation,
            )

            generated += 1

        except Exception as exc:
            failed += 1

            failed_items.append(
                {
                    "prompt_id": response_record.get("prompt_id"),
                    "response_id": response_record.get("response_id"),
                    "refactoring_attempt_id": response_record.get(
                        "refactoring_attempt_id"
                    ),
                    "repetition": response_record.get("repetition"),
                    "error": str(exc),
                }
            )

            print()
            print("[ERROR] Failed to evaluate post-refactor response")
            print(f"  prompt_id:                {response_record.get('prompt_id')}")
            print(f"  response_id:              {response_record.get('response_id')}")
            print(
                f"  refactoring_attempt_id:   "
                f"{response_record.get('refactoring_attempt_id')}"
            )
            print(f"  repetition:               {response_record.get('repetition')}")
            print(f"  error:                    {exc}")

    print("\n" + "=" * 90)
    print("POST-REFACTOR RESPONSE EVALUATION COMPLETED")
    print("=" * 90)

    print(f"\nGenerated evaluations:       {generated}")
    print(f"Failed evaluations:          {failed}")
    print(f"Output file:                 {OUTPUT_POST_REFACTOR_EVALUATIONS_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:50]:
            print(
                f"  prompt_id={item['prompt_id']} | "
                f"response_id={item['response_id']} | "
                f"attempt_id={item['refactoring_attempt_id']} | "
                f"repetition={item['repetition']} | "
                f"error={item['error']}"
            )

    if generated == len(selected_response_records):
        print("\nPost-refactor response evaluation completed successfully.")
        print("Next step: compare baseline and post-refactoring evaluations.")
    else:
        print("\n[WARNING] Some evaluations failed. Inspect the errors above.")


if __name__ == "__main__":
    main()