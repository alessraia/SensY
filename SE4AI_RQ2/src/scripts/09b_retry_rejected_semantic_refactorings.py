from pathlib import Path

from tqdm import tqdm

from ..llm.client import OpenAICompatibleClient
from ..refactoring.knowledge_base import load_refactoring_knowledge_base
from ..refactoring.refactoring_models import (
    RefactoringAttempt,
    SemanticPreservationEvaluation,
)
from ..refactoring.semantic_retry_refactorer import SemanticRetryRefactorer
from ..utils.jsonl import append_jsonl, read_jsonl


INPUT_REFACTORING_ATTEMPTS_JSONL = (
    "data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl"
)

INPUT_REJECTED_SEMANTIC_PRESERVATION_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_preservation_v1_rejected.jsonl"
)

OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL = (
    "data/intermediate/refactoring/attempts/"
    "refactoring_attempts_v1_semantic_retry.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

REFACTORING_MODEL_NAME = "qwen/qwen3-14b"

OVERWRITE_OUTPUT = True
LIMIT_REJECTED = None


def ensure_parent_dirs() -> None:
    Path(OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_output_if_needed() -> None:
    output_path = Path(OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()


def main() -> None:
    ensure_parent_dirs()
    remove_existing_output_if_needed()

    attempts = read_jsonl(INPUT_REFACTORING_ATTEMPTS_JSONL, RefactoringAttempt)

    rejected_evaluations = read_jsonl(
        INPUT_REJECTED_SEMANTIC_PRESERVATION_JSONL,
        SemanticPreservationEvaluation,
    )

    if LIMIT_REJECTED is not None:
        selected_rejected_evaluations = rejected_evaluations[:LIMIT_REJECTED]
    else:
        selected_rejected_evaluations = rejected_evaluations

    attempts_by_id = {
        attempt.refactoring_attempt_id: attempt
        for attempt in attempts
    }

    knowledge_base = load_refactoring_knowledge_base()

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=REFACTORING_MODEL_NAME,
        timeout=180,
    )

    retry_refactorer = SemanticRetryRefactorer(
        client=client,
        model_name=REFACTORING_MODEL_NAME,
        knowledge_base=knowledge_base,
        temperature=0.1,
        max_tokens=1400,
    )

    print("\n" + "=" * 90)
    print("SEMANTIC RETRY FOR REJECTED REFACTORINGS - V1")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Refactoring attempts:      {INPUT_REFACTORING_ATTEMPTS_JSONL}")
    print(f"  Rejected evaluations:      {INPUT_REJECTED_SEMANTIC_PRESERVATION_JSONL}")
    print(f"  Output retry attempts:     {OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL}")

    print("\nRetry configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Refactoring model:         {REFACTORING_MODEL_NAME}")
    print(f"  Temperature:               0.1")
    print(f"  Max tokens:                1400")
    print(f"  Overwrite output:          {OVERWRITE_OUTPUT}")
    print(f"  Limit rejected:            {LIMIT_REJECTED}")

    print("\nDataset:")
    print(f"  Total rejected:            {len(rejected_evaluations)}")
    print(f"  Selected rejected:         {len(selected_rejected_evaluations)}")

    print("\n" + "-" * 90)

    generated = 0
    failed = 0
    failed_items: list[dict] = []

    for semantic_evaluation in tqdm(
        selected_rejected_evaluations,
        desc="Retrying rejected refactorings",
    ):
        failed_attempt = attempts_by_id.get(
            semantic_evaluation.refactoring_attempt_id
        )

        if failed_attempt is None:
            failed += 1
            failed_items.append(
                {
                    "prompt_id": semantic_evaluation.prompt_id,
                    "refactoring_attempt_id": semantic_evaluation.refactoring_attempt_id,
                    "error": "Parent refactoring attempt not found.",
                }
            )
            continue

        try:
            retry_attempt = retry_refactorer.retry(
                failed_attempt=failed_attempt,
                semantic_evaluation=semantic_evaluation,
            )

            append_jsonl(OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL, retry_attempt)
            generated += 1

        except Exception as exc:
            failed += 1
            failed_items.append(
                {
                    "prompt_id": semantic_evaluation.prompt_id,
                    "refactoring_attempt_id": semantic_evaluation.refactoring_attempt_id,
                    "error": str(exc),
                }
            )

            print()
            print("[ERROR] Failed to retry rejected refactoring")
            print(f"  prompt_id:              {semantic_evaluation.prompt_id}")
            print(f"  refactoring_attempt_id: {semantic_evaluation.refactoring_attempt_id}")
            print(f"  error:                  {exc}")

    print("\n" + "=" * 90)
    print("SEMANTIC RETRY COMPLETED")
    print("=" * 90)

    print(f"\nGenerated retry attempts:    {generated}")
    print(f"Failed retry attempts:       {failed}")
    print(f"Output file:                 {OUTPUT_SEMANTIC_RETRY_ATTEMPTS_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:50]:
            print(
                f"  prompt_id={item['prompt_id']} | "
                f"attempt_id={item['refactoring_attempt_id']} | "
                f"error={item['error']}"
            )

    if generated == len(selected_rejected_evaluations):
        print("\nSemantic retry completed successfully.")
        print("Next step: run semantic preservation on retry attempts.")
    else:
        print("\n[WARNING] Some retry attempts failed. Inspect the errors above.")


if __name__ == "__main__":
    main()