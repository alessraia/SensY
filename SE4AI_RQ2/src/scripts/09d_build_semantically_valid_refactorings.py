from pathlib import Path

from ..refactoring.refactoring_models import (
    RefactoringAttempt,
    SemanticPreservationEvaluation,
)
from ..utils.jsonl import append_jsonl, read_jsonl


INPUT_REFACTORING_ATTEMPTS_V1_JSONL = (
    "data/intermediate/refactoring/attempts/"
    "refactoring_attempts_v1.jsonl"
)

INPUT_REFACTORING_ATTEMPTS_RETRY_JSONL = (
    "data/intermediate/refactoring/attempts/"
    "refactoring_attempts_v1_semantic_retry.jsonl"
)

INPUT_SEMANTIC_V1_ACCEPTED_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_preservation_v1_accepted.jsonl"
)

INPUT_SEMANTIC_RETRY_ACCEPTED_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_preservation_v1_semantic_retry_accepted.jsonl"
)

INPUT_SEMANTIC_RETRY_REJECTED_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_preservation_v1_semantic_retry_rejected.jsonl"
)

OUTPUT_ACCEPTED_REFACTORINGS_JSONL = (
    "data/intermediate/refactoring/accepted_refactorings/"
    "accepted_refactoring_attempts_v1.jsonl"
)

OUTPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_drift_unresolved_v1.jsonl"
)

OVERWRITE_OUTPUT = True


def ensure_parent_dirs() -> None:
    Path(OUTPUT_ACCEPTED_REFACTORINGS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(OUTPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_outputs_if_needed() -> None:
    if not OVERWRITE_OUTPUT:
        return

    for output_path in [
        Path(OUTPUT_ACCEPTED_REFACTORINGS_JSONL),
        Path(OUTPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL),
    ]:
        if output_path.exists():
            output_path.unlink()


def load_attempts_by_id(
    attempts_path: str,
) -> dict[str, RefactoringAttempt]:
    attempts = read_jsonl(attempts_path, RefactoringAttempt)

    return {
        attempt.refactoring_attempt_id: attempt
        for attempt in attempts
    }


def main() -> None:
    ensure_parent_dirs()
    remove_existing_outputs_if_needed()

    attempts_v1_by_id = load_attempts_by_id(INPUT_REFACTORING_ATTEMPTS_V1_JSONL)
    retry_attempts_by_id = load_attempts_by_id(INPUT_REFACTORING_ATTEMPTS_RETRY_JSONL)

    semantic_v1_accepted = read_jsonl(
        INPUT_SEMANTIC_V1_ACCEPTED_JSONL,
        SemanticPreservationEvaluation,
    )

    semantic_retry_accepted = read_jsonl(
        INPUT_SEMANTIC_RETRY_ACCEPTED_JSONL,
        SemanticPreservationEvaluation,
    )

    semantic_retry_rejected = read_jsonl(
        INPUT_SEMANTIC_RETRY_REJECTED_JSONL,
        SemanticPreservationEvaluation,
    )

    print("\n" + "=" * 90)
    print("BUILD SEMANTICALLY VALID REFACTORINGS - V1")
    print("=" * 90)

    print("\nInput files:")
    print(f"  Refactoring attempts v1:       {INPUT_REFACTORING_ATTEMPTS_V1_JSONL}")
    print(f"  Refactoring attempts retry:    {INPUT_REFACTORING_ATTEMPTS_RETRY_JSONL}")
    print(f"  Semantic v1 accepted:          {INPUT_SEMANTIC_V1_ACCEPTED_JSONL}")
    print(f"  Semantic retry accepted:       {INPUT_SEMANTIC_RETRY_ACCEPTED_JSONL}")
    print(f"  Semantic retry rejected:       {INPUT_SEMANTIC_RETRY_REJECTED_JSONL}")

    print("\nOutput files:")
    print(f"  Accepted refactorings:         {OUTPUT_ACCEPTED_REFACTORINGS_JSONL}")
    print(f"  Semantic drift unresolved:     {OUTPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL}")

    print("\n" + "-" * 90)

    accepted_written = 0
    unresolved_written = 0
    missing_attempts = 0

    selected_prompt_ids: set[str] = set()

    for semantic_evaluation in semantic_v1_accepted:
        attempt = attempts_v1_by_id.get(
            semantic_evaluation.refactoring_attempt_id
        )

        if attempt is None:
            missing_attempts += 1
            print(
                "[WARNING] Missing v1 attempt for semantic evaluation: "
                f"{semantic_evaluation.refactoring_attempt_id}"
            )
            continue

        append_jsonl(OUTPUT_ACCEPTED_REFACTORINGS_JSONL, attempt)
        selected_prompt_ids.add(attempt.prompt_id)
        accepted_written += 1

    for semantic_evaluation in semantic_retry_accepted:
        attempt = retry_attempts_by_id.get(
            semantic_evaluation.refactoring_attempt_id
        )

        if attempt is None:
            missing_attempts += 1
            print(
                "[WARNING] Missing retry attempt for semantic evaluation: "
                f"{semantic_evaluation.refactoring_attempt_id}"
            )
            continue

        if attempt.prompt_id in selected_prompt_ids:
            print(
                "[WARNING] Prompt already selected from v1 accepted; "
                f"skipping retry duplicate: {attempt.prompt_id}"
            )
            continue

        append_jsonl(OUTPUT_ACCEPTED_REFACTORINGS_JSONL, attempt)
        selected_prompt_ids.add(attempt.prompt_id)
        accepted_written += 1

    for semantic_evaluation in semantic_retry_rejected:
        append_jsonl(OUTPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL, semantic_evaluation)
        unresolved_written += 1

    print("\n" + "=" * 90)
    print("SEMANTICALLY VALID REFACTORINGS BUILT")
    print("=" * 90)

    print(f"\nAccepted v1 evaluations:         {len(semantic_v1_accepted)}")
    print(f"Accepted retry evaluations:      {len(semantic_retry_accepted)}")
    print(f"Retry rejected evaluations:      {len(semantic_retry_rejected)}")
    print(f"Accepted refactorings written:   {accepted_written}")
    print(f"Semantic drift unresolved:       {unresolved_written}")
    print(f"Missing attempts:                {missing_attempts}")
    print(f"Unique prompt IDs selected:      {len(selected_prompt_ids)}")

    if accepted_written == len(selected_prompt_ids):
        print("\nDataset built successfully.")

    if unresolved_written > 0:
        print(
            "\nSome prompts remain semantically unresolved after one retry. "
            "They will not be sent to the target model in the current experiment."
        )


if __name__ == "__main__":
    main()