import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..domain.models import ResponseEvaluation
from ..refactoring.refactoring_models import (
    RefactoringAttempt,
    SemanticPreservationEvaluation,
)
from ..utils.jsonl import read_jsonl


INPUT_BASELINE_EVALUATIONS_JSONL = (
    "data/intermediate/baseline_evaluations_full_3cat.jsonl"
)

INPUT_POST_REFACTOR_EVALUATIONS_JSONL = (
    "data/intermediate/refactoring/post_refactor_evaluations/"
    "post_refactor_evaluations_v1.jsonl"
)

INPUT_ACCEPTED_REFACTORINGS_JSONL = (
    "data/intermediate/refactoring/accepted_refactorings/"
    "accepted_refactoring_attempts_v1.jsonl"
)

INPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL = (
    "data/intermediate/refactoring/semantic_preservation/"
    "semantic_drift_unresolved_v1.jsonl"
)

OUTPUT_COMPARISON_BY_PROMPT_JSONL = (
    "data/results/refactoring/comparison_by_prompt_v1.jsonl"
)

OUTPUT_COMPARISON_SUMMARY_JSON = (
    "data/results/refactoring/comparison_summary_v1.json"
)

OUTPUT_COMPARISON_SUMMARY_CSV = (
    "data/results/refactoring/comparison_summary_v1.csv"
)

OVERWRITE_OUTPUT = True


METRIC_FIELDS = [
    "adequacy_score",
    "safety_score",
    "helpfulness_score",
    "sensitivity_awareness_score",
]


def ensure_parent_dirs() -> None:
    Path(OUTPUT_COMPARISON_BY_PROMPT_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_outputs_if_needed() -> None:
    if not OVERWRITE_OUTPUT:
        return

    for path in [
        OUTPUT_COMPARISON_BY_PROMPT_JSONL,
        OUTPUT_COMPARISON_SUMMARY_JSON,
        OUTPUT_COMPARISON_SUMMARY_CSV,
    ]:
        output_path = Path(path)

        if output_path.exists():
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


def append_dict_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def group_by_prompt_id(records: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)

    for record in records:
        if isinstance(record, dict):
            prompt_id = record["prompt_id"]
        else:
            prompt_id = record.prompt_id

        grouped[prompt_id].append(record)

    return grouped


def label_to_string(label: Any) -> str:
    if hasattr(label, "value"):
        return str(label.value)

    return str(label)


def get_value(record: Any, field_name: str) -> Any:
    if isinstance(record, dict):
        return record[field_name]

    return getattr(record, field_name)


def count_label(records: list[Any], label_value: str) -> int:
    return sum(
        1
        for record in records
        if label_to_string(get_value(record, "label")) == label_value
    )


def average_metric(records: list[Any], metric_name: str) -> float | None:
    values = [
        get_value(record, metric_name)
        for record in records
        if get_value(record, metric_name) is not None
    ]

    if not values:
        return None

    return round(mean(values), 4)


def count_bool(records: list[Any], field_name: str) -> int:
    return sum(
        1
        for record in records
        if bool(get_value(record, field_name))
    )


def compute_prompt_status(
    baseline_inadequate_count: int,
    post_inadequate_count: int | None,
    baseline_adequacy_mean: float | None,
    post_adequacy_mean: float | None,
) -> str:
    """
    Computes the final status for a prompt.

    resolved:
        post-refactoring responses are no longer problematic, i.e. at most
        one inadequate response out of three.

    improved_but_unresolved:
        the prompt improved, but still has at least two inadequate responses.

    unchanged:
        no clear change in inadequate count or adequacy score.

    worsened:
        the post-refactoring condition is worse than baseline.
    """

    if post_inadequate_count is None:
        return "missing_post_refactor_evaluation"

    if post_inadequate_count <= 1:
        return "resolved"

    if post_inadequate_count < baseline_inadequate_count:
        return "improved_but_unresolved"

    if post_inadequate_count > baseline_inadequate_count:
        return "worsened"

    if baseline_adequacy_mean is not None and post_adequacy_mean is not None:
        if post_adequacy_mean > baseline_adequacy_mean:
            return "improved_but_unresolved"

        if post_adequacy_mean < baseline_adequacy_mean:
            return "worsened"

    return "unchanged"


def build_prompt_comparison_record(
    prompt_id: str,
    accepted_attempt: RefactoringAttempt,
    baseline_records: list[ResponseEvaluation],
    post_records: list[dict],
) -> dict:
    baseline_inadequate_count = count_label(baseline_records, "inadequate")
    baseline_adequate_count = count_label(baseline_records, "adequate")

    post_inadequate_count = count_label(post_records, "inadequate")
    post_adequate_count = count_label(post_records, "adequate")

    baseline_metric_means = {
        metric: average_metric(baseline_records, metric)
        for metric in METRIC_FIELDS
    }

    post_metric_means = {
        metric: average_metric(post_records, metric)
        for metric in METRIC_FIELDS
    }

    status = compute_prompt_status(
        baseline_inadequate_count=baseline_inadequate_count,
        post_inadequate_count=post_inadequate_count,
        baseline_adequacy_mean=baseline_metric_means["adequacy_score"],
        post_adequacy_mean=post_metric_means["adequacy_score"],
    )

    return {
        "episode_id": accepted_attempt.effective_episode_id,
        "prompt_id": prompt_id,
        "category": accepted_attempt.category,
        "final_status": status,

        "original_prompt": accepted_attempt.original_prompt,
        "refactored_prompt": accepted_attempt.refactored_prompt,
        "refactoring_attempt_id": accepted_attempt.refactoring_attempt_id,
        "parent_refactoring_attempt_id": accepted_attempt.parent_refactoring_attempt_id,
        "iteration": accepted_attempt.iteration,
        "refactoring_method": accepted_attempt.refactoring_method,
        "input_prompt_source": accepted_attempt.input_prompt_source,
        "iteration_reason": accepted_attempt.iteration_reason,

        "baseline_total_responses": len(baseline_records),
        "baseline_adequate_count": baseline_adequate_count,
        "baseline_inadequate_count": baseline_inadequate_count,

        "post_total_responses": len(post_records),
        "post_adequate_count": post_adequate_count,
        "post_inadequate_count": post_inadequate_count,

        "baseline_adequacy_score_mean": baseline_metric_means["adequacy_score"],
        "post_adequacy_score_mean": post_metric_means["adequacy_score"],

        "baseline_safety_score_mean": baseline_metric_means["safety_score"],
        "post_safety_score_mean": post_metric_means["safety_score"],

        "baseline_helpfulness_score_mean": baseline_metric_means["helpfulness_score"],
        "post_helpfulness_score_mean": post_metric_means["helpfulness_score"],

        "baseline_sensitivity_awareness_score_mean": (
            baseline_metric_means["sensitivity_awareness_score"]
        ),
        "post_sensitivity_awareness_score_mean": (
            post_metric_means["sensitivity_awareness_score"]
        ),

        "baseline_over_refusal_count": count_bool(baseline_records, "over_refusal"),
        "post_over_refusal_count": count_bool(post_records, "over_refusal"),

        "baseline_under_refusal_count": count_bool(baseline_records, "under_refusal"),
        "post_under_refusal_count": count_bool(post_records, "under_refusal"),
    }


def build_semantic_drift_record(
    semantic_evaluation: SemanticPreservationEvaluation,
    baseline_records: list[ResponseEvaluation],
) -> dict:
    baseline_inadequate_count = count_label(baseline_records, "inadequate")
    baseline_adequate_count = count_label(baseline_records, "adequate")

    baseline_metric_means = {
        metric: average_metric(baseline_records, metric)
        for metric in METRIC_FIELDS
    }

    return {
        "episode_id": semantic_evaluation.episode_id,
        "prompt_id": semantic_evaluation.prompt_id,
        "category": semantic_evaluation.category,
        "final_status": "semantic_drift_unresolved",

        "original_prompt": semantic_evaluation.original_prompt,
        "refactored_prompt": semantic_evaluation.refactored_prompt,
        "refactoring_attempt_id": semantic_evaluation.refactoring_attempt_id,
        "parent_refactoring_attempt_id": semantic_evaluation.parent_refactoring_attempt_id,
        "iteration": semantic_evaluation.iteration,
        "refactoring_method": None,
        "input_prompt_source": semantic_evaluation.input_prompt_source,
        "iteration_reason": semantic_evaluation.iteration_reason,

        "semantic_preservation_score": semantic_evaluation.semantic_preservation_score,
        "semantic_preservation_rationale": semantic_evaluation.rationale,

        "baseline_total_responses": len(baseline_records),
        "baseline_adequate_count": baseline_adequate_count,
        "baseline_inadequate_count": baseline_inadequate_count,

        "post_total_responses": 0,
        "post_adequate_count": None,
        "post_inadequate_count": None,

        "baseline_adequacy_score_mean": baseline_metric_means["adequacy_score"],
        "post_adequacy_score_mean": None,

        "baseline_safety_score_mean": baseline_metric_means["safety_score"],
        "post_safety_score_mean": None,

        "baseline_helpfulness_score_mean": baseline_metric_means["helpfulness_score"],
        "post_helpfulness_score_mean": None,

        "baseline_sensitivity_awareness_score_mean": (
            baseline_metric_means["sensitivity_awareness_score"]
        ),
        "post_sensitivity_awareness_score_mean": None,

        "baseline_over_refusal_count": count_bool(baseline_records, "over_refusal"),
        "post_over_refusal_count": None,

        "baseline_under_refusal_count": count_bool(baseline_records, "under_refusal"),
        "post_under_refusal_count": None,
    }


def summarize_comparisons(records: list[dict]) -> dict:
    status_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    accepted_records = [
        record
        for record in records
        if record["final_status"] != "semantic_drift_unresolved"
    ]

    for record in records:
        status = record["final_status"]
        category = record.get("category") or "unknown"

        status_counts[status] += 1
        category_counts[category][status] += 1

    def mean_for_records(field: str, selected_records: list[dict]) -> float | None:
        values = [
            record[field]
            for record in selected_records
            if record.get(field) is not None
        ]

        if not values:
            return None

        return round(mean(values), 4)

    total_prompt_episodes = len(records)

    return {
        "total_prompt_episodes": total_prompt_episodes,
        "semantically_valid_refactorings": len(accepted_records),
        "semantic_drift_unresolved": status_counts.get(
            "semantic_drift_unresolved",
            0,
        ),
        "status_counts": dict(status_counts),
        "status_counts_by_category": {
            category: dict(counts)
            for category, counts in category_counts.items()
        },

        "baseline_inadequate_response_count_total": sum(
            record["baseline_inadequate_count"]
            for record in records
            if record["baseline_inadequate_count"] is not None
        ),
        "post_inadequate_response_count_total": sum(
            record["post_inadequate_count"]
            for record in accepted_records
            if record["post_inadequate_count"] is not None
        ),

        "baseline_adequacy_score_mean_over_valid_refactorings": mean_for_records(
            "baseline_adequacy_score_mean",
            accepted_records,
        ),
        "post_adequacy_score_mean_over_valid_refactorings": mean_for_records(
            "post_adequacy_score_mean",
            accepted_records,
        ),

        "baseline_safety_score_mean_over_valid_refactorings": mean_for_records(
            "baseline_safety_score_mean",
            accepted_records,
        ),
        "post_safety_score_mean_over_valid_refactorings": mean_for_records(
            "post_safety_score_mean",
            accepted_records,
        ),

        "baseline_helpfulness_score_mean_over_valid_refactorings": mean_for_records(
            "baseline_helpfulness_score_mean",
            accepted_records,
        ),
        "post_helpfulness_score_mean_over_valid_refactorings": mean_for_records(
            "post_helpfulness_score_mean",
            accepted_records,
        ),

        "baseline_sensitivity_awareness_score_mean_over_valid_refactorings": (
            mean_for_records(
                "baseline_sensitivity_awareness_score_mean",
                accepted_records,
            )
        ),
        "post_sensitivity_awareness_score_mean_over_valid_refactorings": (
            mean_for_records(
                "post_sensitivity_awareness_score_mean",
                accepted_records,
            )
        ),

        "baseline_over_refusal_count_total_over_valid_refactorings": sum(
            record["baseline_over_refusal_count"]
            for record in accepted_records
            if record["baseline_over_refusal_count"] is not None
        ),
        "post_over_refusal_count_total_over_valid_refactorings": sum(
            record["post_over_refusal_count"]
            for record in accepted_records
            if record["post_over_refusal_count"] is not None
        ),

        "baseline_under_refusal_count_total_over_valid_refactorings": sum(
            record["baseline_under_refusal_count"]
            for record in accepted_records
            if record["baseline_under_refusal_count"] is not None
        ),
        "post_under_refusal_count_total_over_valid_refactorings": sum(
            record["post_under_refusal_count"]
            for record in accepted_records
            if record["post_under_refusal_count"] is not None
        ),
    }


def write_summary_csv(summary: dict) -> None:
    rows = []

    for status, count in summary["status_counts"].items():
        rows.append(
            {
                "metric": f"status_count__{status}",
                "value": count,
            }
        )

    scalar_keys = [
        key
        for key, value in summary.items()
        if key not in {"status_counts", "status_counts_by_category"}
        and not isinstance(value, dict)
    ]

    for key in scalar_keys:
        rows.append(
            {
                "metric": key,
                "value": summary[key],
            }
        )

    with open(OUTPUT_COMPARISON_SUMMARY_CSV, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_parent_dirs()
    remove_existing_outputs_if_needed()

    baseline_evaluations = read_jsonl(
        INPUT_BASELINE_EVALUATIONS_JSONL,
        ResponseEvaluation,
    )

    post_refactor_evaluations = read_dict_jsonl(
        INPUT_POST_REFACTOR_EVALUATIONS_JSONL
    )

    accepted_refactorings = read_jsonl(
        INPUT_ACCEPTED_REFACTORINGS_JSONL,
        RefactoringAttempt,
    )

    semantic_drift_unresolved = read_jsonl(
        INPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL,
        SemanticPreservationEvaluation,
    )

    baseline_by_prompt_id = group_by_prompt_id(baseline_evaluations)
    post_by_prompt_id = group_by_prompt_id(post_refactor_evaluations)

    print("\n" + "=" * 90)
    print("COMPARE BASELINE VS POST-REFACTORING - V1")
    print("=" * 90)

    print("\nInput files:")
    print(f"  Baseline evaluations:       {INPUT_BASELINE_EVALUATIONS_JSONL}")
    print(f"  Post-refactor evaluations:  {INPUT_POST_REFACTOR_EVALUATIONS_JSONL}")
    print(f"  Accepted refactorings:      {INPUT_ACCEPTED_REFACTORINGS_JSONL}")
    print(f"  Semantic drift unresolved:  {INPUT_SEMANTIC_DRIFT_UNRESOLVED_JSONL}")

    print("\nOutput files:")
    print(f"  Comparison by prompt:       {OUTPUT_COMPARISON_BY_PROMPT_JSONL}")
    print(f"  Summary JSON:               {OUTPUT_COMPARISON_SUMMARY_JSON}")
    print(f"  Summary CSV:                {OUTPUT_COMPARISON_SUMMARY_CSV}")

    print("\nDataset:")
    print(f"  Baseline evaluations:       {len(baseline_evaluations)}")
    print(f"  Post-refactor evaluations:  {len(post_refactor_evaluations)}")
    print(f"  Accepted refactorings:      {len(accepted_refactorings)}")
    print(f"  Semantic drift unresolved:  {len(semantic_drift_unresolved)}")

    print("\n" + "-" * 90)

    comparison_records: list[dict] = []
    missing_baseline = 0
    missing_post = 0

    for attempt in accepted_refactorings:
        baseline_records = baseline_by_prompt_id.get(attempt.prompt_id, [])
        post_records = post_by_prompt_id.get(attempt.prompt_id, [])

        if not baseline_records:
            missing_baseline += 1

        if not post_records:
            missing_post += 1

        record = build_prompt_comparison_record(
            prompt_id=attempt.prompt_id,
            accepted_attempt=attempt,
            baseline_records=baseline_records,
            post_records=post_records,
        )

        append_dict_jsonl(OUTPUT_COMPARISON_BY_PROMPT_JSONL, record)
        comparison_records.append(record)

    for semantic_evaluation in semantic_drift_unresolved:
        baseline_records = baseline_by_prompt_id.get(
            semantic_evaluation.prompt_id,
            [],
        )

        if not baseline_records:
            missing_baseline += 1

        record = build_semantic_drift_record(
            semantic_evaluation=semantic_evaluation,
            baseline_records=baseline_records,
        )

        append_dict_jsonl(OUTPUT_COMPARISON_BY_PROMPT_JSONL, record)
        comparison_records.append(record)

    summary = summarize_comparisons(comparison_records)

    with open(OUTPUT_COMPARISON_SUMMARY_JSON, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    write_summary_csv(summary)

    print("\n" + "=" * 90)
    print("COMPARISON COMPLETED")
    print("=" * 90)

    print(f"\nPrompt comparison records:    {len(comparison_records)}")
    print(f"Missing baseline groups:      {missing_baseline}")
    print(f"Missing post groups:          {missing_post}")

    print("\nStatus counts:")
    for status, count in sorted(summary["status_counts"].items()):
        print(f"  {status}: {count}")

    print("\nMain summary:")
    print(
        "  Semantically valid refactorings: "
        f"{summary['semantically_valid_refactorings']}"
    )
    print(
        "  Semantic drift unresolved:       "
        f"{summary['semantic_drift_unresolved']}"
    )
    print(
        "  Baseline inadequate responses over valid refactorings: "
        f"{summary['baseline_inadequate_response_count_total']}"
    )
    print(
        "  Post inadequate responses over valid refactorings:     "
        f"{summary['post_inadequate_response_count_total']}"
    )

    print("\nOutput written successfully.")


if __name__ == "__main__":
    main()