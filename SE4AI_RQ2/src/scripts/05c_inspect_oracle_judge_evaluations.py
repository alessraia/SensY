from collections import Counter

from ..domain.models import OracleJudgeEvaluation
from ..utils.jsonl import read_jsonl


INPUT_JSONL = "data/oracle/judge_oracle_evaluations.jsonl"


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def print_counter(title: str, values: list) -> None:
    print(format_subsection(title))

    counts = Counter(values)

    if not counts:
        print("No data.")
        return

    max_label_len = max(len(str(key)) for key in counts.keys())
    max_count_len = max(len(str(value)) for value in counts.values())

    for key, count in counts.most_common():
        print(f"{str(key):<{max_label_len}}  {str(count):>{max_count_len}}")


def average(values: list[int]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def main() -> None:
    evaluations = read_jsonl(INPUT_JSONL, OracleJudgeEvaluation)

    print(format_section("ORACLE JUDGE EVALUATIONS INSPECTION"))

    print("\nInput file:")
    print(f"  {INPUT_JSONL}")

    print(format_subsection("BASIC COUNTS"))

    print(f"Evaluations loaded:              {len(evaluations)}")

    if not evaluations:
        print("\nNo evaluations found.")
        return

    oracle_ids = [item.oracle_id for item in evaluations]
    source_models = [item.source_model for item in evaluations]
    manual_labels = [item.manual_label.value for item in evaluations]
    judge_labels = [item.judge_label.value for item in evaluations]
    judge_models = [item.judge_model for item in evaluations]

    print(f"Unique oracle_ids:               {len(set(oracle_ids))}")
    print(f"Source models:                   {sorted(set(source_models))}")
    print(f"Judge models:                    {sorted(set(judge_models))}")

    print_counter("MANUAL LABEL DISTRIBUTION", manual_labels)
    print_counter("JUDGE LABEL DISTRIBUTION", judge_labels)
    print_counter("SOURCE MODEL DISTRIBUTION", source_models)

    adequacy_scores = [item.adequacy_score for item in evaluations]
    safety_scores = [item.safety_score for item in evaluations]
    helpfulness_scores = [item.helpfulness_score for item in evaluations]
    sensitivity_scores = [
        item.sensitivity_awareness_score for item in evaluations
    ]

    print_counter("ADEQUACY SCORE DISTRIBUTION", adequacy_scores)
    print_counter("SAFETY SCORE DISTRIBUTION", safety_scores)
    print_counter("HELPFULNESS SCORE DISTRIBUTION", helpfulness_scores)
    print_counter("SENSITIVITY AWARENESS SCORE DISTRIBUTION", sensitivity_scores)

    print(format_subsection("AVERAGE SCORES"))
    print(f"Average adequacy score:              {average(adequacy_scores):.2f}")
    print(f"Average safety score:                {average(safety_scores):.2f}")
    print(f"Average helpfulness score:           {average(helpfulness_scores):.2f}")
    print(f"Average sensitivity awareness score: {average(sensitivity_scores):.2f}")

    invalid_scores = [
        item
        for item in evaluations
        if not (1 <= item.adequacy_score <= 5)
        or not (1 <= item.safety_score <= 5)
        or not (1 <= item.helpfulness_score <= 5)
        or not (1 <= item.sensitivity_awareness_score <= 5)
    ]

    empty_rationales = [
        item
        for item in evaluations
        if not item.rationale.strip()
    ]

    inconsistent_label_score = [
        item
        for item in evaluations
        if (
            item.adequacy_score >= 4
            and item.judge_label.value == "inadequate"
        )
        or (
            item.adequacy_score <= 2
            and item.judge_label.value == "adequate"
        )
    ]

    print(format_subsection("QUALITY CHECKS"))
    print(f"Invalid score ranges:                {len(invalid_scores)}")
    print(f"Empty rationales:                    {len(empty_rationales)}")
    print(f"Inconsistent label/score cases:      {len(inconsistent_label_score)}")

    print(format_subsection("FIRST 5 EVALUATIONS"))

    for i, item in enumerate(evaluations[:5], start=1):
        response_preview = item.response_text

        if len(response_preview) > 250:
            response_preview = response_preview[:250] + "..."

        print(f"\n[{i}]")
        print(f"oracle_evaluation_id:          {item.oracle_evaluation_id}")
        print(f"oracle_id:                     {item.oracle_id}")
        print(f"source_model:                  {item.source_model}")
        print(f"manual_label:                  {item.manual_label.value}")
        print(f"judge_label:                   {item.judge_label.value}")
        print(f"adequacy_score:                {item.adequacy_score}")
        print(f"safety_score:                  {item.safety_score}")
        print(f"helpfulness_score:             {item.helpfulness_score}")
        print(f"sensitivity_awareness_score:   {item.sensitivity_awareness_score}")
        print(f"over_refusal:                  {item.over_refusal}")
        print(f"under_refusal:                 {item.under_refusal}")
        print(f"failure_modes:                 {item.failure_modes}")
        print(f"rationale:                     {item.rationale}")
        print(f"question:                      {item.question}")
        print(f"response_text:                 {response_preview}")

    print(format_section("INSPECTION COMPLETED"))


if __name__ == "__main__":
    main()