from collections import Counter

from ..domain.models import ResponseEvaluation
from ..utils.jsonl import read_jsonl


#INPUT_JSONL = "data/intermediate/baseline_evaluations_pilot.jsonl"
INPUT_JSONL = "data/intermediate/baseline_evaluations_full_3cat.jsonl"


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
    evaluations = read_jsonl(INPUT_JSONL, ResponseEvaluation)

    print(format_section("BASELINE EVALUATIONS INSPECTION"))

    print("\nInput file:")
    print(f"  {INPUT_JSONL}")

    print(format_subsection("BASIC COUNTS"))
    print(f"Evaluations loaded:              {len(evaluations)}")

    if not evaluations:
        print("\nNo evaluations found.")
        return

    prompt_ids = [evaluation.prompt_id for evaluation in evaluations]
    response_ids = [evaluation.response_id for evaluation in evaluations]
    target_models = [evaluation.target_model for evaluation in evaluations]
    judge_models = [evaluation.judge_model for evaluation in evaluations]
    repetitions = [evaluation.repetition for evaluation in evaluations]

    print(f"Unique prompt_ids:               {len(set(prompt_ids))}")
    print(f"Unique response_ids:             {len(set(response_ids))}")
    print(f"Target models:                   {sorted(set(target_models))}")
    print(f"Judge models:                    {sorted(set(judge_models))}")
    print(f"Repetitions found:               {sorted(set(repetitions))}")

    labels = [evaluation.label.value for evaluation in evaluations]
    adequacy_scores = [evaluation.adequacy_score for evaluation in evaluations]
    safety_scores = [evaluation.safety_score for evaluation in evaluations]
    helpfulness_scores = [evaluation.helpfulness_score for evaluation in evaluations]
    sensitivity_awareness_scores = [
        evaluation.sensitivity_awareness_score for evaluation in evaluations
    ]
    over_refusals = [evaluation.over_refusal for evaluation in evaluations]
    under_refusals = [evaluation.under_refusal for evaluation in evaluations]

    print_counter("LABEL DISTRIBUTION", labels)
    print_counter("ADEQUACY SCORE DISTRIBUTION", adequacy_scores)
    print_counter("SAFETY SCORE DISTRIBUTION", safety_scores)
    print_counter("HELPFULNESS SCORE DISTRIBUTION", helpfulness_scores)
    print_counter(
        "SENSITIVITY AWARENESS SCORE DISTRIBUTION",
        sensitivity_awareness_scores,
    )
    print_counter("OVER-REFUSAL DISTRIBUTION", over_refusals)
    print_counter("UNDER-REFUSAL DISTRIBUTION", under_refusals)

    print(format_subsection("AVERAGE SCORES"))
    print(f"Average adequacy score:              {average(adequacy_scores):.2f}")
    print(f"Average safety score:                {average(safety_scores):.2f}")
    print(f"Average helpfulness score:           {average(helpfulness_scores):.2f}")
    print(
        "Average sensitivity awareness score: "
        f"{average(sensitivity_awareness_scores):.2f}"
    )

    invalid_scores = [
        evaluation
        for evaluation in evaluations
        if not (1 <= evaluation.adequacy_score <= 5)
        or not (1 <= evaluation.safety_score <= 5)
        or not (1 <= evaluation.helpfulness_score <= 5)
        or not (1 <= evaluation.sensitivity_awareness_score <= 5)
    ]

    empty_rationales = [
        evaluation
        for evaluation in evaluations
        if not evaluation.rationale.strip()
    ]

    print(format_subsection("QUALITY CHECKS"))
    print(f"Invalid score ranges:                {len(invalid_scores)}")
    print(f"Empty rationales:                    {len(empty_rationales)}")

    print(format_subsection("FIRST 5 EVALUATIONS"))

    for i, evaluation in enumerate(evaluations[:5], start=1):
        print(f"\n[{i}]")
        print(f"evaluation_id:                 {evaluation.evaluation_id}")
        print(f"response_id:                   {evaluation.response_id}")
        print(f"prompt_id:                     {evaluation.prompt_id}")
        print(f"target_model:                  {evaluation.target_model}")
        print(f"judge_model:                   {evaluation.judge_model}")
        print(f"repetition:                    {evaluation.repetition}")
        print(f"label:                         {evaluation.label.value}")
        print(f"adequacy_score:                {evaluation.adequacy_score}")
        print(f"safety_score:                  {evaluation.safety_score}")
        print(f"helpfulness_score:             {evaluation.helpfulness_score}")
        print(f"sensitivity_awareness_score:   {evaluation.sensitivity_awareness_score}")
        print(f"over_refusal:                  {evaluation.over_refusal}")
        print(f"under_refusal:                 {evaluation.under_refusal}")
        print(f"failure_modes:                 {evaluation.failure_modes}")
        print(f"rationale:                     {evaluation.rationale}")

    print(format_section("INSPECTION COMPLETED"))
    print("\nNext step:")
    print("  Run: python -m src.scripts.07_build_sensy_refactor_seed")


if __name__ == "__main__":
    main()