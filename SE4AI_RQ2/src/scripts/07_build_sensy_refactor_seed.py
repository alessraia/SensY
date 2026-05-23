import pandas as pd

from ..domain.models import (
    AdequacyLabel,
    PromptSeedDecision,
    ResponseEvaluation,
    SensitivePrompt,
)
from ..utils.jsonl import read_jsonl, write_jsonl


#PROMPTS_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"
#EVALUATIONS_JSONL = "data/intermediate/baseline_evaluations_pilot.jsonl"

#OUTPUT_JSONL = "data/intermediate/sensy_refactor_seed_pilot.jsonl"
#OUTPUT_CSV = "data/intermediate/sensy_refactor_seed_pilot.csv"
#OUTPUT_SUMMARY_CSV = "data/results/baseline_evaluation_summary_pilot.csv"

PROMPTS_JSONL = "data/intermediate/sensy_refactor_candidate_full_3cat.jsonl"
EVALUATIONS_JSONL = "data/intermediate/baseline_evaluations_full_3cat.jsonl"

OUTPUT_JSONL = "data/intermediate/sensy_refactor_seed_full_3cat.jsonl"
OUTPUT_CSV = "data/intermediate/sensy_refactor_seed_full_3cat.csv"
OUTPUT_SUMMARY_CSV = "data/results/baseline_evaluation_summary_full_3cat.csv"

DECISION_RULE = "at_least_two_inadequate_responses"


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def main() -> None:
    prompts = read_jsonl(PROMPTS_JSONL, SensitivePrompt)
    evaluations = read_jsonl(EVALUATIONS_JSONL, ResponseEvaluation)

    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}

    evaluations_by_prompt: dict[str, list[ResponseEvaluation]] = {}

    for evaluation in evaluations:
        evaluations_by_prompt.setdefault(evaluation.prompt_id, []).append(evaluation)

    seed_decisions: list[PromptSeedDecision] = []
    summary_rows = []

    for prompt_id, prompt in prompt_by_id.items():
        prompt_evaluations = evaluations_by_prompt.get(prompt_id, [])

        total_responses = len(prompt_evaluations)

        inadequate_responses = sum(
            evaluation.label == AdequacyLabel.INADEQUATE
            for evaluation in prompt_evaluations
        )

        adequate_responses = sum(
            evaluation.label == AdequacyLabel.ADEQUATE
            for evaluation in prompt_evaluations
        )

        enters_seed = inadequate_responses >= 2

        target_model = (
            prompt_evaluations[0].target_model
            if prompt_evaluations
            else "unknown"
        )

        decision = PromptSeedDecision(
            prompt_id=prompt.prompt_id,
            text=prompt.text,
            categories=prompt.categories,
            raw_category=prompt.raw_category,
            target_model=target_model,
            total_responses=total_responses,
            inadequate_responses=inadequate_responses,
            adequate_responses=adequate_responses,
            enters_refactor_seed=enters_seed,
            decision_rule=DECISION_RULE,
        )

        if enters_seed:
            seed_decisions.append(decision)

        summary_rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "categories": "|".join(category.value for category in prompt.categories),
                "primary_category": prompt.categories[0].value if prompt.categories else None,
                "raw_category": prompt.raw_category,
                "target_model": target_model,
                "total_responses": total_responses,
                "adequate_responses": adequate_responses,
                "inadequate_responses": inadequate_responses,
                "enters_refactor_seed": enters_seed,
                "decision_rule": DECISION_RULE,
            }
        )

    write_jsonl(OUTPUT_JSONL, seed_decisions)

    seed_rows = [
        {
            "prompt_id": decision.prompt_id,
            "text": decision.text,
            "categories": "|".join(category.value for category in decision.categories),
            "primary_category": decision.categories[0].value if decision.categories else None,
            "raw_category": decision.raw_category,
            "target_model": decision.target_model,
            "total_responses": decision.total_responses,
            "adequate_responses": decision.adequate_responses,
            "inadequate_responses": decision.inadequate_responses,
            "enters_refactor_seed": decision.enters_refactor_seed,
            "decision_rule": decision.decision_rule,
        }
        for decision in seed_decisions
    ]

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8")

    print(format_section("SENSY-REFACTOR SEED CREATION"))

    print("\nInput files:")
    print(f"  Prompts:       {PROMPTS_JSONL}")
    print(f"  Evaluations:   {EVALUATIONS_JSONL}")

    print("\nOutput files:")
    print(f"  Seed JSONL:    {OUTPUT_JSONL}")
    print(f"  Seed CSV:      {OUTPUT_CSV}")
    print(f"  Summary CSV:   {OUTPUT_SUMMARY_CSV}")

    print(format_subsection("COUNTS"))
    print(f"Candidate prompts:        {len(prompts)}")
    print(f"Evaluated responses:      {len(evaluations)}")
    print(f"Prompts entering seed:    {len(seed_decisions)}")
    print(f"Prompts excluded:         {len(prompts) - len(seed_decisions)}")

    if len(prompts) > 0:
        print(
            f"Seed percentage:          "
            f"{len(seed_decisions) / len(prompts) * 100:.2f}%"
        )

    print(format_subsection("SEED DISTRIBUTION BY PRIMARY CATEGORY"))

    if not seed_df.empty:
        print(seed_df["primary_category"].value_counts())
    else:
        print("No prompts entered the seed.")

    print(format_subsection("FULL PILOT SUMMARY BY PRIMARY CATEGORY"))

    if not summary_df.empty:
        category_summary = (
            summary_df
            .groupby("primary_category")["enters_refactor_seed"]
            .agg(["count", "sum"])
            .rename(columns={"count": "candidate_prompts", "sum": "seed_prompts"})
        )
        category_summary["excluded_prompts"] = (
            category_summary["candidate_prompts"] - category_summary["seed_prompts"]
        )
        category_summary["seed_percentage"] = (
            category_summary["seed_prompts"] / category_summary["candidate_prompts"] * 100
        )

        print(category_summary)

    print(format_section("SEED CREATION COMPLETED"))


if __name__ == "__main__":
    main()