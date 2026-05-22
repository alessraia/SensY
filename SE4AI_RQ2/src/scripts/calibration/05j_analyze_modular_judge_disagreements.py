from pathlib import Path

import pandas as pd


INPUT_COMPARISON_CSV = "data/calibration/results/modular_judge_oracle_comparisonv4.csv"

OUTPUT_DISAGREEMENTS_CSV = "data/calibration/results/modular_judge_oracle_disagreement_analysisv4.csv"
OUTPUT_FALSE_NEGATIVES_CSV = "data/calibration/results/modular_judge_oracle_false_negativesv4.csv"
OUTPUT_FALSE_POSITIVES_CSV = "data/calibration/results/modular_judge_oracle_false_positivesv4.csv"


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def preview_text(text: str, max_chars: int = 280) -> str:
    if pd.isna(text):
        return ""

    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def add_manual_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["review_priority"] = ""
    df["suspected_missing_criterion"] = ""
    df["human_oracle_possible_reason"] = ""
    df["rubric_fix_needed"] = ""
    df["router_fix_needed"] = ""
    df["notes"] = ""

    return df


def save_outputs(df: pd.DataFrame) -> None:
    Path(OUTPUT_DISAGREEMENTS_CSV).parent.mkdir(parents=True, exist_ok=True)

    disagreements = df[df["error_type"] != "none"].copy()
    false_negatives = df[df["error_type"] == "false_negative"].copy()
    false_positives = df[df["error_type"] == "false_positive"].copy()

    disagreements = add_manual_review_columns(disagreements)
    false_negatives = add_manual_review_columns(false_negatives)
    false_positives = add_manual_review_columns(false_positives)

    disagreements.to_csv(OUTPUT_DISAGREEMENTS_CSV, index=False, encoding="utf-8")
    false_negatives.to_csv(OUTPUT_FALSE_NEGATIVES_CSV, index=False, encoding="utf-8")
    false_positives.to_csv(OUTPUT_FALSE_POSITIVES_CSV, index=False, encoding="utf-8")


def print_error_summary(df: pd.DataFrame) -> None:
    print(format_subsection("GLOBAL ERROR DISTRIBUTION"))
    print(df["error_type"].value_counts())

    print(format_subsection("ERROR DISTRIBUTION BY SOURCE MODEL"))
    print(
        df.groupby(["source_model", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    print(format_subsection("ERROR DISTRIBUTION BY CATEGORY"))
    error_df = df[df["error_type"] != "none"]

    if error_df.empty:
        print("No disagreements.")
    else:
        category_errors = (
            error_df.groupby(["category", "error_type"])
            .size()
            .unstack(fill_value=0)
        )

        sort_columns = [
            column
            for column in ["false_negative", "false_positive"]
            if column in category_errors.columns
        ]

        if sort_columns:
            category_errors = category_errors.sort_values(
                by=sort_columns,
                ascending=[False] * len(sort_columns),
            )

        print(category_errors)

    print(format_subsection("AVERAGE SCORES BY ERROR TYPE"))
    score_columns = [
        "adequacy_score",
        "safety_score",
        "helpfulness_score",
        "sensitivity_awareness_score",
    ]

    print(
        df.groupby("error_type")[score_columns]
        .mean()
        .round(2)
    )

    print(format_subsection("FALSE NEGATIVES BY SOURCE MODEL AND CATEGORY"))
    false_negatives = df[df["error_type"] == "false_negative"]

    if false_negatives.empty:
        print("No false negatives.")
    else:
        print(
            false_negatives.groupby(["source_model", "category"])
            .size()
            .unstack(fill_value=0)
        )


def print_examples(
    df: pd.DataFrame,
    error_type: str,
    title: str,
    limit: int = 10,
    source_model: str | None = None,
) -> None:
    print(format_subsection(title))

    subset = df[df["error_type"] == error_type].copy()

    if source_model is not None:
        subset = subset[subset["source_model"] == source_model]

    if subset.empty:
        print(f"No {error_type} cases.")
        return

    if error_type == "false_negative":
        subset = subset.sort_values(
            by=["adequacy_score", "safety_score", "sensitivity_awareness_score"],
            ascending=[False, False, False],
        )
    elif error_type == "false_positive":
        subset = subset.sort_values(
            by=["adequacy_score", "safety_score", "sensitivity_awareness_score"],
            ascending=[True, True, True],
        )

    for i, (_, row) in enumerate(subset.head(limit).iterrows(), start=1):
        print(f"\n[{i}]")
        print(f"oracle_id:       {row['oracle_id']}")
        print(f"source_model:    {row['source_model']}")
        print(f"category:        {row['category']}")
        print(f"manual_label:    {row['manual_label']}")
        print(f"judge_label:     {row['judge_label']}")
        print(f"adequacy_score:  {row['adequacy_score']}")
        print(f"safety_score:    {row['safety_score']}")
        print(f"helpfulness:     {row['helpfulness_score']}")
        print(f"sensitivity:     {row['sensitivity_awareness_score']}")
        print(f"over_refusal:    {row['over_refusal']}")
        print(f"under_refusal:   {row['under_refusal']}")
        print(f"failure_modes:   {row['failure_modes']}")
        print(f"question:        {preview_text(row['question'])}")
        print(f"response:        {preview_text(row['response_text'])}")
        print(f"judge rationale: {preview_text(row['rationale'])}")


def print_actionable_interpretation(df: pd.DataFrame) -> None:
    print(format_subsection("ACTIONABLE INTERPRETATION"))

    total = len(df)
    false_negatives = len(df[df["error_type"] == "false_negative"])
    false_positives = len(df[df["error_type"] == "false_positive"])

    print(f"Total evaluated cases:        {total}")
    print(f"False negatives:              {false_negatives}")
    print(f"False positives:              {false_positives}")

    if false_negatives > false_positives:
        print(
            "\nMain issue: the modular judge is still too permissive. "
            "It misses many responses that the human oracle marked as inadequate."
        )
        print(
            "Next step: inspect false negatives, especially for qwen, to understand "
            "whether the issue is caused by missing risk tags, weak addenda, or an overly "
            "permissive final label policy."
        )
    elif false_positives > false_negatives:
        print(
            "\nMain issue: the modular judge is too severe. "
            "It marks too many human-adequate responses as inadequate."
        )
    else:
        print(
            "\nFalse positives and false negatives are balanced. "
            "Inspect category-specific disagreements."
        )


def main() -> None:
    print(format_section("MODULAR JUDGE ORACLE DISAGREEMENT ANALYSIS"))

    print("\nInput file:")
    print(f"  {INPUT_COMPARISON_CSV}")

    df = pd.read_csv(INPUT_COMPARISON_CSV)

    if df.empty:
        print("\nNo comparison rows found.")
        return

    required_columns = {
        "oracle_id",
        "question",
        "category",
        "response_text",
        "source_model",
        "manual_label",
        "judge_label",
        "agreement",
        "error_type",
        "adequacy_score",
        "safety_score",
        "helpfulness_score",
        "sensitivity_awareness_score",
        "over_refusal",
        "under_refusal",
        "failure_modes",
        "rationale",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns in comparison CSV: "
            f"{sorted(missing_columns)}"
        )

    print(format_subsection("BASIC COUNTS"))
    print(f"Rows loaded:              {len(df)}")
    print(f"Unique oracle_ids:        {df['oracle_id'].nunique()}")
    print(f"Source models:            {sorted(df['source_model'].unique())}")
    print(f"Manual labels:            {df['manual_label'].value_counts().to_dict()}")
    print(f"Judge labels:             {df['judge_label'].value_counts().to_dict()}")

    print_error_summary(df)

    print_examples(
        df=df,
        error_type="false_negative",
        title="FALSE NEGATIVE EXAMPLES: human=inadequate, modular judge=adequate",
        limit=10,
    )

    print_examples(
        df=df,
        error_type="false_negative",
        title="FALSE NEGATIVE EXAMPLES — QWEN ONLY",
        limit=10,
        source_model="qwen",
    )

    print_examples(
        df=df,
        error_type="false_positive",
        title="FALSE POSITIVE EXAMPLES: human=adequate, modular judge=inadequate",
        limit=10,
    )

    save_outputs(df)

    print(format_subsection("OUTPUT FILES"))
    print(f"All disagreements: {OUTPUT_DISAGREEMENTS_CSV}")
    print(f"False negatives:   {OUTPUT_FALSE_NEGATIVES_CSV}")
    print(f"False positives:   {OUTPUT_FALSE_POSITIVES_CSV}")

    print_actionable_interpretation(df)

    print(format_section("MODULAR DISAGREEMENT ANALYSIS COMPLETED"))


if __name__ == "__main__":
    main()