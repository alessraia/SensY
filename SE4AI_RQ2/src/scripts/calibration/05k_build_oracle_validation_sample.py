import random
from pathlib import Path

import pandas as pd

from src.domain.models import HumanLabeledResponse
from src.utils.jsonl import read_jsonl, write_jsonl


FULL_ORACLE_JSONL = "data/calibration/oracle/human_labeled_responses.jsonl"
CALIBRATION_SAMPLE_JSONL = "data/calibration/oracle/human_oracle_calibration_sample.jsonl"

OUTPUT_VALIDATION_JSONL = "data/calibration/oracle/human_oracle_validation_sample_180.jsonl"
OUTPUT_VALIDATION_CSV = "data/calibration/oracle/human_oracle_validation_sample_180.csv"

SAMPLES_PER_SOURCE_MODEL_PER_LABEL = 30
RANDOM_STATE = 123


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def records_to_dataframe(records: list[HumanLabeledResponse]) -> pd.DataFrame:
    rows = []

    for item in records:
        rows.append(
            {
                "oracle_id": item.oracle_id,
                "question": item.question,
                "category": item.category,
                "response_text": item.response_text,
                "source_model": item.source_model,
                "response_index": item.response_index,
                "oracle_source_file": item.oracle_source_file,
                "manual_adequate": item.manual_adequate,
                "manual_label": item.manual_label.value,
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_records(
    df: pd.DataFrame,
    original_records_by_id: dict[str, HumanLabeledResponse],
) -> list[HumanLabeledResponse]:
    records = []

    for oracle_id in df["oracle_id"].tolist():
        records.append(original_records_by_id[oracle_id])

    return records


def sample_validation_set(df_available: pd.DataFrame) -> pd.DataFrame:
    sampled_groups = []

    random.seed(RANDOM_STATE)

    grouped = df_available.groupby(["source_model", "manual_label"])

    for (source_model, manual_label), group in grouped:
        if len(group) < SAMPLES_PER_SOURCE_MODEL_PER_LABEL:
            raise ValueError(
                f"Not enough records for source_model={source_model}, "
                f"manual_label={manual_label}. "
                f"Required={SAMPLES_PER_SOURCE_MODEL_PER_LABEL}, "
                f"available={len(group)}"
            )

        sampled = group.sample(
            n=SAMPLES_PER_SOURCE_MODEL_PER_LABEL,
            random_state=RANDOM_STATE,
        )

        sampled_groups.append(sampled)

    validation_df = pd.concat(sampled_groups, ignore_index=True)

    validation_df = validation_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return validation_df


def preview(text: str, max_chars: int = 180) -> str:
    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def print_first_records(df: pd.DataFrame, limit: int = 10) -> None:
    print(format_subsection(f"FIRST {limit} VALIDATION RECORDS"))

    for i, (_, row) in enumerate(df.head(limit).iterrows(), start=1):
        print(f"\n[{i}]")
        print(f"oracle_id:      {row['oracle_id']}")
        print(f"source_model:   {row['source_model']}")
        print(f"manual_label:   {row['manual_label']}")
        print(f"category:       {row['category']}")
        print(f"question:       {preview(row['question'])}")
        print(f"response_text:  {preview(row['response_text'])}")


def main() -> None:
    print(format_section("ORACLE VALIDATION SAMPLE CREATION"))

    print("\nInput files:")
    print(f"  Full oracle:          {FULL_ORACLE_JSONL}")
    print(f"  Calibration sample:   {CALIBRATION_SAMPLE_JSONL}")

    full_records = read_jsonl(FULL_ORACLE_JSONL, HumanLabeledResponse)
    calibration_records = read_jsonl(
        CALIBRATION_SAMPLE_JSONL,
        HumanLabeledResponse,
    )

    full_df = records_to_dataframe(full_records)
    calibration_df = records_to_dataframe(calibration_records)

    calibration_ids = set(calibration_df["oracle_id"].tolist())

    available_df = full_df[
        ~full_df["oracle_id"].isin(calibration_ids)
    ].copy()

    print(format_subsection("FULL ORACLE COUNTS"))
    print(f"Human-labeled responses:      {len(full_df)}")
    print(f"Unique questions:             {full_df['question'].nunique()}")
    print(f"Source models:                {full_df['source_model'].nunique()}")

    print(format_subsection("CALIBRATION SAMPLE ALREADY USED"))
    print(f"Calibration responses:        {len(calibration_df)}")
    print(f"Calibration oracle_ids:       {len(calibration_ids)}")

    print(format_subsection("AVAILABLE RECORDS AFTER EXCLUSION"))
    print(f"Available responses:          {len(available_df)}")
    print(f"Unique questions:             {available_df['question'].nunique()}")

    print(format_subsection("AVAILABLE LABEL DISTRIBUTION"))
    print(available_df["manual_label"].value_counts())

    print(format_subsection("AVAILABLE LABEL DISTRIBUTION BY SOURCE MODEL"))
    print(
        available_df.groupby(["source_model", "manual_label"])
        .size()
        .unstack(fill_value=0)
    )

    print(format_subsection("SAMPLING CONFIGURATION"))
    print(
        "Samples per source model per label: "
        f"{SAMPLES_PER_SOURCE_MODEL_PER_LABEL}"
    )
    print(f"Random state:                       {RANDOM_STATE}")
    print("Expected validation sample size:    180")

    validation_df = sample_validation_set(available_df)

    overlap_ids = set(validation_df["oracle_id"].tolist()) & calibration_ids

    if overlap_ids:
        raise ValueError(
            "Validation sample overlaps with calibration sample. "
            f"Overlapping ids: {sorted(overlap_ids)[:10]}"
        )

    print(format_subsection("VALIDATION SAMPLE COUNTS"))
    print(f"Validation sample size:             {len(validation_df)}")
    print(f"Unique questions:                   {validation_df['question'].nunique()}")
    print(f"Source models:                      {validation_df['source_model'].nunique()}")
    print(f"Overlap with calibration sample:    {len(overlap_ids)}")

    print(format_subsection("VALIDATION LABEL DISTRIBUTION"))
    print(validation_df["manual_label"].value_counts())

    print(format_subsection("VALIDATION LABEL DISTRIBUTION BY SOURCE MODEL"))
    print(
        validation_df.groupby(["source_model", "manual_label"])
        .size()
        .unstack(fill_value=0)
    )

    print(format_subsection("VALIDATION CATEGORY DISTRIBUTION"))
    print(validation_df["category"].value_counts())

    print_first_records(validation_df, limit=10)

    original_records_by_id = {
        item.oracle_id: item for item in full_records
    }

    validation_records = dataframe_to_records(
        validation_df,
        original_records_by_id,
    )

    Path(OUTPUT_VALIDATION_JSONL).parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUTPUT_VALIDATION_JSONL, validation_records)
    validation_df.to_csv(
        OUTPUT_VALIDATION_CSV,
        index=False,
        encoding="utf-8",
    )

    print(format_subsection("OUTPUT FILES"))
    print(f"JSONL: {OUTPUT_VALIDATION_JSONL}")
    print(f"CSV:   {OUTPUT_VALIDATION_CSV}")

    print(format_section("ORACLE VALIDATION SAMPLE CREATION COMPLETED"))


if __name__ == "__main__":
    main()