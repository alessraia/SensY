from pathlib import Path

import pandas as pd

from src.domain.models import HumanLabeledResponse
from src.utils.jsonl import read_jsonl, write_jsonl


INPUT_JSONL = "data/calibration/oracle/human_labeled_responses.jsonl"

OUTPUT_JSONL = "data/calibration/oracle/human_oracle_calibration_sample.jsonl"
OUTPUT_CSV = "data/calibration/oracle/human_oracle_calibration_sample.csv"

SAMPLES_PER_MODEL_PER_LABEL = 30
RANDOM_STATE = 42


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def records_to_dataframe(records: list[HumanLabeledResponse]) -> pd.DataFrame:
    rows = []

    for record in records:
        rows.append(
            {
                "oracle_id": record.oracle_id,
                "question": record.question,
                "category": record.category,
                "response_text": record.response_text,
                "manual_adequate": record.manual_adequate,
                "manual_label": record.manual_label.value,
                "source_model": record.source_model,
                "response_index": record.response_index,
                "oracle_source_file": record.oracle_source_file,
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_records(df: pd.DataFrame) -> list[HumanLabeledResponse]:
    records: list[HumanLabeledResponse] = []

    for _, row in df.iterrows():
        records.append(
            HumanLabeledResponse(
                oracle_id=row["oracle_id"],
                question=row["question"],
                category=row["category"] if pd.notna(row["category"]) else None,
                response_text=row["response_text"],
                manual_adequate=int(row["manual_adequate"]),
                manual_label=row["manual_label"],
                source_model=row["source_model"],
                response_index=int(row["response_index"]),
                oracle_source_file=row["oracle_source_file"],
            )
        )

    return records


def build_balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    sampled_parts = []

    grouped = df.groupby(["source_model", "manual_label"])

    for (source_model, manual_label), group in grouped:
        available = len(group)

        if available < SAMPLES_PER_MODEL_PER_LABEL:
            print(
                f"[WARNING] Not enough examples for "
                f"source_model={source_model}, manual_label={manual_label}. "
                f"Available={available}, requested={SAMPLES_PER_MODEL_PER_LABEL}. "
                f"Using all available examples."
            )

            sample = group.copy()
        else:
            sample = group.sample(
                n=SAMPLES_PER_MODEL_PER_LABEL,
                random_state=RANDOM_STATE,
            )

        sampled_parts.append(sample)

    if not sampled_parts:
        raise ValueError("No sampled records were produced.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    sampled_df = sampled_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return sampled_df


def main() -> None:
    print(format_section("ORACLE CALIBRATION SAMPLE CREATION"))

    print("\nInput file:")
    print(f"  {INPUT_JSONL}")

    records = read_jsonl(INPUT_JSONL, HumanLabeledResponse)
    df = records_to_dataframe(records)

    print(format_subsection("FULL ORACLE COUNTS"))
    print(f"Human-labeled responses:      {len(df)}")
    print(f"Unique questions:             {df['question'].nunique()}")
    print(f"Source models:                {df['source_model'].nunique()}")

    print(format_subsection("FULL ORACLE LABEL DISTRIBUTION"))
    print(df["manual_label"].value_counts())

    print(format_subsection("FULL ORACLE LABEL DISTRIBUTION BY SOURCE MODEL"))
    print(
        df.groupby(["source_model", "manual_label"])
        .size()
        .unstack(fill_value=0)
    )

    sampled_df = build_balanced_sample(df)

    output_jsonl_path = Path(OUTPUT_JSONL)
    output_csv_path = Path(OUTPUT_CSV)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_records = dataframe_to_records(sampled_df)

    write_jsonl(OUTPUT_JSONL, sampled_records)
    sampled_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(format_subsection("SAMPLING CONFIGURATION"))
    print(f"Samples per source model per label: {SAMPLES_PER_MODEL_PER_LABEL}")
    print(f"Random state:                       {RANDOM_STATE}")

    print(format_subsection("CALIBRATION SAMPLE COUNTS"))
    print(f"Calibration sample size:             {len(sampled_df)}")
    print(f"Unique questions:                    {sampled_df['question'].nunique()}")
    print(f"Source models:                       {sampled_df['source_model'].nunique()}")

    print(format_subsection("CALIBRATION SAMPLE LABEL DISTRIBUTION"))
    print(sampled_df["manual_label"].value_counts())

    print(format_subsection("CALIBRATION SAMPLE LABEL DISTRIBUTION BY SOURCE MODEL"))
    print(
        sampled_df.groupby(["source_model", "manual_label"])
        .size()
        .unstack(fill_value=0)
    )

    print(format_subsection("CALIBRATION SAMPLE CATEGORY DISTRIBUTION"))
    print(sampled_df["category"].value_counts().head(20))

    print(format_subsection("FIRST 10 SAMPLE RECORDS"))

    for i, (_, row) in enumerate(sampled_df.head(10).iterrows(), start=1):
        response_preview = row["response_text"]

        if len(response_preview) > 220:
            response_preview = response_preview[:220] + "..."

        print(f"\n[{i}]")
        print(f"oracle_id:      {row['oracle_id']}")
        print(f"source_model:   {row['source_model']}")
        print(f"manual_label:   {row['manual_label']}")
        print(f"category:       {row['category']}")
        print(f"question:       {row['question']}")
        print(f"response_text:  {response_preview}")

    print(format_subsection("OUTPUT FILES"))
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

    print(format_section("ORACLE CALIBRATION SAMPLE CREATION COMPLETED"))


if __name__ == "__main__":
    main()