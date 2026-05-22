import json
from pathlib import Path

import pandas as pd

from ..domain.models import AdequacyLabel, HumanLabeledResponse
from ..utils.jsonl import write_jsonl


RAW_FILES = {
    "deepseek": "data/oracle/raw/deepseek_response.json",
    "llama": "data/oracle/raw/llama_response.json",
    "qwen": "data/oracle/raw/qwen_response.json",
}

OUTPUT_JSONL = "data/oracle/human_labeled_responses.jsonl"
OUTPUT_CSV = "data/oracle/human_labeled_responses.csv"


def load_json_list(path: str | Path) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, found {type(data)}")

    return data


def adequate_to_label(value: int) -> AdequacyLabel:
    if value == 1:
        return AdequacyLabel.ADEQUATE

    if value == 0:
        return AdequacyLabel.INADEQUATE

    raise ValueError(f"Unexpected adequate value: {value}")


def build_oracle_id(
    source_model: str,
    prompt_index: int,
    response_index: int,
) -> str:
    return f"oracle_{source_model}_{prompt_index:06d}_r{response_index}"


def normalize_records(
    source_model: str,
    source_path: str,
    raw_records: list[dict],
) -> list[HumanLabeledResponse]:
    normalized: list[HumanLabeledResponse] = []

    source_file_name = Path(source_path).name

    for prompt_index, record in enumerate(raw_records, start=1):
        question = record.get("question")
        category = record.get("category")

        if not question:
            raise ValueError(
                f"Missing question in {source_file_name}, record {prompt_index}"
            )

        for response_index in [1, 2, 3]:
            response_key = f"response{response_index}"
            adequate_key = f"adequate{response_index}"

            response_text = record.get(response_key)
            adequate_value = record.get(adequate_key)

            if response_text is None:
                raise ValueError(
                    f"Missing {response_key} in {source_file_name}, "
                    f"record {prompt_index}"
                )

            if adequate_value is None:
                raise ValueError(
                    f"Missing {adequate_key} in {source_file_name}, "
                    f"record {prompt_index}"
                )

            oracle_item = HumanLabeledResponse(
                oracle_id=build_oracle_id(
                    source_model=source_model,
                    prompt_index=prompt_index,
                    response_index=response_index,
                ),
                question=str(question).strip(),
                category=str(category).strip() if category is not None else None,
                response_text=str(response_text).strip(),
                manual_adequate=int(adequate_value),
                manual_label=adequate_to_label(int(adequate_value)),
                source_model=source_model,
                response_index=response_index,
                oracle_source_file=source_file_name,
            )

            normalized.append(oracle_item)

    return normalized


def to_csv_rows(records: list[HumanLabeledResponse]) -> list[dict]:
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

    return rows


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subsection(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def main() -> None:
    print_section("HUMAN ORACLE PREPARATION")

    print("\nInput files:")

    all_records: list[HumanLabeledResponse] = []

    for source_model, path in RAW_FILES.items():
        print(f"  {source_model:<10} {path}")

        raw_records = load_json_list(path)

        normalized = normalize_records(
            source_model=source_model,
            source_path=path,
            raw_records=raw_records,
        )

        all_records.extend(normalized)

    write_jsonl(OUTPUT_JSONL, all_records)

    csv_rows = to_csv_rows(all_records)
    df = pd.DataFrame(csv_rows)

    output_csv_path = Path(OUTPUT_CSV)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print_subsection("OUTPUT FILES")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

    print_subsection("GLOBAL COUNTS")
    print(f"Human-labeled responses: {len(all_records)}")
    print(f"Unique questions:        {df['question'].nunique()}")
    print(f"Source models:           {df['source_model'].nunique()}")

    print_subsection("RESPONSES BY SOURCE MODEL")
    print(df["source_model"].value_counts())

    print_subsection("MANUAL LABEL DISTRIBUTION")
    print(df["manual_label"].value_counts())

    print_subsection("MANUAL LABEL DISTRIBUTION BY SOURCE MODEL")
    print(
        df.groupby(["source_model", "manual_label"])
        .size()
        .unstack(fill_value=0)
    )

    print_subsection("CATEGORY DISTRIBUTION")
    print(df["category"].value_counts().head(20))

    print_subsection("FIRST 5 NORMALIZED RECORDS")

    for i, record in enumerate(all_records[:5], start=1):
        response_preview = record.response_text

        if len(response_preview) > 250:
            response_preview = response_preview[:250] + "..."

        print(f"\n[{i}]")
        print(f"oracle_id:          {record.oracle_id}")
        print(f"source_model:       {record.source_model}")
        print(f"response_index:     {record.response_index}")
        print(f"manual_label:       {record.manual_label.value}")
        print(f"category:           {record.category}")
        print(f"question:           {record.question}")
        print(f"response_text:      {response_preview}")

    print_section("HUMAN ORACLE PREPARATION COMPLETED")


if __name__ == "__main__":
    main()