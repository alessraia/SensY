import json
from pathlib import Path

import pandas as pd

from src.judge.risk_router import PromptRiskRouter


SENSITIVE_POOL_JSONL = "data/intermediate/sensy2_sensitive_pool.jsonl"
PILOT_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"

OUTPUT_POOL_CSV = "data/results/risk_router_sensitive_pool.csv"
OUTPUT_PILOT_CSV = "data/results/risk_router_candidate_pilot.csv"


def read_jsonl_as_dicts(path: str) -> list[dict]:
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def get_primary_category(record: dict) -> str:
    categories = record.get("categories")

    if isinstance(categories, list) and categories:
        first = categories[0]

        if isinstance(first, dict) and "value" in first:
            return first["value"]

        return str(first)

    return "Unknown"


def route_records(records: list[dict], dataset_name: str) -> pd.DataFrame:
    router = PromptRiskRouter()

    rows = []

    for record in records:
        prompt_id = record.get("prompt_id")
        text = record.get("text", "")
        raw_category = record.get("raw_category")
        primary_category = get_primary_category(record)

        result = router.route(text)

        risk_tags = [tag.value for tag in result.risk_tags]

        rows.append(
            {
                "dataset": dataset_name,
                "prompt_id": prompt_id,
                "text": text,
                "primary_category": primary_category,
                "raw_category": raw_category,
                "risk_tags": "|".join(risk_tags),
                "risk_tag_count": len(risk_tags),
                "only_general": result.only_general,
                "matched_rules_json": json.dumps(
                    result.matched_rules,
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subsection(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def explode_tags(df: pd.DataFrame) -> pd.DataFrame:
    exploded_rows = []

    for _, row in df.iterrows():
        tags = str(row["risk_tags"]).split("|") if row["risk_tags"] else []

        for tag in tags:
            exploded_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "primary_category": row["primary_category"],
                    "risk_tag": tag,
                    "only_general": row["only_general"],
                }
            )

    return pd.DataFrame(exploded_rows)


def print_router_summary(df: pd.DataFrame, title: str) -> None:
    print_section(title)

    print_subsection("BASIC COUNTS")
    print(f"Prompts:                 {len(df)}")
    print(f"Prompts with only general tag: {int(df['only_general'].sum())}")
    print(
        f"Only-general percentage: "
        f"{df['only_general'].mean() * 100:.2f}%"
    )

    print_subsection("RISK TAG COUNT DISTRIBUTION")
    print(df["risk_tag_count"].value_counts().sort_index())

    exploded = explode_tags(df)

    print_subsection("RISK TAG DISTRIBUTION")
    if exploded.empty:
        print("No tags.")
    else:
        print(exploded["risk_tag"].value_counts())

    print_subsection("PRIMARY CATEGORY DISTRIBUTION")
    print(df["primary_category"].value_counts())

    print_subsection("RISK TAGS BY PRIMARY CATEGORY")
    if exploded.empty:
        print("No tags.")
    else:
        table = (
            exploded.groupby(["primary_category", "risk_tag"])
            .size()
            .unstack(fill_value=0)
        )
        print(table)

    print_subsection("EXAMPLES WITH ONLY GENERAL TAG")

    only_general_df = df[df["only_general"]].head(10)

    if only_general_df.empty:
        print("No only-general examples.")
    else:
        for i, (_, row) in enumerate(only_general_df.iterrows(), start=1):
            text = row["text"]
            if len(text) > 220:
                text = text[:220] + "..."

            print(f"\n[{i}]")
            print(f"prompt_id:         {row['prompt_id']}")
            print(f"primary_category:  {row['primary_category']}")
            print(f"text:              {text}")

    print_subsection("FIRST 10 ROUTED PROMPTS")

    for i, (_, row) in enumerate(df.head(10).iterrows(), start=1):
        text = row["text"]
        if len(text) > 220:
            text = text[:220] + "..."

        print(f"\n[{i}]")
        print(f"prompt_id:         {row['prompt_id']}")
        print(f"primary_category:  {row['primary_category']}")
        print(f"risk_tags:         {row['risk_tags']}")
        print(f"text:              {text}")


def main() -> None:
    print_section("PROMPT RISK ROUTER TEST")

    print("\nInput files:")
    print(f"  Sensitive pool:   {SENSITIVE_POOL_JSONL}")
    print(f"  Candidate pilot:  {PILOT_JSONL}")

    sensitive_pool_records = read_jsonl_as_dicts(SENSITIVE_POOL_JSONL)
    pilot_records = read_jsonl_as_dicts(PILOT_JSONL)

    sensitive_pool_df = route_records(
        records=sensitive_pool_records,
        dataset_name="sensy2_sensitive_pool",
    )

    pilot_df = route_records(
        records=pilot_records,
        dataset_name="sensy_refactor_candidate_pilot",
    )

    Path(OUTPUT_POOL_CSV).parent.mkdir(parents=True, exist_ok=True)

    sensitive_pool_df.to_csv(
        OUTPUT_POOL_CSV,
        index=False,
        encoding="utf-8",
    )

    pilot_df.to_csv(
        OUTPUT_PILOT_CSV,
        index=False,
        encoding="utf-8",
    )

    print_router_summary(
        df=sensitive_pool_df,
        title="ROUTER SUMMARY — SENSITIVE POOL",
    )

    print_router_summary(
        df=pilot_df,
        title="ROUTER SUMMARY — CANDIDATE PILOT",
    )

    print_section("PROMPT RISK ROUTER TEST COMPLETED")

    print("\nOutput files:")
    print(f"  {OUTPUT_POOL_CSV}")
    print(f"  {OUTPUT_PILOT_CSV}")


if __name__ == "__main__":
    main()