import pandas as pd

from ..data.sampler import SensitivePromptSampler
from ..domain.models import SensitivePrompt
from ..utils.jsonl import read_jsonl, write_jsonl


INPUT_JSONL = "data/intermediate/sensy2_sensitive_pool.jsonl"

OUTPUT_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"
OUTPUT_CSV = "data/intermediate/sensy_refactor_candidate_pilot.csv"

N_PER_CATEGORY = 30
RANDOM_STATE = 42


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def print_distribution(df: pd.DataFrame, column: str, title: str) -> None:
    print(format_subsection(title))

    if df.empty or column not in df.columns:
        print("No data.")
        return

    counts = df[column].value_counts(dropna=False)

    max_label_len = max(len(str(index)) for index in counts.index)
    max_count_len = max(len(str(value)) for value in counts.values)

    for index, value in counts.items():
        print(f"{str(index):<{max_label_len}}  {str(value):>{max_count_len}}")


def print_examples(df: pd.DataFrame, title: str, max_rows: int = 10) -> None:
    print(format_subsection(title))

    if df.empty:
        print("No examples.")
        return

    columns = [
        "prompt_id",
        "primary_category",
        "categories",
        "raw_category",
        "text",
    ]

    available_columns = [col for col in columns if col in df.columns]

    for i, (_, row) in enumerate(df[available_columns].head(max_rows).iterrows(), start=1):
        print(f"\n[{i}]")
        for col in available_columns:
            value = row[col]
            value = "" if pd.isna(value) else str(value)

            if col == "text" and len(value) > 220:
                value = value[:220] + "..."

            print(f"{col}: {value}")


def main() -> None:
    prompts = read_jsonl(INPUT_JSONL, SensitivePrompt)

    sampler = SensitivePromptSampler(random_state=RANDOM_STATE)

    sampled = sampler.sample_balanced(
        prompts=prompts,
        n_per_category=N_PER_CATEGORY,
        include_other=False,
    )

    write_jsonl(OUTPUT_JSONL, sampled)

    rows = [
        {
            "prompt_id": p.prompt_id,
            "text": p.text,
            "categories": "|".join(category.value for category in p.categories),
            "num_categories": len(p.categories),
            "primary_category": p.categories[0].value if p.categories else None,
            "raw_category": p.raw_category,
            "subcategory": p.subcategory,
            "style_type": p.style_type,
            "source": p.source,
        }
        for p in sampled
    ]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(format_section("SENSY-REFACTOR CANDIDATE PILOT CREATION"))

    print("\nInput / output files:")
    print(f"  Input JSONL:          {INPUT_JSONL}")
    print(f"  Output JSONL:         {OUTPUT_JSONL}")
    print(f"  Output CSV:           {OUTPUT_CSV}")

    print(format_subsection("SAMPLING CONFIGURATION"))
    print(f"Prompts per category:                 {N_PER_CATEGORY}")
    print(f"Random state:                         {RANDOM_STATE}")
    print(f"Include Other category:               False")
    print(f"Expected size if all categories exist: {N_PER_CATEGORY * 7}")

    print(format_subsection("PILOT SET SIZE"))
    print(f"Candidate pilot prompts:              {len(sampled)}")

    print_distribution(
        df,
        column="primary_category",
        title="DISTRIBUTION BY PRIMARY CATEGORY",
    )

    multi_category_count = int((df["num_categories"] > 1).sum()) if not df.empty else 0

    print(format_subsection("MULTI-CATEGORY PROMPTS IN PILOT"))
    print(f"Multi-category prompts:               {multi_category_count}")

    if len(sampled) > 0:
        print(f"Multi-category percentage:            {multi_category_count / len(sampled) * 100:.2f}%")

    multi_category_df = df[df["num_categories"] > 1]
    print_examples(
        multi_category_df,
        title="EXAMPLES OF MULTI-CATEGORY PROMPTS IN PILOT",
        max_rows=10,
    )

    print_examples(
        df,
        title="FIRST PROMPTS IN PILOT SET",
        max_rows=10,
    )

    print(format_section("CANDIDATE PILOT CREATION COMPLETED"))

    print("\nFiles created:")
    print(f"  {OUTPUT_JSONL}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()