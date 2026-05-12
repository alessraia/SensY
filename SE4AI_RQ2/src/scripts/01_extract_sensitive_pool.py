import pandas as pd

from ..data.dataset_loader import SensYDatasetLoader
from ..domain.models import SensitivityCategory
from ..utils.jsonl import write_jsonl


INPUT_PATH = "data/raw/dataset_SENSY2.0.json"

OUTPUT_JSONL = "data/intermediate/sensy2_sensitive_pool.jsonl"
OUTPUT_CSV = "data/intermediate/sensy2_sensitive_pool.csv"


def is_recognized_category(categories: list[SensitivityCategory]) -> bool:
    """
    Ritorna True se almeno una categoria normalizzata è diversa da OTHER.
    """
    if not categories:
        return False

    return any(category != SensitivityCategory.OTHER for category in categories)


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def print_counter(series: pd.Series, title: str, max_rows: int | None = None) -> None:
    """
    Stampa una distribuzione in modo leggibile.
    """
    print(format_subsection(title))

    if series.empty:
        print("No data.")
        return

    counts = series.value_counts(dropna=False)

    if max_rows is not None:
        counts = counts.head(max_rows)

    max_label_len = max(len(str(index)) for index in counts.index)
    max_count_len = max(len(str(value)) for value in counts.values)

    for index, value in counts.items():
        print(f"{str(index):<{max_label_len}}  {str(value):>{max_count_len}}")


def print_examples(df: pd.DataFrame, title: str, columns: list[str], max_rows: int = 10) -> None:
    """
    Stampa esempi leggibili evitando output enorme sul terminale.
    """
    print(format_subsection(title))

    if df.empty:
        print("No examples.")
        return

    available_columns = [col for col in columns if col in df.columns]
    examples = df[available_columns].head(max_rows)

    for i, (_, row) in enumerate(examples.iterrows(), start=1):
        print(f"\n[{i}]")
        for col in available_columns:
            value = row[col]
            value = "" if pd.isna(value) else str(value)

            if col == "text" and len(value) > 220:
                value = value[:220] + "..."

            print(f"{col}: {value}")


def main() -> None:
    loader = SensYDatasetLoader()

    # Carichiamo sia tutto il dataset sia il pool sensitive.
    # Questo ci permette di stampare anche i conteggi globali.
    all_prompts = loader.load_all_prompts(INPUT_PATH)
    sensitive_prompts = loader.load_sensitive_prompts(INPUT_PATH)

    write_jsonl(OUTPUT_JSONL, sensitive_prompts)

    rows = []

    for prompt in sensitive_prompts:
        category_recognized = is_recognized_category(prompt.categories)
        num_categories = len(prompt.categories)

        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "categories": "|".join(category.value for category in prompt.categories),
                "num_categories": num_categories,
                "primary_category": prompt.categories[0].value if prompt.categories else None,
                "raw_category": prompt.raw_category,
                "category_present": getattr(prompt, "category_present", prompt.raw_category is not None),
                "category_recognized": category_recognized,
                "subcategory": prompt.subcategory,
                "style_type": prompt.style_type,
                "source": prompt.source,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    total_prompts = len(all_prompts)
    total_sensitive = len(sensitive_prompts)
    total_non_sensitive = total_prompts - total_sensitive

    categorized_sensitive = int(df["category_present"].sum()) if not df.empty else 0
    uncategorized_sensitive = total_sensitive - categorized_sensitive

    recognized_sensitive = int(df["category_recognized"].sum()) if not df.empty else 0
    unrecognized_sensitive = total_sensitive - recognized_sensitive

    multi_category_count = int((df["num_categories"] > 1).sum()) if not df.empty else 0

    category_present_but_unrecognized = df[
        (df["category_present"] == True) & (df["category_recognized"] == False)
    ]

    sensitive_without_category = df[df["category_present"] == False]

    multi_category_df = df[df["num_categories"] > 1]

    print(format_section("SENSY 2.0 — SENSITIVE POOL EXTRACTION"))

    print("\nInput / output files:")
    print(f"  Input dataset:        {INPUT_PATH}")
    print(f"  Output JSONL:         {OUTPUT_JSONL}")
    print(f"  Output CSV:           {OUTPUT_CSV}")

    print(format_subsection("GLOBAL COUNTS"))
    print(f"Total prompts:                         {total_prompts}")
    print(f"Sensitive prompts:                     {total_sensitive}")
    print(f"Non-sensitive prompts:                 {total_non_sensitive}")

    if total_prompts > 0:
        print(f"Sensitive percentage:                  {total_sensitive / total_prompts * 100:.2f}%")
        print(f"Non-sensitive percentage:              {total_non_sensitive / total_prompts * 100:.2f}%")

    print(format_subsection("CATEGORY FIELD AMONG SENSITIVE PROMPTS"))
    print(f"Sensitive with category field:         {categorized_sensitive}")
    print(f"Sensitive without category field:      {uncategorized_sensitive}")

    if total_sensitive > 0:
        print(f"With category field (%):               {categorized_sensitive / total_sensitive * 100:.2f}%")
        print(f"Without category field (%):            {uncategorized_sensitive / total_sensitive * 100:.2f}%")

    print(format_subsection("CATEGORY NORMALIZATION"))
    print(f"Sensitive with recognized category:    {recognized_sensitive}")
    print(f"Sensitive without recognized category: {unrecognized_sensitive}")

    if total_sensitive > 0:
        print(f"Recognized among sensitive (%):        {recognized_sensitive / total_sensitive * 100:.2f}%")
        print(f"Not recognized among sensitive (%):    {unrecognized_sensitive / total_sensitive * 100:.2f}%")

    if categorized_sensitive > 0:
        recognized_among_categorized = len(
            df[(df["category_present"] == True) & (df["category_recognized"] == True)]
        )
        unrecognized_among_categorized = len(category_present_but_unrecognized)

        print()
        print(f"Recognized among categorized:          {recognized_among_categorized}")
        print(f"Unrecognized among categorized:        {unrecognized_among_categorized}")
        print(f"Recognized among categorized (%):      {recognized_among_categorized / categorized_sensitive * 100:.2f}%")
        print(f"Unrecognized among categorized (%):    {unrecognized_among_categorized / categorized_sensitive * 100:.2f}%")

    print(format_subsection("MULTI-CATEGORY PROMPTS"))
    print(f"Sensitive prompts with >1 category:    {multi_category_count}")

    if total_sensitive > 0:
        print(f"Multi-category among sensitive (%):    {multi_category_count / total_sensitive * 100:.2f}%")

    print_counter(
        df["primary_category"],
        title="DISTRIBUTION BY PRIMARY CATEGORY",
    )

    print_counter(
        df["num_categories"],
        title="DISTRIBUTION BY NUMBER OF NORMALIZED CATEGORIES",
    )

    print_counter(
        df["raw_category"].dropna(),
        title="RAW CATEGORY VALUES",
        max_rows=100,
    )

    if not category_present_but_unrecognized.empty:
        print_counter(
            category_present_but_unrecognized["raw_category"],
            title="RAW CATEGORY VALUES PRESENT BUT NOT RECOGNIZED",
            max_rows=100,
        )
    else:
        print(format_subsection("RAW CATEGORY VALUES PRESENT BUT NOT RECOGNIZED"))
        print("None. All present category values are recognized.")

    print_examples(
        multi_category_df,
        title="EXAMPLES OF MULTI-CATEGORY PROMPTS",
        columns=[
            "prompt_id",
            "raw_category",
            "categories",
            "primary_category",
            "text",
        ],
        max_rows=10,
    )

    print_examples(
        category_present_but_unrecognized,
        title="EXAMPLES OF CATEGORY PRESENT BUT NOT RECOGNIZED",
        columns=[
            "prompt_id",
            "raw_category",
            "categories",
            "primary_category",
            "text",
        ],
        max_rows=10,
    )

    print_examples(
        sensitive_without_category,
        title="EXAMPLES OF SENSITIVE PROMPTS WITHOUT CATEGORY FIELD",
        columns=[
            "prompt_id",
            "raw_category",
            "categories",
            "primary_category",
            "text",
        ],
        max_rows=10,
    )

    print(format_section("EXTRACTION COMPLETED"))

    print("\nFiles created:")
    print(f"  {OUTPUT_JSONL}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()