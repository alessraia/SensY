import pandas as pd
from collections import Counter

from ..domain.models import SensitivePrompt, SensitivityCategory
from ..utils.jsonl import read_jsonl, write_jsonl


INPUT_JSONL = "data/intermediate/sensy2_sensitive_pool.jsonl"

OUTPUT_JSONL = "data/intermediate/sensy_refactor_candidate_full_3cat.jsonl"
OUTPUT_CSV = "data/intermediate/sensy_refactor_candidate_full_3cat.csv"


TARGET_PRIMARY_CATEGORIES = {
    SensitivityCategory.SECURITY,
    SensitivityCategory.HEALTH_MENTAL_WELLBEING,
    SensitivityCategory.IDENTITY_DIVERSITY,
}


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def main() -> None:
    prompts = read_jsonl(INPUT_JSONL, SensitivePrompt)

    selected: list[SensitivePrompt] = []

    for prompt in prompts:
        if not prompt.categories:
            continue

        primary_category = prompt.categories[0]

        if primary_category in TARGET_PRIMARY_CATEGORIES:
            selected.append(prompt)

    write_jsonl(OUTPUT_JSONL, selected)

    rows = []

    for prompt in selected:
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "categories": "|".join(category.value for category in prompt.categories),
                "num_categories": len(prompt.categories),
                "primary_category": prompt.categories[0].value if prompt.categories else None,
                "raw_category": prompt.raw_category,
                "subcategory": prompt.subcategory,
                "style_type": prompt.style_type,
                "source": prompt.source,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(format_section("SENSY-REFACTOR FULL 3-CATEGORY CANDIDATE CREATION"))

    print("\nInput / output files:")
    print(f"  Input JSONL:          {INPUT_JSONL}")
    print(f"  Output JSONL:         {OUTPUT_JSONL}")
    print(f"  Output CSV:           {OUTPUT_CSV}")

    print(format_subsection("SELECTION CONFIGURATION"))
    print("Selected primary categories:")
    for category in TARGET_PRIMARY_CATEGORIES:
        print(f"  - {category.value}")

    print(format_subsection("OUTPUT SIZE"))
    print(f"Selected prompts:       {len(selected)}")

    print(format_subsection("DISTRIBUTION BY PRIMARY CATEGORY"))

    counts = Counter(
        prompt.categories[0].value
        for prompt in selected
        if prompt.categories
    )

    for category, count in counts.items():
        print(f"{category}: {count}")

    print(format_subsection("MULTI-CATEGORY PROMPTS"))

    multi_category_count = sum(1 for prompt in selected if len(prompt.categories) > 1)
    print(f"Multi-category prompts: {multi_category_count}")

    if selected:
        print(f"Multi-category percentage: {multi_category_count / len(selected) * 100:.2f}%")

    print(format_section("FULL 3-CATEGORY CANDIDATE CREATION COMPLETED"))

    print("\nFiles created:")
    print(f"  {OUTPUT_JSONL}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()