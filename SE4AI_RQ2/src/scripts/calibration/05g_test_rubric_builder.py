import json
from pathlib import Path

from src.judge.risk_router import PromptRiskRouter
from src.judge.rubric_builder import RubricBuilder


PILOT_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"

LIMIT_PROMPTS = 10


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


def preview(text: str, max_chars: int = 600) -> str:
    text = text.replace("\n", " ").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_subsection(title: str) -> None:
    print("\n" + title)
    print("-" * 90)


def main() -> None:
    router = PromptRiskRouter()
    builder = RubricBuilder(max_addenda=4)

    records = read_jsonl_as_dicts(PILOT_JSONL)

    if LIMIT_PROMPTS is not None:
        records = records[:LIMIT_PROMPTS]

    print_section("MODULAR RUBRIC BUILDER TEST")

    print("\nInput file:")
    print(f"  {PILOT_JSONL}")

    print("\nConfiguration:")
    print(f"  Limit prompts: {LIMIT_PROMPTS}")
    print(f"  Max addenda:   {builder.max_addenda}")

    for i, record in enumerate(records, start=1):
        prompt_id = record.get("prompt_id")
        text = record.get("text", "")
        primary_category = get_primary_category(record)

        routing_result = router.route(text)
        selected_tags = builder.select_tags(routing_result.risk_tags)
        rubric = builder.build(routing_result.risk_tags)

        print_subsection(f"PROMPT {i}")

        print(f"prompt_id:          {prompt_id}")
        print(f"primary_category:   {primary_category}")
        print(f"risk_tags:          {[tag.value for tag in routing_result.risk_tags]}")
        print(f"selected_tags:      {[tag.value for tag in selected_tags]}")
        print(f"only_general:       {routing_result.only_general}")
        print(f"rubric_length:      {len(rubric)} characters")
        print(f"prompt_text:        {preview(text, max_chars=300)}")

        print("\nRubric preview:")
        print(preview(rubric, max_chars=900))

    print_section("MODULAR RUBRIC BUILDER TEST COMPLETED")


if __name__ == "__main__":
    main()