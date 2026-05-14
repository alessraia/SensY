from collections import Counter, defaultdict

from ..domain.models import TargetResponse
from ..utils.jsonl import read_jsonl


INPUT_JSONL = "data/intermediate/baseline_responses_pilot.jsonl"

EXPECTED_REPETITIONS = 3


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def main() -> None:
    responses = read_jsonl(INPUT_JSONL, TargetResponse)

    print(format_section("BASELINE RESPONSES INSPECTION"))

    print("\nInput file:")
    print(f"  {INPUT_JSONL}")

    print(format_subsection("BASIC COUNTS"))

    print(f"Responses loaded:                  {len(responses)}")

    if not responses:
        print("\nNo responses found. Nothing to inspect.")
        return

    prompt_ids = [response.prompt_id for response in responses]
    target_models = [response.target_model for response in responses]
    repetitions = [response.repetition for response in responses]

    unique_prompt_ids = sorted(set(prompt_ids))
    unique_target_models = sorted(set(target_models))
    unique_repetitions = sorted(set(repetitions))

    print(f"Unique prompt_ids:                 {len(unique_prompt_ids)}")
    print(f"Target models:                     {len(unique_target_models)}")
    print(f"Repetitions found:                 {unique_repetitions}")

    expected_total_if_complete = len(unique_prompt_ids) * EXPECTED_REPETITIONS

    print(f"Expected repetitions per prompt:   {EXPECTED_REPETITIONS}")
    print(f"Expected total responses:          {expected_total_if_complete}")

    print(format_subsection("RESPONSES BY MODEL"))

    model_counts = Counter(target_models)

    for model, count in model_counts.items():
        print(f"{model}: {count}")

    print(format_subsection("REPETITION DISTRIBUTION"))

    repetition_counts = Counter(repetitions)

    for repetition, count in sorted(repetition_counts.items()):
        print(f"repetition {repetition}: {count}")

    print(format_subsection("RESPONSES PER PROMPT CHECK"))

    responses_per_prompt = Counter(prompt_ids)

    prompts_with_expected_repetitions = [
        prompt_id
        for prompt_id, count in responses_per_prompt.items()
        if count == EXPECTED_REPETITIONS
    ]

    prompts_with_missing_or_extra_repetitions = {
        prompt_id: count
        for prompt_id, count in responses_per_prompt.items()
        if count != EXPECTED_REPETITIONS
    }

    print(f"Prompts with expected repetitions:     {len(prompts_with_expected_repetitions)}")
    print(f"Prompts with missing/extra responses:  {len(prompts_with_missing_or_extra_repetitions)}")

    if prompts_with_missing_or_extra_repetitions:
        print("\nProblematic prompt repetition counts:")
        for prompt_id, count in list(prompts_with_missing_or_extra_repetitions.items())[:30]:
            print(f"  {prompt_id}: {count}")

        if len(prompts_with_missing_or_extra_repetitions) > 30:
            remaining = len(prompts_with_missing_or_extra_repetitions) - 30
            print(f"  ... and {remaining} more.")

    print(format_subsection("REPETITION IDS PER PROMPT CHECK"))

    repetitions_by_prompt: dict[str, set[int]] = defaultdict(set)

    for response in responses:
        repetitions_by_prompt[response.prompt_id].add(response.repetition)

    expected_repetition_set = set(range(EXPECTED_REPETITIONS))

    prompts_with_wrong_repetition_ids = {
        prompt_id: sorted(rep_set)
        for prompt_id, rep_set in repetitions_by_prompt.items()
        if rep_set != expected_repetition_set
    }

    print(f"Expected repetition ids:               {sorted(expected_repetition_set)}")
    print(f"Prompts with wrong repetition ids:      {len(prompts_with_wrong_repetition_ids)}")

    if prompts_with_wrong_repetition_ids:
        print("\nExamples:")
        for prompt_id, rep_ids in list(prompts_with_wrong_repetition_ids.items())[:30]:
            print(f"  {prompt_id}: {rep_ids}")

        if len(prompts_with_wrong_repetition_ids) > 30:
            remaining = len(prompts_with_wrong_repetition_ids) - 30
            print(f"  ... and {remaining} more.")

    print(format_subsection("QUALITY CHECKS"))

    empty_responses = [
        response
        for response in responses
        if not response.response_text.strip()
    ]

    very_short_responses = [
        response
        for response in responses
        if 0 < len(response.response_text.strip()) < 5
    ]

    print(f"Empty responses:                       {len(empty_responses)}")
    print(f"Very short responses (<5 chars):        {len(very_short_responses)}")

    if empty_responses:
        print("\nEmpty response examples:")
        for response in empty_responses[:10]:
            print(
                f"  response_id={response.response_id} | "
                f"prompt_id={response.prompt_id} | "
                f"repetition={response.repetition}"
            )

    print(format_subsection("FIRST 5 RESPONSES"))

    for i, response in enumerate(responses[:5], start=1):
        print(f"\n[{i}]")
        print(f"response_id:   {response.response_id}")
        print(f"prompt_id:     {response.prompt_id}")
        print(f"target_model:  {response.target_model}")
        print(f"repetition:    {response.repetition}")
        print(f"source:        {response.source}")
        print(f"prompt_text:   {truncate(response.prompt_text, 250)}")
        print(f"response_text: {truncate(response.response_text, 500)}")

    print(format_section("INSPECTION COMPLETED"))

    print("\nSummary:")
    print(f"  Responses loaded:                  {len(responses)}")
    print(f"  Unique prompt_ids:                 {len(unique_prompt_ids)}")
    print(f"  Expected repetitions per prompt:   {EXPECTED_REPETITIONS}")
    print(f"  Empty responses:                   {len(empty_responses)}")
    print(f"  Prompts with missing/extra outputs:{len(prompts_with_missing_or_extra_repetitions)}")
    print(f"  Prompts with wrong repetition ids: {len(prompts_with_wrong_repetition_ids)}")

    if (
        len(empty_responses) == 0
        and len(prompts_with_missing_or_extra_repetitions) == 0
        and len(prompts_with_wrong_repetition_ids) == 0
    ):
        print("\nBaseline responses file looks complete and consistent.")
    else:
        print("\n[WARNING] Baseline responses file has issues. Check sections above.")


if __name__ == "__main__":
    main()