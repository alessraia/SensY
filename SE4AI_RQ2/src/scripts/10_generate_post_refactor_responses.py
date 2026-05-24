import json
from pathlib import Path

from tqdm import tqdm

from ..llm.client import OpenAICompatibleClient
from ..refactoring.post_refactor_response_generator import (
    PostRefactorResponseGenerator,
)
from ..refactoring.refactoring_models import RefactoringAttempt
from ..utils.jsonl import read_jsonl


INPUT_ACCEPTED_REFACTORINGS_JSONL = (
    "data/intermediate/refactoring/accepted_refactorings/"
    "accepted_refactoring_attempts_v1.jsonl"
)

OUTPUT_POST_REFACTOR_RESPONSES_JSONL = (
    "data/intermediate/refactoring/post_refactor_responses/"
    "post_refactor_responses_v1.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

# IMPORTANT:
# Usa qui lo stesso target model della baseline.
# Controlla in LM Studio il valore esatto di "API Model Identifier".
# Non usare qwen/qwen3-14b.
TARGET_MODEL_NAME = "qwen2.5-7b-instruct"

OVERWRITE_OUTPUT = True

REPETITIONS_PER_PROMPT = 3

TARGET_TEMPERATURE = 0.2
TARGET_MAX_TOKENS = 256

LIMIT_PROMPTS = None


def ensure_parent_dirs() -> None:
    Path(OUTPUT_POST_REFACTOR_RESPONSES_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_output_if_needed() -> None:
    output_path = Path(OUTPUT_POST_REFACTOR_RESPONSES_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()


def append_dict_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def main() -> None:
    ensure_parent_dirs()
    remove_existing_output_if_needed()

    accepted_attempts = read_jsonl(
        INPUT_ACCEPTED_REFACTORINGS_JSONL,
        RefactoringAttempt,
    )

    total_available_attempts = len(accepted_attempts)

    if LIMIT_PROMPTS is not None:
        selected_attempts = accepted_attempts[:LIMIT_PROMPTS]
    else:
        selected_attempts = accepted_attempts

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=TARGET_MODEL_NAME,
        timeout=180,
    )

    generator = PostRefactorResponseGenerator(
        client=client,
        model_name=TARGET_MODEL_NAME,
        temperature=TARGET_TEMPERATURE,
        max_tokens=TARGET_MAX_TOKENS,
    )

    print("\n" + "=" * 90)
    print("POST-REFACTOR TARGET RESPONSE GENERATION - V1")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Accepted refactorings:     {INPUT_ACCEPTED_REFACTORINGS_JSONL}")
    print(f"  Output responses:          {OUTPUT_POST_REFACTOR_RESPONSES_JSONL}")

    print("\nTarget model configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Target model:              {TARGET_MODEL_NAME}")
    print(f"  Temperature:               {TARGET_TEMPERATURE}")
    print(f"  Max tokens:                {TARGET_MAX_TOKENS}")
    print(f"  Repetitions per prompt:    {REPETITIONS_PER_PROMPT}")
    print(f"  Overwrite output:          {OVERWRITE_OUTPUT}")
    print(f"  Limit prompts:             {LIMIT_PROMPTS}")

    print("\nDataset:")
    print(f"  Total accepted attempts:   {total_available_attempts}")
    print(f"  Attempts selected:         {len(selected_attempts)}")
    print(f"  Expected responses:        {len(selected_attempts) * REPETITIONS_PER_PROMPT}")

    print("\n" + "-" * 90)

    generated = 0
    failed = 0
    failed_items: list[dict] = []

    for attempt in tqdm(selected_attempts, desc="Generating post-refactor responses"):
        for repetition in range(1, REPETITIONS_PER_PROMPT + 1):
            try:
                record = generator.generate_response_record(
                    attempt=attempt,
                    repetition=repetition,
                )

                append_dict_jsonl(OUTPUT_POST_REFACTOR_RESPONSES_JSONL, record)
                generated += 1

            except Exception as exc:
                failed += 1
                failed_items.append(
                    {
                        "prompt_id": attempt.prompt_id,
                        "refactoring_attempt_id": attempt.refactoring_attempt_id,
                        "repetition": repetition,
                        "error": str(exc),
                    }
                )

                print()
                print("[ERROR] Failed to generate post-refactor response")
                print(f"  prompt_id:                {attempt.prompt_id}")
                print(f"  refactoring_attempt_id:   {attempt.refactoring_attempt_id}")
                print(f"  repetition:               {repetition}")
                print(f"  error:                    {exc}")

    print("\n" + "=" * 90)
    print("POST-REFACTOR TARGET RESPONSE GENERATION COMPLETED")
    print("=" * 90)

    print(f"\nGenerated responses:         {generated}")
    print(f"Failed responses:            {failed}")
    print(f"Output file:                 {OUTPUT_POST_REFACTOR_RESPONSES_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:50]:
            print(
                f"  prompt_id={item['prompt_id']} | "
                f"attempt_id={item['refactoring_attempt_id']} | "
                f"repetition={item['repetition']} | "
                f"error={item['error']}"
            )

    expected = len(selected_attempts) * REPETITIONS_PER_PROMPT

    if generated == expected:
        print("\nPost-refactor response generation completed successfully.")
        print("Next step: evaluate these responses with the modular judge.")
    else:
        print("\n[WARNING] Some responses failed. Inspect the errors above.")


if __name__ == "__main__":
    main()