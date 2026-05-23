from pathlib import Path

from tqdm import tqdm

from ..domain.models import SensitivePrompt
from ..llm.client import OpenAICompatibleClient
from ..llm.target_llm import TargetLLM
from ..utils.jsonl import append_jsonl, read_jsonl


#INPUT_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"
#OUTPUT_JSONL = "data/intermediate/baseline_responses_pilot.jsonl"
INPUT_JSONL = "data/intermediate/sensy_refactor_candidate_full_3cat.jsonl"
OUTPUT_JSONL = "data/intermediate/baseline_responses_full_3cat.jsonl"

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen2.5-7b-instruct"

REPETITIONS = 3

OVERWRITE_OUTPUT = True

# Per il test iniziale lascia 3.
# Quando il test funziona, cambia in None per generare tutte le risposte:
LIMIT_PROMPTS = None



def main() -> None:
    prompts = read_jsonl(INPUT_JSONL, SensitivePrompt)

    total_available_prompts = len(prompts)

    if LIMIT_PROMPTS is not None:
        prompts = prompts[:LIMIT_PROMPTS]

    output_path = Path(OUTPUT_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
    )

    target_llm = TargetLLM(
        client=client,
        model_name=MODEL_NAME,
    )

    expected_outputs = len(prompts) * REPETITIONS

    print("\n" + "=" * 90)
    print("BASELINE RESPONSE GENERATION")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Input prompts:             {INPUT_JSONL}")
    print(f"  Output responses:          {OUTPUT_JSONL}")

    print("\nGeneration configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Target model:              {MODEL_NAME}")
    print(f"  Repetitions per prompt:    {REPETITIONS}")
    print(f"  Temperature:               0.2")
    print(f"  Max tokens:                256")
    print(f"  Overwrite output:          {OVERWRITE_OUTPUT}")
    print(f"  Limit prompts:             {LIMIT_PROMPTS}")

    print("\nDataset:")
    print(f"  Total prompts available:   {total_available_prompts}")
    print(f"  Prompts selected:          {len(prompts)}")
    print(f"  Expected responses:        {expected_outputs}")

    print("\n" + "-" * 90)

    generated = 0
    failed = 0

    failed_items: list[dict] = []

    for prompt in tqdm(prompts, desc="Generating baseline responses"):
        for repetition in range(REPETITIONS):
            try:
                response = target_llm.answer(
                    prompt=prompt,
                    repetition=repetition,
                )

                append_jsonl(OUTPUT_JSONL, response)
                generated += 1

            except Exception as exc:
                failed += 1

                failed_items.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "repetition": repetition,
                        "error": str(exc),
                    }
                )

                print()
                print("[ERROR] Failed to generate response")
                print(f"  prompt_id:  {prompt.prompt_id}")
                print(f"  repetition: {repetition}")
                print(f"  error:      {exc}")

    print("\n" + "=" * 90)
    print("BASELINE GENERATION COMPLETED")
    print("=" * 90)

    print(f"\nGenerated responses:         {generated}")
    print(f"Failed generations:          {failed}")
    print(f"Expected responses:          {expected_outputs}")
    print(f"Output file:                 {OUTPUT_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:20]:
            print(
                f"  prompt_id={item['prompt_id']} | "
                f"repetition={item['repetition']} | "
                f"error={item['error']}"
            )

        if len(failed_items) > 20:
            print(f"  ... and {len(failed_items) - 20} more failures.")

    if generated != expected_outputs:
        print("\n[WARNING] Generated responses do not match expected responses.")
        print("Some prompts may have failed. Check errors above.")
    else:
        print("\nAll expected baseline responses were generated successfully.")

    print("\nNext step:")
    print("  Run: python -m src.scripts.04_inspect_baseline_responses")


if __name__ == "__main__":
    main()