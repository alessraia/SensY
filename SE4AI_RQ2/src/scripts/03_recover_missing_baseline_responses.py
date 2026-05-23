from ..domain.models import SensitivePrompt
from ..llm.client import OpenAICompatibleClient
from ..llm.target_llm import TargetLLM
from ..utils.jsonl import append_jsonl, read_jsonl


"""
Recupera singole risposte baseline mancanti senza rigenerare tutto il file.

Uso:
1. Lancia:
   python -m src.scripts.04_inspect_baseline_responses

2. Se trovi un prompt con repetition mancanti, inseriscilo in MISSING_ITEMS.

3. Lancia:
   python -m src.scripts.03_recover_missing_baseline_responses

4. Ricontrolla:
   python -m src.scripts.04_inspect_baseline_responses
"""


INPUT_JSONL = "data/intermediate/sensy_refactor_candidate_full_3cat.jsonl"
OUTPUT_JSONL = "data/intermediate/baseline_responses_full_3cat.jsonl"

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen2.5-7b-instruct"


# Inserisci qui le risposte mancanti da recuperare.
MISSING_ITEMS = [
    {"prompt_id": "sensy2_5281", "repetition": 2},
]


def main() -> None:
    prompts = read_jsonl(INPUT_JSONL, SensitivePrompt)
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
    )

    target_llm = TargetLLM(
        client=client,
        model_name=MODEL_NAME,
    )

    recovered = 0
    failed = 0

    print("\nMISSING BASELINE RESPONSE RECOVERY")
    print("-" * 80)

    for item in MISSING_ITEMS:
        prompt_id = item["prompt_id"]
        repetition = item["repetition"]

        prompt = prompt_by_id.get(prompt_id)

        if prompt is None:
            print(f"[ERROR] Prompt not found: {prompt_id}")
            failed += 1
            continue

        try:
            response = target_llm.answer(
                prompt=prompt,
                repetition=repetition,
            )

            append_jsonl(OUTPUT_JSONL, response)

            print(f"[OK] Recovered: prompt_id={prompt_id}, repetition={repetition}")
            recovered += 1

        except Exception as exc:
            print(f"[ERROR] Failed: prompt_id={prompt_id}, repetition={repetition}")
            print(f"        {exc}")
            failed += 1

    print("-" * 80)
    print(f"Recovered: {recovered}")
    print(f"Failed:    {failed}")
    print(f"Output:    {OUTPUT_JSONL}")
    print("\nNext: python -m src.scripts.04_inspect_baseline_responses")


if __name__ == "__main__":
    main()