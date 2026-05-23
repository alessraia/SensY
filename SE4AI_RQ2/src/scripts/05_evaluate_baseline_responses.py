from pathlib import Path

from tqdm import tqdm

from ..domain.models import SensitivePrompt, TargetResponse
from ..judge.llm_judge import LLMJudge
from ..llm.client import OpenAICompatibleClient
from ..utils.jsonl import append_jsonl, read_jsonl


#PROMPTS_JSONL = "data/intermediate/sensy_refactor_candidate_pilot.jsonl"
#RESPONSES_JSONL = "data/intermediate/baseline_responses_pilot.jsonl"
#OUTPUT_JSONL = "data/intermediate/baseline_evaluations_pilot.jsonl"
PROMPTS_JSONL = "data/intermediate/sensy_refactor_candidate_full_3cat.jsonl"
RESPONSES_JSONL = "data/intermediate/baseline_responses_full_3cat.jsonl"
OUTPUT_JSONL = "data/intermediate/baseline_evaluations_full_3cat.jsonl"

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

# Sostituisci con il model id esatto mostrato da LM Studio.
JUDGE_MODEL_NAME = "deepseek-r1-distill-qwen-32b"

OVERWRITE_OUTPUT = True

# Per test iniziale lascia 6.
# Quando funziona, cambia in None per valutare tutte le risposte baseline.
LIMIT_RESPONSES = None


def main() -> None:
    prompts = read_jsonl(PROMPTS_JSONL, SensitivePrompt)
    responses = read_jsonl(RESPONSES_JSONL, TargetResponse)

    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}

    total_available_responses = len(responses)

    if LIMIT_RESPONSES is not None:
        responses = responses[:LIMIT_RESPONSES]

    output_path = Path(OUTPUT_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=JUDGE_MODEL_NAME,
        timeout=240,
    )

    judge = LLMJudge(
        client=client,
        model_name=JUDGE_MODEL_NAME,
    )

    print("\n" + "=" * 90)
    print("BASELINE RESPONSE EVALUATION")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Prompts:                    {PROMPTS_JSONL}")
    print(f"  Responses:                  {RESPONSES_JSONL}")
    print(f"  Output evaluations:         {OUTPUT_JSONL}")

    print("\nJudge configuration:")
    print(f"  Base URL:                   {BASE_URL}")
    print(f"  Judge model:                {JUDGE_MODEL_NAME}")
    print(f"  Temperature:                0.0")
    print(f"  Max tokens:                 768")
    print(f"  Overwrite output:           {OVERWRITE_OUTPUT}")
    print(f"  Limit responses:            {LIMIT_RESPONSES}")

    print("\nDataset:")
    print(f"  Total responses available:  {total_available_responses}")
    print(f"  Responses selected:         {len(responses)}")

    print("\n" + "-" * 90)

    evaluated = 0
    failed = 0
    failed_items = []

    for response in tqdm(responses, desc="Evaluating baseline responses"):
        prompt = prompt_by_id.get(response.prompt_id)

        if prompt is None:
            failed += 1
            failed_items.append(
                {
                    "response_id": response.response_id,
                    "prompt_id": response.prompt_id,
                    "error": "Prompt not found",
                }
            )
            continue


        try:
            evaluation = judge.evaluate_baseline_response(
                prompt=prompt,
                response=response,
            )

            append_jsonl(OUTPUT_JSONL, evaluation)
            evaluated += 1

        except Exception as exc:
            failed += 1
            failed_items.append(
                {
                    "response_id": response.response_id,
                    "prompt_id": response.prompt_id,
                    "error": str(exc),
                }
            )

            print()
            print("[ERROR] Failed to evaluate response")
            print(f"  response_id: {response.response_id}")
            print(f"  prompt_id:   {response.prompt_id}")
            print(f"  repetition:  {response.repetition}")
            print(f"  error:       {exc}")

    print("\n" + "=" * 90)
    print("BASELINE EVALUATION COMPLETED")
    print("=" * 90)

    print(f"\nEvaluated responses:          {evaluated}")
    print(f"Failed evaluations:           {failed}")
    print(f"Expected evaluations:         {len(responses)}")
    print(f"Output file:                  {OUTPUT_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:20]:
            print(
                f"  response_id={item['response_id']} | "
                f"prompt_id={item['prompt_id']} | "
                f"error={item['error']}"
            )

        if len(failed_items) > 20:
            print(f"  ... and {len(failed_items) - 20} more failures.")

    if evaluated != len(responses):
        print("\n[WARNING] Some responses were not evaluated.")
    else:
        print("\nAll selected baseline responses were evaluated successfully.")

    print("\nNext step:")
    print("  Run: python -m src.scripts.06_inspect_baseline_evaluations")


if __name__ == "__main__":
    main()