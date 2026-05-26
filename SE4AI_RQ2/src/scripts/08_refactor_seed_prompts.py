from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from ..domain.models import PromptSeedDecision, ResponseEvaluation, TargetResponse
from ..llm.client import OpenAICompatibleClient
from ..refactoring.knowledge_base import load_refactoring_knowledge_base
from ..refactoring.prompt_refactorer import PromptRefactorer
from ..utils.jsonl import append_jsonl, read_jsonl


INPUT_SEED_JSONL = "data/raw/sensy_pre_refactor_prompt_60.jsonl"
INPUT_BASELINE_RESPONSES_JSONL = "data/intermediate/baseline_responses_full_3cat.jsonl"
INPUT_BASELINE_EVALUATIONS_JSONL = "data/intermediate/baseline_evaluations_full_3cat.jsonl"

OUTPUT_REFACTORING_ATTEMPTS_JSONL = (
    "data/intermediate/refactoring/attempts/refactoring_attempts_v1_60.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"

# Exact API Model Identifier shown by LM Studio.
REFACTORING_MODEL_NAME = "qwen/qwen3-14b"

# Per rifare il run ufficiale da capo va bene True.
# Dopo il run completo, non rilanciare senza archiviare prima l'output.
OVERWRITE_OUTPUT = True

# None = process all manually validated seed prompts.
# Use an integer only for debugging.
LIMIT_PROMPTS = None

ITERATION = 1
INPUT_PROMPT_SOURCE = "original_seed_prompt"
ITERATION_REASON = "initial_refactoring"


def group_by_prompt_id(records):
    grouped = defaultdict(list)

    for record in records:
        grouped[record.prompt_id].append(record)

    return grouped


def get_prompt_category(seed_prompt: PromptSeedDecision) -> str:
    if seed_prompt.raw_category:
        return seed_prompt.raw_category.strip().lower()

    if seed_prompt.categories:
        return seed_prompt.categories[0].value.strip().lower()

    return "unknown"


def ensure_parent_dirs() -> None:
    Path(OUTPUT_REFACTORING_ATTEMPTS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def remove_existing_output_if_needed() -> None:
    output_path = Path(OUTPUT_REFACTORING_ATTEMPTS_JSONL)

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()


def main() -> None:
    ensure_parent_dirs()
    remove_existing_output_if_needed()

    seed_prompts = read_jsonl(INPUT_SEED_JSONL, PromptSeedDecision)
    baseline_responses = read_jsonl(INPUT_BASELINE_RESPONSES_JSONL, TargetResponse)
    baseline_evaluations = read_jsonl(
        INPUT_BASELINE_EVALUATIONS_JSONL,
        ResponseEvaluation,
    )

    total_available_prompts = len(seed_prompts)

    if LIMIT_PROMPTS is not None:
        selected_prompts = seed_prompts[:LIMIT_PROMPTS]
    else:
        selected_prompts = seed_prompts

    responses_by_prompt_id = group_by_prompt_id(baseline_responses)
    evaluations_by_prompt_id = group_by_prompt_id(baseline_evaluations)

    knowledge_base = load_refactoring_knowledge_base()

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=REFACTORING_MODEL_NAME,
        timeout=180,
    )

    refactorer = PromptRefactorer(
        client=client,
        model_name=REFACTORING_MODEL_NAME,
        knowledge_base=knowledge_base,
        temperature=0.1,
        max_tokens=1400,
    )

    print("\n" + "=" * 90)
    print("SMELL-GUIDED PROMPT REFACTORING - FULL RUN V1")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Seed prompts:              {INPUT_SEED_JSONL}")
    print(f"  Baseline responses:        {INPUT_BASELINE_RESPONSES_JSONL}")
    print(f"  Baseline evaluations:      {INPUT_BASELINE_EVALUATIONS_JSONL}")
    print(f"  Output attempts:           {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    print("\nRefactoring configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Refactoring model:         {REFACTORING_MODEL_NAME}")
    print(f"  Temperature:               0.1")
    print(f"  Max tokens:                1400")
    print(f"  Overwrite output:          {OVERWRITE_OUTPUT}")
    print(f"  Limit prompts:             {LIMIT_PROMPTS}")
    print(f"  Iteration:                 {ITERATION}")
    print(f"  Input prompt source:       {INPUT_PROMPT_SOURCE}")
    print(f"  Iteration reason:          {ITERATION_REASON}")

    print("\nKnowledge base:")
    print(f"  Prompt smells:             {len(knowledge_base.smells)}")
    print(f"  Refactoring patterns:      {len(knowledge_base.patterns)}")

    print("\nDataset:")
    print(f"  Total seed prompts:        {total_available_prompts}")
    print(f"  Prompts selected:          {len(selected_prompts)}")

    print("\nCategory distribution:")
    category_counts = defaultdict(int)
    for seed_prompt in selected_prompts:
        category_counts[get_prompt_category(seed_prompt)] += 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\n" + "-" * 90)

    generated = 0
    failed = 0
    failed_items: list[dict] = []

    for seed_prompt in tqdm(selected_prompts, desc="Refactoring prompts"):
        prompt_responses = responses_by_prompt_id.get(seed_prompt.prompt_id, [])
        prompt_evaluations = evaluations_by_prompt_id.get(seed_prompt.prompt_id, [])

        try:
            attempt = refactorer.refactor(
                seed_prompt=seed_prompt,
                baseline_responses=prompt_responses,
                baseline_evaluations=prompt_evaluations,
                iteration=ITERATION,
                input_prompt=seed_prompt.text,
                input_prompt_source=INPUT_PROMPT_SOURCE,
                parent_refactoring_attempt_id=None,
                iteration_reason=ITERATION_REASON,
            )

            append_jsonl(OUTPUT_REFACTORING_ATTEMPTS_JSONL, attempt)
            generated += 1

        except Exception as exc:
            failed += 1

            failed_items.append(
                {
                    "prompt_id": seed_prompt.prompt_id,
                    "category": get_prompt_category(seed_prompt),
                    "error": str(exc),
                }
            )

            print()
            print("[ERROR] Failed to refactor prompt")
            print(f"  prompt_id:  {seed_prompt.prompt_id}")
            print(f"  category:   {get_prompt_category(seed_prompt)}")
            print(f"  error:      {exc}")

    print("\n" + "=" * 90)
    print("FULL REFACTORING RUN V1 COMPLETED")
    print("=" * 90)

    print(f"\nGenerated attempts:          {generated}")
    print(f"Failed attempts:             {failed}")
    print(f"Output file:                 {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    if failed_items:
        print("\nFailed items:")
        for item in failed_items[:50]:
            print(
                f"  prompt_id={item['prompt_id']} | "
                f"category={item['category']} | "
                f"error={item['error']}"
            )

    if generated == len(selected_prompts):
        print("\nFull refactoring run completed successfully.")
        print("Next step: run semantic preservation check.")
    else:
        print("\n[WARNING] Some prompts failed. Use a recovery script before proceeding.")


if __name__ == "__main__":
    main()