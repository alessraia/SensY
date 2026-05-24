from collections import defaultdict
from pathlib import Path

from ..domain.models import PromptSeedDecision, ResponseEvaluation, TargetResponse
from ..llm.client import OpenAICompatibleClient
from ..refactoring.knowledge_base import load_refactoring_knowledge_base
from ..refactoring.prompt_refactorer import PromptRefactorer
from ..refactoring.refactoring_models import RefactoringAttempt
from ..utils.jsonl import append_jsonl, read_jsonl


"""
Recupera singoli refactoring attempt mancanti senza rigenerare tutto il file.

Uso:
1. Esegui il run completo:
   python -m src.scripts.08_refactor_seed_prompts

2. Se alcuni prompt falliscono, inserisci i loro prompt_id in MISSING_PROMPT_IDS.

3. Lancia:
   python -m src.scripts.08_recover_missing_refactoring_attempts

4. Ricontrolla il numero di righe:
   (Get-Content data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl).Count

Questo script NON sovrascrive il file ufficiale.
Aggiunge solo i refactoring mancanti.
"""


INPUT_SEED_JSONL = "data/raw/sensy_pre_refactor_prompt.jsonl"
INPUT_BASELINE_RESPONSES_JSONL = "data/intermediate/baseline_responses_full_3cat.jsonl"
INPUT_BASELINE_EVALUATIONS_JSONL = "data/intermediate/baseline_evaluations_full_3cat.jsonl"

OUTPUT_REFACTORING_ATTEMPTS_JSONL = (
    "data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl"
)

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
REFACTORING_MODEL_NAME = "qwen/qwen3-14b"

# Inserisci qui i prompt_id falliti nel run dello script 08.
MISSING_PROMPT_IDS = {
    "sensy2_5734",
}

ITERATION = 1
INPUT_PROMPT_SOURCE = "original_seed_prompt"
ITERATION_REASON = "initial_refactoring_recovery"

RECOVERY_MAX_TOKENS = 2400
RECOVERY_TEMPERATURE = 0.1


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


def load_existing_attempt_prompt_ids(output_path: str) -> set[str]:
    path = Path(output_path)

    if not path.exists():
        return set()

    existing_attempts = read_jsonl(path, RefactoringAttempt)

    return {
        attempt.prompt_id
        for attempt in existing_attempts
        if attempt.iteration == ITERATION
    }


def ensure_parent_dirs() -> None:
    Path(OUTPUT_REFACTORING_ATTEMPTS_JSONL).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    ensure_parent_dirs()

    seed_prompts = read_jsonl(INPUT_SEED_JSONL, PromptSeedDecision)
    baseline_responses = read_jsonl(INPUT_BASELINE_RESPONSES_JSONL, TargetResponse)
    baseline_evaluations = read_jsonl(
        INPUT_BASELINE_EVALUATIONS_JSONL,
        ResponseEvaluation,
    )

    prompt_by_id = {
        prompt.prompt_id: prompt
        for prompt in seed_prompts
    }

    responses_by_prompt_id = group_by_prompt_id(baseline_responses)
    evaluations_by_prompt_id = group_by_prompt_id(baseline_evaluations)

    existing_prompt_ids = load_existing_attempt_prompt_ids(
        OUTPUT_REFACTORING_ATTEMPTS_JSONL
    )

    knowledge_base = load_refactoring_knowledge_base()

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=REFACTORING_MODEL_NAME,
        timeout=240,
    )

    refactorer = PromptRefactorer(
        client=client,
        model_name=REFACTORING_MODEL_NAME,
        knowledge_base=knowledge_base,
        temperature=RECOVERY_TEMPERATURE,
        max_tokens=RECOVERY_MAX_TOKENS,
    )

    print("\n" + "=" * 90)
    print("MISSING REFACTORING ATTEMPT RECOVERY - V1")
    print("=" * 90)

    print("\nInput / output files:")
    print(f"  Seed prompts:              {INPUT_SEED_JSONL}")
    print(f"  Baseline responses:        {INPUT_BASELINE_RESPONSES_JSONL}")
    print(f"  Baseline evaluations:      {INPUT_BASELINE_EVALUATIONS_JSONL}")
    print(f"  Output attempts:           {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    print("\nRecovery configuration:")
    print(f"  Base URL:                  {BASE_URL}")
    print(f"  Refactoring model:         {REFACTORING_MODEL_NAME}")
    print(f"  Temperature:               {RECOVERY_TEMPERATURE}")
    print(f"  Max tokens:                {RECOVERY_MAX_TOKENS}")
    print(f"  Iteration:                 {ITERATION}")
    print(f"  Input prompt source:       {INPUT_PROMPT_SOURCE}")
    print(f"  Iteration reason:          {ITERATION_REASON}")
    print(f"  Missing prompt IDs:        {sorted(MISSING_PROMPT_IDS)}")

    print("\nExisting output:")
    print(f"  Existing attempts:         {len(existing_prompt_ids)}")

    print("\n" + "-" * 90)

    recovered = 0
    skipped = 0
    failed = 0

    for prompt_id in sorted(MISSING_PROMPT_IDS):
        if prompt_id in existing_prompt_ids:
            print(f"[SKIP] Already present: prompt_id={prompt_id}")
            skipped += 1
            continue

        seed_prompt = prompt_by_id.get(prompt_id)

        if seed_prompt is None:
            print(f"[ERROR] Prompt not found in seed: prompt_id={prompt_id}")
            failed += 1
            continue

        prompt_responses = responses_by_prompt_id.get(prompt_id, [])
        prompt_evaluations = evaluations_by_prompt_id.get(prompt_id, [])

        print()
        print(f"[RECOVER] prompt_id={prompt_id}")
        print(f"          category={get_prompt_category(seed_prompt)}")
        print(f"          baseline responses={len(prompt_responses)}")
        print(f"          baseline evaluations={len(prompt_evaluations)}")

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

            print(f"[OK] Recovered: prompt_id={prompt_id}")
            recovered += 1

        except Exception as exc:
            print(f"[ERROR] Failed to recover: prompt_id={prompt_id}")
            print(f"        {exc}")
            failed += 1

    print("\n" + "=" * 90)
    print("REFACTORING RECOVERY COMPLETED")
    print("=" * 90)

    print(f"\nRecovered:                  {recovered}")
    print(f"Skipped:                    {skipped}")
    print(f"Failed:                     {failed}")
    print(f"Output file:                {OUTPUT_REFACTORING_ATTEMPTS_JSONL}")

    print("\nNext checks:")
    print(
        "  (Get-Content "
        "data/intermediate/refactoring/attempts/refactoring_attempts_v1.jsonl).Count"
    )


if __name__ == "__main__":
    main()