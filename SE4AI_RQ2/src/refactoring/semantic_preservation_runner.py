from pathlib import Path

from tqdm import tqdm

from src.llm.client import OpenAICompatibleClient
from src.refactoring.refactoring_models import (
    RefactoringAttempt,
    SemanticPreservationEvaluation,
)
from src.refactoring.semantic_preservation import SemanticPreservationChecker
from src.utils.jsonl import append_jsonl, read_jsonl


class SemanticPreservationRunConfig:
    """
    Configuration object for a semantic preservation run.

    The same runner can be reused for:
    - first-round refactorings;
    - semantic retry refactorings;
    - future v2/v3 refactoring attempts.
    """

    def __init__(
        self,
        input_refactoring_attempts_jsonl: str,
        output_all_jsonl: str,
        output_accepted_jsonl: str,
        output_rejected_jsonl: str,
        evaluator_model_name: str = "qwen/qwen3-14b",
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.0,
        max_tokens: int = 900,
        overwrite_output: bool = True,
        limit_attempts: int | None = None,
        run_label: str = "SEMANTIC PRESERVATION CHECK",
    ):
        self.input_refactoring_attempts_jsonl = input_refactoring_attempts_jsonl
        self.output_all_jsonl = output_all_jsonl
        self.output_accepted_jsonl = output_accepted_jsonl
        self.output_rejected_jsonl = output_rejected_jsonl

        self.evaluator_model_name = evaluator_model_name
        self.base_url = base_url
        self.api_key = api_key

        self.temperature = temperature
        self.max_tokens = max_tokens

        self.overwrite_output = overwrite_output
        self.limit_attempts = limit_attempts
        self.run_label = run_label


class SemanticPreservationRunner:
    """
    Reusable runner for semantic preservation evaluation.

    It reads a JSONL file containing RefactoringAttempt records and writes:
    - all semantic preservation evaluations;
    - accepted evaluations;
    - rejected evaluations.
    """

    def __init__(self, config: SemanticPreservationRunConfig):
        self.config = config

    def run(self) -> None:
        self._ensure_parent_dirs()
        self._remove_existing_outputs_if_needed()

        attempts = read_jsonl(
            self.config.input_refactoring_attempts_jsonl,
            RefactoringAttempt,
        )

        total_available_attempts = len(attempts)

        if self.config.limit_attempts is not None:
            selected_attempts = attempts[: self.config.limit_attempts]
        else:
            selected_attempts = attempts

        client = OpenAICompatibleClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model_name=self.config.evaluator_model_name,
            timeout=180,
        )

        checker = SemanticPreservationChecker(
            client=client,
            model_name=self.config.evaluator_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        self._print_header(
            total_available_attempts=total_available_attempts,
            selected_attempts=len(selected_attempts),
        )

        generated = 0
        failed = 0
        accepted = 0
        rejected = 0
        failed_items: list[dict] = []

        for attempt in tqdm(selected_attempts, desc="Checking semantic preservation"):
            try:
                evaluation = checker.evaluate(attempt)

                self._write_evaluation(evaluation)

                generated += 1

                if evaluation.decision == "accepted":
                    accepted += 1
                elif evaluation.decision == "rejected":
                    rejected += 1

            except Exception as exc:
                failed += 1

                failed_items.append(
                    {
                        "prompt_id": attempt.prompt_id,
                        "refactoring_attempt_id": attempt.refactoring_attempt_id,
                        "iteration": attempt.iteration,
                        "error": str(exc),
                    }
                )

                print()
                print("[ERROR] Failed to evaluate semantic preservation")
                print(f"  prompt_id:                {attempt.prompt_id}")
                print(f"  refactoring_attempt_id:   {attempt.refactoring_attempt_id}")
                print(f"  iteration:                {attempt.iteration}")
                print(f"  error:                    {exc}")

        self._print_footer(
            generated=generated,
            failed=failed,
            accepted=accepted,
            rejected=rejected,
            selected_count=len(selected_attempts),
            failed_items=failed_items,
        )

    def _ensure_parent_dirs(self) -> None:
        Path(self.config.output_all_jsonl).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _remove_existing_outputs_if_needed(self) -> None:
        if not self.config.overwrite_output:
            return

        output_paths = [
            Path(self.config.output_all_jsonl),
            Path(self.config.output_accepted_jsonl),
            Path(self.config.output_rejected_jsonl),
        ]

        for output_path in output_paths:
            if output_path.exists():
                output_path.unlink()

    def _write_evaluation(
        self,
        evaluation: SemanticPreservationEvaluation,
    ) -> None:
        append_jsonl(self.config.output_all_jsonl, evaluation)

        if evaluation.decision == "accepted":
            append_jsonl(self.config.output_accepted_jsonl, evaluation)

        elif evaluation.decision == "rejected":
            append_jsonl(self.config.output_rejected_jsonl, evaluation)

        else:
            raise ValueError(
                f"Unexpected semantic preservation decision: {evaluation.decision}"
            )

    def _print_header(
        self,
        total_available_attempts: int,
        selected_attempts: int,
    ) -> None:
        print("\n" + "=" * 90)
        print(self.config.run_label)
        print("=" * 90)

        print("\nInput / output files:")
        print(
            f"  Refactoring attempts:      "
            f"{self.config.input_refactoring_attempts_jsonl}"
        )
        print(f"  Output evaluations:        {self.config.output_all_jsonl}")
        print(f"  Output accepted:           {self.config.output_accepted_jsonl}")
        print(f"  Output rejected:           {self.config.output_rejected_jsonl}")

        print("\nEvaluator configuration:")
        print(f"  Base URL:                  {self.config.base_url}")
        print(f"  Evaluator model:           {self.config.evaluator_model_name}")
        print(f"  Temperature:               {self.config.temperature}")
        print(f"  Max tokens:                {self.config.max_tokens}")
        print(f"  Overwrite output:          {self.config.overwrite_output}")
        print(f"  Limit attempts:            {self.config.limit_attempts}")

        print("\nDataset:")
        print(f"  Total attempts:            {total_available_attempts}")
        print(f"  Attempts selected:         {selected_attempts}")

        print("\n" + "-" * 90)

    def _print_footer(
        self,
        generated: int,
        failed: int,
        accepted: int,
        rejected: int,
        selected_count: int,
        failed_items: list[dict],
    ) -> None:
        print("\n" + "=" * 90)
        print("SEMANTIC PRESERVATION CHECK COMPLETED")
        print("=" * 90)

        print(f"\nGenerated evaluations:       {generated}")
        print(f"Failed evaluations:          {failed}")
        print(f"Accepted:                    {accepted}")
        print(f"Rejected:                    {rejected}")
        print(f"Output all:                  {self.config.output_all_jsonl}")
        print(f"Output accepted:             {self.config.output_accepted_jsonl}")
        print(f"Output rejected:             {self.config.output_rejected_jsonl}")

        if failed_items:
            print("\nFailed items:")
            for item in failed_items[:50]:
                print(
                    f"  prompt_id={item['prompt_id']} | "
                    f"attempt_id={item['refactoring_attempt_id']} | "
                    f"iteration={item['iteration']} | "
                    f"error={item['error']}"
                )

        if generated == selected_count:
            print("\nSemantic preservation check completed successfully.")

            if rejected > 0:
                print(
                    "Some refactorings were rejected. "
                    "Next step: inspect the rejected split and run a controlled semantic retry."
                )
            else:
                print(
                    "All refactorings were accepted. "
                    "Next step: generate post-refactoring target-model responses."
                )
        else:
            print("\n[WARNING] Some evaluations failed. Inspect the error messages above.")