from ..refactoring.semantic_preservation_runner import (
    SemanticPreservationRunConfig,
    SemanticPreservationRunner,
)


def main() -> None:
    config = SemanticPreservationRunConfig(
        input_refactoring_attempts_jsonl=(
            "data/intermediate/refactoring/attempts/"
            "refactoring_attempts_v1_semantic_retry.jsonl"
        ),
        output_all_jsonl=(
            "data/intermediate/refactoring/semantic_preservation/"
            "semantic_preservation_v1_semantic_retry.jsonl"
        ),
        output_accepted_jsonl=(
            "data/intermediate/refactoring/semantic_preservation/"
            "semantic_preservation_v1_semantic_retry_accepted.jsonl"
        ),
        output_rejected_jsonl=(
            "data/intermediate/refactoring/semantic_preservation/"
            "semantic_preservation_v1_semantic_retry_rejected.jsonl"
        ),
        evaluator_model_name="qwen/qwen3-14b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.0,
        max_tokens=900,
        overwrite_output=True,
        limit_attempts=None,
        run_label="SEMANTIC PRESERVATION CHECK - V1 SEMANTIC RETRY",
    )

    runner = SemanticPreservationRunner(config)
    runner.run()


if __name__ == "__main__":
    main()