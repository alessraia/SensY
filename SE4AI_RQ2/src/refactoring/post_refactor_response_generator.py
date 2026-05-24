from src.llm.client import OpenAICompatibleClient
from src.refactoring.refactoring_models import RefactoringAttempt
from src.utils.ids import new_id


TARGET_SYSTEM_PROMPT = "You are a helpful assistant."


class PostRefactorResponseGenerator:
    """
    Generates target-model responses for semantically accepted refactored prompts.

    This component is used after:
    - refactoring attempt generation;
    - semantic preservation check;
    - optional semantic retry;
    - construction of accepted_refactoring_attempts_v1.jsonl.

    It does not evaluate responses.
    It only asks the target model to answer the refactored prompt.
    """

    def __init__(
        self,
        client: OpenAICompatibleClient,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response_record(
        self,
        attempt: RefactoringAttempt,
        repetition: int,
    ) -> dict:
        response_text = self.client.generate(
            system_prompt=TARGET_SYSTEM_PROMPT,
            user_prompt=attempt.refactored_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "response_id": new_id("postresp"),
            "prompt_id": attempt.prompt_id,
            "prompt_text": attempt.refactored_prompt,
            "response_text": response_text,
            "model_name": self.model_name,
            "repetition": repetition,
            "source": "post_refactor_response_v1",

            "episode_id": attempt.effective_episode_id,
            "refactoring_attempt_id": attempt.refactoring_attempt_id,
            "parent_refactoring_attempt_id": attempt.parent_refactoring_attempt_id,
            "iteration": attempt.iteration,
            "original_prompt": attempt.original_prompt,
            "refactored_prompt": attempt.refactored_prompt,
            "refactoring_method": attempt.refactoring_method,
            "input_prompt_source": attempt.input_prompt_source,
            "iteration_reason": attempt.iteration_reason,
            "category": attempt.category,
        }