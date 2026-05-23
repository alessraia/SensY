from ..domain.models import SensitivePrompt, TargetResponse
from ..utils.ids import new_id
from .client import OpenAICompatibleClient


# Minimal system prompt used to preserve a realistic baseline setting.
# No safety-specific instructions or predefined safe-answer are injected here.
TARGET_SYSTEM_PROMPT = "You are a helpful assistant."


class TargetLLM:
    """
    Wrapper del modello target.

    Riceve un SensitivePrompt e restituisce una TargetResponse.

    In questa fase lo usiamo per generare le risposte baseline.
    Il prompt sensibile viene inviato al modello senza istruzioni aggiuntive
    di safety, safe-answer o refactoring, in modo da osservare il comportamento
    baseline del target LLM.
    """

    def __init__(self, client: OpenAICompatibleClient, model_name: str):
        self.client = client
        self.model_name = model_name

    def answer(
        self,
        prompt: SensitivePrompt,
        repetition: int = 0,
    ) -> TargetResponse:
        response_text = self.client.generate(
            system_prompt=TARGET_SYSTEM_PROMPT,
            user_prompt=prompt.text,
            temperature=0.2,
            max_tokens=256,
        )

        return TargetResponse(
            response_id=new_id("resp"),
            prompt_id=prompt.prompt_id,
            target_model=self.model_name,
            repetition=repetition,
            prompt_text=prompt.text,
            response_text=response_text,
            source="baseline",
        )