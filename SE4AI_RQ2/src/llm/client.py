import requests


class OpenAICompatibleClient:
    """
    Client minimale per endpoint compatibili con OpenAI Chat Completions.

    È pensato per funzionare con LM Studio o altri server locali che espongono
    un endpoint compatibile con:

        /v1/chat/completions
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()