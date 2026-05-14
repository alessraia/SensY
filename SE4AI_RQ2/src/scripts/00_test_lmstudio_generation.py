from ..llm.client import OpenAICompatibleClient


BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "qwen2.5-7b-instruct"


def main() -> None:
    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        timeout=120,
    )

    print("=" * 80)
    print("LM STUDIO GENERATION TEST")
    print("=" * 80)

    print(f"\nBase URL: {BASE_URL}")
    print(f"Model:    {MODEL_NAME}")

    try:
        response = client.generate(
            system_prompt="You are a helpful assistant. Answer briefly.",
            user_prompt="Say hello in one sentence.",
            temperature=0.2,
            max_tokens=64,
        )

        print("\nModel response:")
        print(response)

    except Exception as exc:
        print("\n[ERROR] Generation failed.")
        print("Details:")
        print(exc)


if __name__ == "__main__":
    main()