import requests


BASE_URL = "http://127.0.0.1:1234/v1"


def main() -> None:
    print("=" * 80)
    print("LM STUDIO SERVER CHECK")
    print("=" * 80)

    url = f"{BASE_URL}/models"
    print(f"\nChecking: {url}")

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    print("\nAvailable models:")

    models = data.get("data", [])

    if not models:
        print("No models found.")
        return

    for i, model in enumerate(models, start=1):
        print(f"{i}. {model.get('id')}")


if __name__ == "__main__":
    main()