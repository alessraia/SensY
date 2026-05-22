import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def remove_think_blocks(text: str) -> str:
    """
    Rimuove eventuali blocchi <think>...</think> prodotti da modelli reasoning-style.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_object(text: str) -> dict:
    """
    Estrae un oggetto JSON dall'output del modello.

    Il judge dovrebbe restituire solo JSON, ma alcuni modelli possono aggiungere
    testo prima o dopo. Questa funzione prova a recuperare il primo oggetto JSON.
    """

    text = remove_think_blocks(text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "A JSON-like object was found, but it could not be parsed.\n"
            f"Extracted text:\n{json_text}\n\n"
            f"Original error: {exc}"
        )


def parse_model_output(text: str, model_cls: Type[T]) -> T:
    """
    Converte l'output JSON del modello in un oggetto Pydantic.
    Compatibile con Pydantic v1 e v2.
    """

    obj = extract_json_object(text)

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(obj)

    return model_cls.parse_obj(obj)