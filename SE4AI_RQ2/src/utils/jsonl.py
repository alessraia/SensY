from pathlib import Path
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def model_to_json(record: BaseModel) -> str:
    """
    Serializza un modello Pydantic in JSON.

    Compatibile sia con Pydantic v2 sia con Pydantic v1.
    """

    if hasattr(record, "model_dump_json"):
        return record.model_dump_json()

    return record.json()


def model_from_json(model_cls: Type[T], line: str) -> T:
    """
    Deserializza una riga JSON in un modello Pydantic.

    Compatibile sia con Pydantic v2 sia con Pydantic v1.
    """

    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(line)

    return model_cls.parse_raw(line)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    """
    Scrive più record in un file JSONL, sovrascrivendo il file esistente.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(model_to_json(record) + "\n")


def append_jsonl(path: str | Path, record: BaseModel) -> None:
    """
    Aggiunge un singolo record a un file JSONL.

    Utile per salvare progressivamente le risposte dei modelli.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(model_to_json(record) + "\n")


def read_jsonl(path: str | Path, model_cls: Type[T]) -> list[T]:
    """
    Legge un file JSONL e restituisce una lista di modelli Pydantic.
    """

    path = Path(path)

    records: list[T] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(model_from_json(model_cls, line))

    return records