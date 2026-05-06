import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split

from llm.config import TRAIN_FILE, SQUARE_FILE, TEMP_DIR, RANDOM_SEED, TEST_SIZE
from llm.prompts import build_prompt


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sensy_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": str(record["question_en"]).strip(),
        "label": int(record["sensitive?"]),
        "category": record.get("category"),
        "source": "sensy",
    }


def normalize_square_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": str(record["question_en"]).strip(),
        "label": int(record["sensitive?"]),
        "category": record.get("category"),
        "source": "square",
    }


def to_sft_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(record["text"])
    completion = str(record["label"])

    return {
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "label": record["label"],
        "category": record.get("category"),
        "source": record["source"],
    }


def to_eval_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": build_prompt(record["text"]),
        "text": record["text"],
        "label": record["label"],
        "category": record.get("category"),
        "source": record["source"],
    }


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    sensy_raw = load_json(TRAIN_FILE)
    square_raw = load_json(SQUARE_FILE)

    sensy_data = [normalize_sensy_record(r) for r in sensy_raw]
    square_data = [normalize_square_record(r) for r in square_raw]

    train_data, dev_data = train_test_split(
        sensy_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=[r["label"] for r in sensy_data],
    )

    train_sft = [to_sft_record(r) for r in train_data]
    dev_sft = [to_sft_record(r) for r in dev_data]
    square_eval = [to_eval_record(r) for r in square_data]

    save_jsonl(train_sft, TEMP_DIR / "sensy_train_sft.jsonl")
    save_jsonl(dev_sft, TEMP_DIR / "sensy_dev_sft.jsonl")
    save_jsonl(square_eval, TEMP_DIR / "square_eval.jsonl")

    print(f"Train size: {len(train_sft)}")
    print(f"Dev size: {len(dev_sft)}")
    print(f"SQuARe eval size: {len(square_eval)}")
    print(f"Saved: {TEMP_DIR / 'sensy_train_sft.jsonl'}")
    print(f"Saved: {TEMP_DIR / 'sensy_dev_sft.jsonl'}")
    print(f"Saved: {TEMP_DIR / 'square_eval.jsonl'}")


if __name__ == "__main__":
    main()