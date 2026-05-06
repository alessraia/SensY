import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from llm.config import OUTPUTS_DIR


def load_predictions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    args = parser.parse_args()

    pred_file = OUTPUTS_DIR / f"{args.model}_square_predictions.json"
    metrics_file = OUTPUTS_DIR / f"{args.model}_square_metrics.json"

    data = load_predictions(pred_file)

    y_true = [int(x["gold_label"]) for x in data]
    y_pred = [int(x["pred_label"]) for x in data]

    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model": args.model,
        "n_samples": len(data),
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_sensitive": precision_binary,
        "recall_sensitive": recall_binary,
        "f1_sensitive": f1_binary,
        "confusion_matrix": cm,
    }

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()