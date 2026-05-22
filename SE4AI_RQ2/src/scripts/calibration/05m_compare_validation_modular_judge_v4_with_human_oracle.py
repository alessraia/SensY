import json
from pathlib import Path

import pandas as pd

from src.domain.models import OracleJudgeEvaluation
from src.utils.jsonl import read_jsonl


INPUT_JSONL = "data/calibration/oracle/modular_judge_validation_evaluations_v4.jsonl"

OUTPUT_COMPARISON_CSV = "data/calibration/results/modular_judge_v4_validation_comparison.csv"
OUTPUT_METRICS_JSON = "data/calibration/results/modular_judge_v4_validation_metrics.json"

POSITIVE_LABEL = "inadequate"


def format_section(title: str) -> str:
    line = "=" * 90
    return f"\n{line}\n{title}\n{line}"


def format_subsection(title: str) -> str:
    line = "-" * 90
    return f"\n{title}\n{line}"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_metrics(df: pd.DataFrame) -> dict:
    manual = df["manual_label"]
    judge = df["judge_label"]

    tp = int(((manual == POSITIVE_LABEL) & (judge == POSITIVE_LABEL)).sum())
    tn = int(((manual != POSITIVE_LABEL) & (judge != POSITIVE_LABEL)).sum())
    fp = int(((manual != POSITIVE_LABEL) & (judge == POSITIVE_LABEL)).sum())
    fn = int(((manual == POSITIVE_LABEL) & (judge != POSITIVE_LABEL)).sum())

    total = int(len(df))

    accuracy = safe_divide(tp + tn, total)

    precision_inadequate = safe_divide(tp, tp + fp)
    recall_inadequate = safe_divide(tp, tp + fn)
    f1_inadequate = safe_divide(
        2 * precision_inadequate * recall_inadequate,
        precision_inadequate + recall_inadequate,
    )

    precision_adequate = safe_divide(tn, tn + fn)
    recall_adequate = safe_divide(tn, tn + fp)
    f1_adequate = safe_divide(
        2 * precision_adequate * recall_adequate,
        precision_adequate + recall_adequate,
    )

    macro_f1 = (f1_adequate + f1_inadequate) / 2

    observed_agreement = accuracy

    manual_adequate = int((manual == "adequate").sum())
    manual_inadequate = int((manual == "inadequate").sum())
    judge_adequate = int((judge == "adequate").sum())
    judge_inadequate = int((judge == "inadequate").sum())

    expected_agreement = (
        (manual_adequate / total) * (judge_adequate / total)
        + (manual_inadequate / total) * (judge_inadequate / total)
        if total > 0
        else 0.0
    )

    kappa = safe_divide(
        observed_agreement - expected_agreement,
        1 - expected_agreement,
    )

    return {
        "total": total,
        "confusion_matrix": {
            "tp_inadequate": tp,
            "tn_adequate": tn,
            "fp_human_adequate_judge_inadequate": fp,
            "fn_human_inadequate_judge_adequate": fn,
        },
        "accuracy": accuracy,
        "precision_inadequate": precision_inadequate,
        "recall_inadequate": recall_inadequate,
        "f1_inadequate": f1_inadequate,
        "precision_adequate": precision_adequate,
        "recall_adequate": recall_adequate,
        "f1_adequate": f1_adequate,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
    }


def evaluations_to_dataframe(evaluations: list[OracleJudgeEvaluation]) -> pd.DataFrame:
    rows = []

    for item in evaluations:
        agreement = item.manual_label == item.judge_label

        if agreement:
            error_type = "none"
        elif item.manual_label.value == "adequate" and item.judge_label.value == "inadequate":
            error_type = "false_positive"
        elif item.manual_label.value == "inadequate" and item.judge_label.value == "adequate":
            error_type = "false_negative"
        else:
            error_type = "unknown"

        rows.append(
            {
                "oracle_id": item.oracle_id,
                "question": item.question,
                "category": item.category,
                "response_text": item.response_text,
                "source_model": item.source_model,
                "response_index": item.response_index,
                "oracle_source_file": item.oracle_source_file,
                "manual_label": item.manual_label.value,
                "judge_label": item.judge_label.value,
                "agreement": agreement,
                "error_type": error_type,
                "judge_model": item.judge_model,
                "adequacy_score": item.adequacy_score,
                "safety_score": item.safety_score,
                "helpfulness_score": item.helpfulness_score,
                "sensitivity_awareness_score": item.sensitivity_awareness_score,
                "over_refusal": item.over_refusal,
                "under_refusal": item.under_refusal,
                "failure_modes": "|".join(item.failure_modes),
                "rationale": item.rationale,
            }
        )

    return pd.DataFrame(rows)


def print_metrics(title: str, metrics: dict) -> None:
    print(format_subsection(title))

    cm = metrics["confusion_matrix"]

    print(f"Total:                                      {metrics['total']}")
    print(f"Accuracy:                                   {metrics['accuracy']:.4f}")
    print(f"Precision inadequate:                       {metrics['precision_inadequate']:.4f}")
    print(f"Recall inadequate:                          {metrics['recall_inadequate']:.4f}")
    print(f"F1 inadequate:                              {metrics['f1_inadequate']:.4f}")
    print(f"Precision adequate:                         {metrics['precision_adequate']:.4f}")
    print(f"Recall adequate:                            {metrics['recall_adequate']:.4f}")
    print(f"F1 adequate:                                {metrics['f1_adequate']:.4f}")
    print(f"Macro F1:                                   {metrics['macro_f1']:.4f}")
    print(f"Cohen's kappa:                              {metrics['cohen_kappa']:.4f}")

    print("\nConfusion matrix:")
    print(f"  TP inadequate:                            {cm['tp_inadequate']}")
    print(f"  TN adequate:                              {cm['tn_adequate']}")
    print(f"  FP human adequate / judge inadequate:     {cm['fp_human_adequate_judge_inadequate']}")
    print(f"  FN human inadequate / judge adequate:     {cm['fn_human_inadequate_judge_adequate']}")


def main() -> None:
    evaluations = read_jsonl(INPUT_JSONL, OracleJudgeEvaluation)

    print(format_section("MODULAR JUDGE VS HUMAN ORACLE COMPARISON"))

    print("\nInput file:")
    print(f"  {INPUT_JSONL}")

    if not evaluations:
        print("\nNo evaluations found.")
        return

    df = evaluations_to_dataframe(evaluations)

    Path(OUTPUT_COMPARISON_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_METRICS_JSON).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_COMPARISON_CSV, index=False, encoding="utf-8")

    global_metrics = compute_binary_metrics(df)

    metrics = {
        "global": global_metrics,
        "by_source_model": {},
        "by_category": {},
    }

    for source_model, group in df.groupby("source_model"):
        metrics["by_source_model"][source_model] = compute_binary_metrics(group)

    for category, group in df.groupby("category"):
        metrics["by_category"][category] = compute_binary_metrics(group)

    with Path(OUTPUT_METRICS_JSON).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(format_subsection("BASIC COUNTS"))
    print(f"Evaluations loaded:              {len(df)}")
    print(f"Unique oracle_ids:               {df['oracle_id'].nunique()}")
    print(f"Source models:                   {sorted(df['source_model'].unique())}")
    print(f"Manual labels:                   {df['manual_label'].value_counts().to_dict()}")
    print(f"Judge labels:                    {df['judge_label'].value_counts().to_dict()}")

    print_metrics("GLOBAL METRICS", global_metrics)

    print(format_subsection("METRICS BY SOURCE MODEL"))

    for source_model, source_metrics in metrics["by_source_model"].items():
        print_metrics(f"source_model = {source_model}", source_metrics)

    print(format_subsection("ERROR DISTRIBUTION"))
    print(df["error_type"].value_counts())

    print(format_subsection("ERROR DISTRIBUTION BY SOURCE MODEL"))
    print(
        df.groupby(["source_model", "error_type"])
        .size()
        .unstack(fill_value=0)
    )

    print(format_subsection("OUTPUT FILES"))
    print(f"Comparison CSV: {OUTPUT_COMPARISON_CSV}")
    print(f"Metrics JSON:   {OUTPUT_METRICS_JSON}")

    print(format_section("MODULAR COMPARISON COMPLETED"))


if __name__ == "__main__":
    main()