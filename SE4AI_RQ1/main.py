from models.evaluate_model import evaluate_model
from models.train_model import train_model
from preprocessing.clean_data import clean_dataset
from preprocessing.feature_extraction import extract_features
from models.cross_validate import cross_validate_10fold
from models.split import compute_holdout_metrics  # oppure fit_and_evaluate_holdout

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd
import pickle
import json
import numpy as np
import os

# ======================
# CONFIG
# ======================
RUN_CV10 = False

# Dataset paths
TRAIN_PATH      = "data/dataset_SENSY2.0.json" #"data/question_train.json"
TEST_PATH       = "data/dataset_SQUARE.json" #"data/question_train.json"  # used only if RUN_CV10=False

# Output
REPORT_DIR  = "samples/report"
MODEL_DIR   = "samples/models"
ERRORS_DIR  = "samples/errors"
RESULTS_DIR = "samples/results"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def rf_ctor():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

if __name__ == "__main__":
    ensure_dir(REPORT_DIR); ensure_dir(MODEL_DIR); ensure_dir(ERRORS_DIR); ensure_dir(RESULTS_DIR)

    print("Loading (without cleaning, without balancing)...")
    df_train = clean_dataset(TRAIN_PATH)

    if RUN_CV10:
        # ====== CV 10-FOLD ======
        print("\n===  10-FOLD STRATIFIED CROSS-VALIDATION ===")
        print(f"Train+Additional: {len(df_train)} rows")

        X_tr, y_tr = extract_features(df_train)
        X_tr = np.asarray(X_tr); y_tr = np.asarray(y_tr, dtype=int)

        metrics = cross_validate_10fold(rf_ctor, X_tr, y_tr, random_state=42)

        out_json = os.path.join(REPORT_DIR, "cv10_report_SensY2.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n>> Report 10-fold saved in: {out_json}\n")

        # --- Robust printer: handles tuples (mean, std) and nested dicts (per-class) ---
        def _print_metric(name, val, indent=0):
            pad = "  " * indent
            if isinstance(val, dict):
                print(f"{pad}{name}:")
                for lbl in sorted(val.keys(), key=lambda x: str(x)):
                    _print_metric(str(lbl), val[lbl], indent=indent+1)
            else:
                try:
                    m, s = val
                    if m is None:
                        print(f"{pad}{name:16s}: n/a")
                    else:
                        print(f"{pad}{name:16s}: mean={m:.4f}  std={s:.4f}")
                except Exception:
                    print(f"{pad}{name:16s}: {val}")

        for k, v in metrics.items():
            _print_metric(k, v)

    else:

        print("\n=== Hold-Out (TRAIN -> TEST) ===")
        df_test = clean_dataset(TEST_PATH)

        print("Feature extraction (TRAIN)...")
        X_tr, y_tr = extract_features(df_train)
        X_tr = np.asarray(X_tr)
        y_tr = np.asarray(y_tr, dtype=int)

        print("Feature extraction (TEST)...")
        X_te, y_te = extract_features(df_test)
        X_te = np.asarray(X_te)
        y_te = np.asarray(y_te, dtype=int)

        print("Training model...")
        # If you prefer to reuse your pipeline:
        model, _, _ = train_model(X_tr, y_tr, split=False)

        print("Evaluation on TEST...")
        # Predictions and, if available, probabilities/decision function for AUC/PR
        y_pred = model.predict(X_te)

        y_score = None
        if hasattr(model, "predict_proba"):
            labels = np.unique(y_te)
            if len(labels) == 2:
                pos_label = labels[1]
                col = list(model.classes_).index(pos_label)
                y_score = model.predict_proba(X_te)[:, col]
        elif hasattr(model, "decision_function"):
            labels = np.unique(y_te)
            if len(labels) == 2:
                y_score = model.decision_function(X_te)

        # Compute metrics in a format compatible with cross_validate
        report = compute_holdout_metrics(y_te, y_pred, y_score=y_score)


        # Print compatible with CV printer (mean/std)
        def _print_metric(name, val, indent=0):
            pad = "  " * indent
            if isinstance(val, dict):
                print(f"{pad}{name}:")
                for lbl in sorted(val.keys(), key=lambda x: str(x)):
                    _print_metric(str(lbl), val[lbl], indent=indent + 1)
            else:
                try:
                    m, s = val
                    if m is None:
                        print(f"{pad}{name:16s}: n/a")
                    else:
                        print(f"{pad}{name:16s}: mean={m:.4f}  std={s:.4f}")
                except Exception:
                    print(f"{pad}{name:16s}: {val}")


        for k, v in report.items():
            _print_metric(k, v)

        # Save report
        report_path = os.path.join(REPORT_DIR, "report_combinati_train-SensY2.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"\n>> Report hold-out saved in: {report_path}")


