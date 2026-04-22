from preprocessing.clean_data import clean_dataset
from preprocessing.feature_extraction_ML import extract_features
from models.cross_validate import cross_validate_10fold
from models.split import compute_holdout_metrics

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

import json
import os
import numpy as np
from datetime import datetime

# LazyPredict
try:
    from lazypredict.Supervised import LazyClassifier
    LAZYPREDICT_AVAILABLE = True
except ImportError:
    LAZYPREDICT_AVAILABLE = False


# ======================
# MODEL REGISTRY
# ======================

def get_model_registry(random_state=42):
    return {
        "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),

        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),

        "linear_svc": lambda: LinearSVC(
            random_state=random_state
        ),

        "svc": lambda: SVC(
            kernel="rbf",
            probability=True
        ),

        "ridge_classifier": lambda: RidgeClassifier(),

        "sgd_classifier": lambda: SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-3,
            random_state=random_state
        ),

        "multinomial_nb": lambda: MultinomialNB(),

        "complement_nb": lambda: ComplementNB(),

        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),

        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
    }


# ======================
# UTILS
# ======================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def ensure_output_dirs(cfg):
    ensure_dir(cfg["report_dir"])
    ensure_dir(cfg["model_dir"])
    ensure_dir(cfg["errors_dir"])
    ensure_dir(cfg["results_dir"])


def build_run_name(model_name, cfg):
    mode = cfg["evaluation_mode"]
    tag = cfg.get("experiment_tag", "default")
    return f"{model_name}__{mode}__{tag}"


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def print_metric(name, val, indent=0):
    pad = "  " * indent
    if isinstance(val, dict):
        print(f"{pad}{name}:")
        for lbl in sorted(val.keys(), key=lambda x: str(x)):
            print_metric(str(lbl), val[lbl], indent=indent + 1)
    else:
        try:
            m, s = val
            if m is None:
                print(f"{pad}{name:16s}: n/a")
            else:
                print(f"{pad}{name:16s}: mean={m:.4f}  std={s:.4f}")
        except Exception:
            print(f"{pad}{name:16s}: {val}")


def print_report(report):
    for k, v in report.items():
        print_metric(k, v)


def save_json_report(report, cfg, model_name, dataset_info):
    run_name = build_run_name(model_name, cfg)
    out_path = os.path.join(cfg["report_dir"], f"{run_name}.json")

    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "evaluation_mode": cfg["evaluation_mode"],
            "experiment_tag": cfg.get("experiment_tag", "default"),
            "train_path": cfg["train_path"],
            "test_path": cfg["test_path"] if cfg["evaluation_mode"] == "holdout" else None,
            "random_state": cfg["random_state"],
            "dataset_info": dataset_info
        },
        "metrics": make_json_serializable(report)
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n>> Report salvato in: {out_path}")
    return out_path


def get_decision_scores(model, X, y_reference=None):
    y_score = None

    if hasattr(model, "predict_proba"):
        if y_reference is not None:
            labels = np.unique(y_reference)
            if len(labels) == 2:
                pos_label = labels[1]
                col = list(model.classes_).index(pos_label)
                y_score = model.predict_proba(X)[:, col]

    elif hasattr(model, "decision_function"):
        if y_reference is not None:
            labels = np.unique(y_reference)
            if len(labels) == 2:
                y_score = model.decision_function(X)

    return y_score


# ======================
# DATA LOADING
# ======================

def load_and_vectorize_dataset(dataset_path):
    print(f"Loading dataset: {dataset_path}")
    df = clean_dataset(dataset_path)

    print(f"Feature extraction: {dataset_path}")
    X, y = extract_features(df)

    y = np.asarray(y, dtype=int)
    return df, X, y


# ======================
# EXPERIMENTS
# ======================

def run_cv10(model_name, model_ctor, X_train, y_train, cfg):
    print("\n=== 10-FOLD STRATIFIED CROSS-VALIDATION ===")
    print(f"Model: {model_name}")
    print(f"Rows: {len(y_train)}")

    metrics = cross_validate_10fold(
        model_ctor,
        X_train,
        y_train,
        random_state=cfg["random_state"]
    )
    return metrics


def run_holdout(model_name, model_ctor, X_train, y_train, X_test, y_test, cfg):
    print("\n=== HOLD-OUT / CROSS-DATASET EVALUATION ===")
    print(f"Model: {model_name}")
    print(f"Train rows: {len(y_train)}")
    print(f"Test rows : {len(y_test)}")

    model = model_ctor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = get_decision_scores(model, X_test, y_reference=y_test)

    report = compute_holdout_metrics(y_test, y_pred, y_score=y_score)
    return report


def run_lazy_screening(X, y, cfg):
    if not LAZYPREDICT_AVAILABLE:
        raise ImportError("LazyPredict non è installato. Esegui: pip install lazypredict")

    print("\n=== LAZYPREDICT SCREENING ===")
    print(f"Rows: {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["lazy_test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False
    )

    models_df, _ = clf.fit(X_train, X_test, y_train, y_test)

    print("\nTop modelli LazyPredict:")
    print(models_df.head(15))

    run_name = build_run_name("lazypredict", cfg)

    csv_path = os.path.join(cfg["results_dir"], f"{run_name}.csv")
    models_df.to_csv(csv_path, encoding="utf-8")

    json_path = os.path.join(cfg["report_dir"], f"{run_name}.json")
    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "evaluation_mode": "lazy",
            "experiment_tag": cfg.get("experiment_tag", "default"),
            "train_path": cfg["train_path"],
            "random_state": cfg["random_state"],
            "lazy_test_size": cfg["lazy_test_size"],
            "rows": len(y)
        },
        "leaderboard": make_json_serializable(models_df.reset_index().to_dict(orient="records"))
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n>> Leaderboard CSV salvata in: {csv_path}")
    print(f">> Leaderboard JSON salvata in: {json_path}")

    return models_df


def run_selected_models(cfg):
    ensure_output_dirs(cfg)
    registry = get_model_registry(random_state=cfg["random_state"])

    selected_models = cfg["selected_models"]
    models_to_run = []

    for model_name in selected_models:
        if model_name not in registry:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {list(registry.keys())}"
            )
        models_to_run.append((model_name, registry[model_name]))

    if cfg["evaluation_mode"] == "cv10":
        df_train, X_train, y_train = load_and_vectorize_dataset(cfg["train_path"])
        dataset_info = {"train_rows": len(df_train)}

        for model_name, model_ctor in models_to_run:
            try:
                print("\n" + "=" * 70)
                print(f"Running experiment for model: {model_name}")
                print("=" * 70)

                report = run_cv10(model_name, model_ctor, X_train, y_train, cfg)
                print_report(report)
                save_json_report(report, cfg, model_name, dataset_info)

            except Exception as e:
                error_path = os.path.join(
                    cfg["errors_dir"],
                    f"{build_run_name(model_name, cfg)}__error.txt"
                )
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
                print(f"\n!! Errore nel modello {model_name}. Dettagli salvati in: {error_path}")
                print(f"Errore: {e}")

    elif cfg["evaluation_mode"] == "holdout":
        df_train, X_train, y_train = load_and_vectorize_dataset(cfg["train_path"])
        df_test, X_test, y_test = load_and_vectorize_dataset(cfg["test_path"])
        dataset_info = {
            "train_rows": len(df_train),
            "test_rows": len(df_test)
        }

        for model_name, model_ctor in models_to_run:
            try:
                print("\n" + "=" * 70)
                print(f"Running experiment for model: {model_name}")
                print("=" * 70)

                report = run_holdout(model_name, model_ctor, X_train, y_train, X_test, y_test, cfg)
                print_report(report)
                save_json_report(report, cfg, model_name, dataset_info)

            except Exception as e:
                error_path = os.path.join(
                    cfg["errors_dir"],
                    f"{build_run_name(model_name, cfg)}__error.txt"
                )
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
                print(f"\n!! Errore nel modello {model_name}. Dettagli salvati in: {error_path}")
                print(f"Errore: {e}")

    else:
        raise ValueError(f"Unsupported evaluation_mode in run_selected_models: {cfg['evaluation_mode']}")


def run_lazy(cfg):
    ensure_output_dirs(cfg)
    _, X, y = load_and_vectorize_dataset(cfg["train_path"])
    run_lazy_screening(X, y, cfg)