import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

def _py_key(lbl):
    if isinstance(lbl, np.generic):
        if isinstance(lbl, np.integer):
            return int(lbl)
        if isinstance(lbl, np.floating):
            return float(lbl)
        return str(lbl)
    if isinstance(lbl, (int, float, str, bool)) or lbl is None:
        return lbl
    return str(lbl)

def _aggregate_per_class(arr_by_label):
    return {k: (float(v[0]), 0.0) if len(v) > 0 else (None, None)
            for k, v in arr_by_label.items()}

def compute_holdout_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro / Micro / Weighted
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    label_keys = [_py_key(lbl) for lbl in labels]
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_prec = {k: [p_c[i]] for i, k in enumerate(label_keys)}
    per_class_rec  = {k: [r_c[i]] for i, k in enumerate(label_keys)}
    per_class_f1   = {k: [f1_c[i]] for i, k in enumerate(label_keys)}

    rocs, pras = None, None
    if y_score is not None and len(labels) == 2:
        try:
            rocs = roc_auc_score(y_true, y_score)
        except Exception:
            rocs = None
        try:
            pras = average_precision_score(y_true, y_score)
        except Exception:
            pras = None

    as_pair = lambda v: (None, None) if v is None else (float(v), 0.0)

    return {
        "accuracy": (float(acc), 0.0),  # accuracy: (correct predictions) / (total predictions)

        "precision_macro": (float(p_macro), 0.0), # precision_macro: average of class precisions (equal weight for each class).
        "recall_macro":    (float(r_macro), 0.0), # recall_macro: average of class recalls (equal sensitivity).
        "f1_macro":        (float(f1_macro), 0.0),# f1_macro: harmonic mean of P/R for each class, then average across classes.

        "precision_micro": (float(p_micro), 0.0), # precision_micro: global precision across all classes.
        "recall_micro":    (float(r_micro), 0.0), # recall_micro: in binary often ≈ accuracy.
        "f1_micro":        (float(f1_micro), 0.0),

        "precision_weighted": (float(p_w), 0.0),  # precision_weighted: weighted average by support (class size).
        "recall_weighted":    (float(r_w), 0.0),
        "f1_weighted":        (float(f1_w), 0.0),

        "precision_per_class": _aggregate_per_class(per_class_prec), 
        "recall_per_class":    _aggregate_per_class(per_class_rec),  
        "f1_per_class":        _aggregate_per_class(per_class_f1),   

        "roc_auc": as_pair(rocs), 
        "pr_auc":  as_pair(pras), 
    }

def fit_and_evaluate_holdout(model_ctor, X_tr, y_tr, X_te, y_te):
    clf = model_ctor()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    y_score = None
    if hasattr(clf, "predict_proba"):
        labels = np.unique(y_te)
        if len(labels) == 2:
            pos_label = labels[1]
            col = list(clf.classes_).index(pos_label)
            y_score = clf.predict_proba(X_te)[:, col]
    elif hasattr(clf, "decision_function"):
        labels = np.unique(y_te)
        if len(labels) == 2:
            y_score = clf.decision_function(X_te)

    metrics = compute_holdout_metrics(y_te, y_pred, y_score=y_score)
    return clf, metrics
