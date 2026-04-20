# models/cross_validate.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

def cross_validate_10fold(model_ctor, X, y, random_state=42):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    labels = np.unique(y)

    accs, rocs, pras = [], [], []

    # macro
    precs_macro, recs_macro, f1s_macro = [], [], []

    # micro
    precs_micro, recs_micro, f1s_micro = [], [], []

    # weighted
    precs_weighted, recs_weighted, f1s_weighted = [], [], []

    # per-classe: usare chiavi JSON-safe
    label_keys = [_py_key(lbl) for lbl in labels]
    per_class_prec = {k: [] for k in label_keys}
    per_class_rec  = {k: [] for k in label_keys}
    per_class_f1   = {k: [] for k in label_keys}

    for tr, te in skf.split(X, y):
        clf = model_ctor()
        clf.fit(X[tr], y[tr])
        y_true = y[te]
        y_pred = clf.predict(X[te])

        if hasattr(clf, "predict_proba"):
            if len(labels) == 2:
                pos_label = labels[1]
                col = list(clf.classes_).index(pos_label)
                y_score = clf.predict_proba(X[te])[:, col]
            else:
                y_score = None
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X[te]) if len(labels) == 2 else None
        else:
            y_score = None

        # Accuracy
        accs.append(accuracy_score(y_true, y_pred))

        # Macro
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precs_macro.append(p_m); recs_macro.append(r_m); f1s_macro.append(f1_m)

        # Micro
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precs_micro.append(p_micro); recs_micro.append(r_micro); f1s_micro.append(f1_micro)

        # Weighted
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        precs_weighted.append(p_w); recs_weighted.append(r_w); f1s_weighted.append(f1_w)

        p_c, r_c, f1_c, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        for i, key in enumerate(label_keys):
            per_class_prec[key].append(p_c[i])
            per_class_rec[key].append(r_c[i])
            per_class_f1[key].append(f1_c[i])

        if (y_score is not None) and (len(labels) == 2):
            try:
                rocs.append(roc_auc_score(y_true, y_score))
            except Exception:
                pass
            pras.append(average_precision_score(y_true, y_score))

    agg = lambda a: (float(np.mean(a)), float(np.std(a))) if a else (None, None)

    def agg_per_class(d):
        return {k: (float(np.mean(vals)), float(np.std(vals))) if len(vals) > 0 else (None, None)
                for k, vals in d.items()}

    return {
        "accuracy": agg(accs),  
        "precision_macro": agg(precs_macro), 
        "recall_macro":    agg(recs_macro),  
        "f1_macro":        agg(f1s_macro),   

        "precision_micro": agg(precs_micro), 
        "recall_micro":    agg(recs_micro),  
        "f1_micro":        agg(f1s_micro),   

        "precision_weighted": agg(precs_weighted), 
        "recall_weighted":    agg(recs_weighted),  
        "f1_weighted":        agg(f1s_weighted),   

        "precision_per_class": agg_per_class(per_class_prec), 
        "recall_per_class":    agg_per_class(per_class_rec),  
        "f1_per_class":        agg_per_class(per_class_f1),   

        "roc_auc": agg(rocs), 
        "pr_auc":  agg(pras), 
    }
