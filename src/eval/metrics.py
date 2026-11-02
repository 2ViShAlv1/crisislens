"""Shared metric code. Every number in the README comes from these functions."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

CLASSES = ["caution_and_advice", "displaced_people_and_evacuations", "infrastructure_and_utility_damage",
           "injured_or_dead_people", "missing_or_found_people", "not_humanitarian",
           "other_relevant_information", "requests_or_urgent_needs",
           "rescue_volunteering_or_donation_effort", "sympathy_and_support"]
LIFE_CRITICAL = ["requests_or_urgent_needs", "injured_or_dead_people"]
INVALID = "__invalid__"

def classification_report(y_true, y_pred, n_boot=1000, seed=0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    present = [c for c in CLASSES if (y_true == c).any()]  # macro over classes with support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    per = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=CLASSES)
    out = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "invalid_rate": float((y_pred == INVALID).mean()),
    }
    for c in LIFE_CRITICAL:
        out[f"recall_{c}"] = float(per.loc[c, "recall"])
    # bootstrap 95% CI for macro-F1
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        bs.append(f1_score(y_true[i], y_pred[i], labels=present, average="macro", zero_division=0))
    out["macro_f1_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES + [INVALID]),
                      index=CLASSES + [INVALID], columns=CLASSES + [INVALID]).iloc[:-1]
    return out, per, cm
