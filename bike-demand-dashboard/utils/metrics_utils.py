import math
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred, n_features: int) -> Dict[str, float]:
    r2 = _finite_or_default(r2_score(y_true, y_pred), 0.0)
    mae = _finite_or_default(mean_absolute_error(y_true, y_pred), 0.0)
    mse = _finite_or_default(mean_squared_error(y_true, y_pred), 0.0)
    rmse = _finite_or_default(math.sqrt(mse), 0.0)
    mape = _finite_or_default(mean_absolute_percentage_error(y_true, y_pred), 0.0)
    med_ae = _finite_or_default(median_absolute_error(y_true, y_pred), 0.0)
    evs = _finite_or_default(explained_variance_score(y_true, y_pred), 0.0)

    n = len(y_true)
    adjusted_r2 = r2
    if n > n_features + 1:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        adjusted_r2 = _finite_or_default(adjusted_r2, r2)

    return {
        "r2_score": r2,
        "adjusted_r2": float(adjusted_r2),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "median_ae": med_ae,
        "explained_variance": evs,
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Returns common classification metrics. For multi-class, uses macro/micro/weighted averaging where relevant.
    If probabilities are provided, computes ROC-AUC when possible.
    """
    out: Dict[str, float] = {}
    out["accuracy"] = _finite_or_default(accuracy_score(y_true, y_pred), 0.0)

    for avg in ("micro", "macro", "weighted"):
        out[f"precision_{avg}"] = _finite_or_default(precision_score(y_true, y_pred, average=avg, zero_division=0), 0.0)
        out[f"recall_{avg}"] = _finite_or_default(recall_score(y_true, y_pred, average=avg, zero_division=0), 0.0)
        out[f"f1_{avg}"] = _finite_or_default(f1_score(y_true, y_pred, average=avg, zero_division=0), 0.0)

    # ROC-AUC (binary or multiclass with probabilities)
    if y_proba is not None:
        try:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                out["roc_auc"] = _finite_or_default(roc_auc_score(y_true, y_proba), 0.0)
            else:
                out["roc_auc_ovr_weighted"] = _finite_or_default(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 0.0
                )
                out["roc_auc_ovr_macro"] = _finite_or_default(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"), 0.0
                )
        except Exception:
            pass

    return out


def _finite_or_default(value, default: float) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)
