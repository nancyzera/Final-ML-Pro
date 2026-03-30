from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from services.preprocess_service import build_preprocessor, split_xy
from services.model_registry import get_estimator


@dataclass
class DiagnosticsConfig:
    max_rows: int = 5000
    cv: int = 3
    learning_points: int = 6
    permute_rows: int = 300
    permute_repeats: int = 5


def build_model_diagnostics(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    test_size: float = 0.2,
    scale_numeric: bool = True,
    config: Optional[DiagnosticsConfig] = None,
) -> Dict[str, Any]:
    cfg = config or DiagnosticsConfig()

    # Keep diagnostics lightweight
    if len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=42)

    X, y = split_xy(df, target_column)
    feature_columns = list(X.columns)

    preprocessor, _schema = build_preprocessor(df, feature_columns, scale_numeric=scale_numeric)
    model = get_estimator(model_name, params={}, task="regression")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)

    # Learning curve
    lc_r2 = _learning_curve(pipe, X_train, y_train, cv=cfg.cv, points=cfg.learning_points, scoring="r2", label="R²")
    lc_rmse = _learning_curve(
        pipe,
        X_train,
        y_train,
        cv=cfg.cv,
        points=cfg.learning_points,
        scoring="neg_root_mean_squared_error",
        label="RMSE",
        negate=True,
    )
    lc_mae = _learning_curve(
        pipe,
        X_train,
        y_train,
        cv=cfg.cv,
        points=cfg.learning_points,
        scoring="neg_mean_absolute_error",
        label="MAE",
        negate=True,
    )

    cv_scores = _cv_scores(pipe, X_train, y_train, cv=cfg.cv)

    # Fit for residual diagnostics and importance
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Residuals vs predicted
    points = []
    for yt, yp in zip(y_test[:600], y_pred[:600]):
        points.append({"x": float(yp), "y": float(yt - yp)})

    importance = _feature_importance(pipe, X_test, y_test, feature_columns, cfg=cfg)
    roc_payload = _high_demand_roc(y_train, y_test, y_pred)
    hd_metrics = _high_demand_classification(y_train, y_test, y_pred)

    return {
        "learning_curve_r2": lc_r2,
        "learning_curve_rmse": lc_rmse,
        "learning_curve_mae": lc_mae,
        "cross_validation": cv_scores,
        "residuals_vs_pred": {"points": points, "x_label": "Predicted", "y_label": "Residual (actual - pred)"},
        "feature_importance": importance,
        "high_demand_roc": roc_payload,
        "high_demand_metrics": hd_metrics,
    }


def _learning_curve(pipe: Pipeline, X, y, cv: int, points: int, scoring: str, label: str, negate: bool = False) -> Dict[str, Any]:
    n = len(y)
    if n < 50:
        return {"train_sizes": [], "train_scores": [], "val_scores": [], "label": label, "note": "Dataset too small for learning curve."}

    train_sizes = np.linspace(0.15, 1.0, num=points)
    sizes, train_scores, val_scores = learning_curve(
        pipe,
        X,
        y,
        cv=int(cv),
        scoring=scoring,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42,
        n_jobs=None,
    )
    if negate:
        train_scores = -train_scores
        val_scores = -val_scores
    return {
        "train_sizes": [int(s) for s in sizes.tolist()],
        "train_scores": [float(np.nanmean(s)) for s in train_scores],
        "val_scores": [float(np.nanmean(s)) for s in val_scores],
        "label": label,
    }


def _cv_scores(pipe: Pipeline, X, y, cv: int) -> Dict[str, Any]:
    try:
        splitter = KFold(n_splits=int(max(2, min(10, cv))), shuffle=True, random_state=42)
        r2 = cross_val_score(pipe, X, y, cv=splitter, scoring="r2")
        out = {
            "folds": int(splitter.get_n_splits()),
            "r2_scores": [float(x) for x in np.asarray(r2, dtype=float).tolist()],
            "r2_mean": float(np.nanmean(r2)),
            "r2_std": float(np.nanstd(r2)),
        }
        try:
            rmse = -cross_val_score(pipe, X, y, cv=splitter, scoring="neg_root_mean_squared_error")
            out.update(
                {
                    "rmse_scores": [float(x) for x in np.asarray(rmse, dtype=float).tolist()],
                    "rmse_mean": float(np.nanmean(rmse)),
                    "rmse_std": float(np.nanstd(rmse)),
                }
            )
        except Exception:
            pass
        return out
    except Exception:
        return {"folds": int(cv), "r2_scores": [], "rmse_scores": []}


def _high_demand_roc(y_train, y_test, y_pred) -> Dict[str, Any]:
    """
    Regression-friendly AUC: treat "high demand" as y > median(train) and use predicted demand as a score.
    This provides a useful ranking diagnostic and satisfies AUC/ROC needs without switching to classification training.
    """
    try:
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_train.size < 20 or y_test.size < 20:
            return {"available": False}
        thr = float(np.nanmedian(y_train))
        y_bin = (y_test > thr).astype(int)
        if int(np.unique(y_bin).size) < 2:
            return {"available": False}
        fpr, tpr, _ = roc_curve(y_bin, y_pred)
        score = float(auc(fpr, tpr))
        pts = [{"x": float(a), "y": float(b)} for a, b in zip(fpr.tolist(), tpr.tolist())]
        if len(pts) > 600:
            idx = np.linspace(0, len(pts) - 1, num=600, dtype=int)
            pts = [pts[int(i)] for i in idx]
        return {"available": True, "auc": score, "threshold": thr, "points": pts}
    except Exception:
        return {"available": False}


def _high_demand_classification(y_train, y_test, y_pred) -> Dict[str, Any]:
    """
    Classification-style metrics derived from the regression output:
    label "High demand" as y > median(train) and classify using y_pred > same threshold.
    """
    try:
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_train.size < 20 or y_test.size < 20:
            return {"available": False}
        thr = float(np.nanmedian(y_train))
        y_true = (y_test > thr).astype(int)
        if int(np.unique(y_true).size) < 2:
            return {"available": False}
        y_hat = (y_pred > thr).astype(int)

        acc = float(accuracy_score(y_true, y_hat))
        prec = float(precision_score(y_true, y_hat, zero_division=0))
        rec = float(recall_score(y_true, y_hat, zero_division=0))
        f1 = float(f1_score(y_true, y_hat, zero_division=0))
        cm = confusion_matrix(y_true, y_hat).tolist()

        # For binary, micro/weighted == standard values; expose requested keys.
        return {
            "available": True,
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "precision_micro": prec,
            "precision_weighted": prec,
            "recall_micro": rec,
            "recall_weighted": rec,
            "f1_micro": f1,
            "f1_weighted": f1,
            "confusion_matrix": cm,
        }
    except Exception:
        return {"available": False}


def _feature_importance(pipe: Pipeline, X_test, y_test, raw_feature_columns: List[str], cfg: DiagnosticsConfig) -> Dict[str, Any]:
    model = pipe.named_steps.get("model")
    pre = pipe.named_steps.get("preprocess")

    # Get feature names
    feature_names = None
    if hasattr(pre, "get_feature_names_out"):
        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            feature_names = None
    if feature_names is None:
        feature_names = list(raw_feature_columns)

    # Direct importances
    importances = None
    if hasattr(model, "feature_importances_"):
        try:
            importances = np.array(model.feature_importances_, dtype=float)
        except Exception:
            importances = None
    elif hasattr(model, "coef_"):
        try:
            importances = np.abs(np.ravel(np.array(model.coef_, dtype=float)))
        except Exception:
            importances = None

    if importances is None or len(importances) != len(feature_names):
        # Permutation importance fallback (works for any regressor)
        try:
            from sklearn.inspection import permutation_importance

            nrows = min(int(cfg.permute_rows), int(X_test.shape[0]))
            if nrows < 30:
                return {"labels": [], "values": [], "method": "none"}
            X_sub = X_test.iloc[:nrows].copy()
            y_sub = y_test[:nrows]
            result = permutation_importance(
                pipe,
                X_sub,
                y_sub,
                scoring="r2",
                n_repeats=int(cfg.permute_repeats),
                random_state=42,
            )
            importances = np.array(result.importances_mean, dtype=float)
            method = "permutation_importance"
        except Exception:
            return {"labels": [], "values": [], "method": "none"}
    else:
        method = "model_importance"

    importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(np.abs(importances))) or 1.0
    scores = np.abs(importances) / total
    idx = np.argsort(scores)[::-1][:15]

    labels = [str(feature_names[i]) for i in idx]
    values = [float(scores[i]) for i in idx]
    return {"labels": labels, "values": values, "method": method}
