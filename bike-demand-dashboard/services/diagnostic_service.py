from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer

from services.chart_service import _detect_demand_breakdown_pairs

from services.preprocess_service import build_preprocessor, split_xy
from services.model_registry import get_estimator


@dataclass
class DiagnosticsConfig:
    max_rows: int = 5000
    cv: int = 10
    learning_points: int = 6
    permute_rows: int = 300
    permute_repeats: int = 5


def build_model_diagnostics(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    test_size: float = 0.2,
    scale_numeric: bool = True,
    task: str = "regression",
    task_details: Optional[Dict[str, Any]] = None,
    config: Optional[DiagnosticsConfig] = None,
) -> Dict[str, Any]:
    cfg = config or DiagnosticsConfig()

    # Keep diagnostics lightweight
    if len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=42)

    X, y = split_xy(df, target_column)
    feature_columns = list(X.columns)
    task = "classification" if str(task).strip().lower() == "classification" else "regression"
    task_details = task_details or {}
    if task == "classification":
        threshold = float(task_details.get("threshold", np.nanmedian(np.asarray(y, dtype=float))))
        y = (np.asarray(y, dtype=float) > threshold).astype(int)

    preprocessor, _schema = build_preprocessor(df, feature_columns, scale_numeric=scale_numeric)
    model = get_estimator(model_name, params={}, task=task)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    stratify = y if task == "classification" and np.unique(y).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42, stratify=stratify
    )

    # Learning curve
    if task == "classification":
        lc_r2 = _learning_curve(pipe, X_train, y_train, cv=cfg.cv, points=cfg.learning_points, scoring="accuracy", label="Accuracy")
        lc_rmse = _learning_curve(pipe, X_train, y_train, cv=cfg.cv, points=cfg.learning_points, scoring="f1_weighted", label="F1 weighted")
        lc_mae = _learning_curve(
            pipe,
            X_train,
            y_train,
            cv=cfg.cv,
            points=cfg.learning_points,
            scoring=make_scorer(precision_score, average="weighted", zero_division=0),
            label="Precision weighted",
        )
    else:
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

    cv_scores = _cv_scores(pipe, X_train, y_train, cv=cfg.cv, task=task)

    # Fit for residual diagnostics and importance
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = _prediction_scores(pipe, X_test)

    # Residuals vs predicted
    points = []
    for yt, yp in zip(y_test[:600], y_pred[:600]):
        points.append({"x": float(yp), "y": float(yt - yp)})

    importance = _feature_importance(pipe, X_test, y_test, feature_columns, cfg=cfg, task=task)
    if task == "classification":
        roc_payload = _classification_roc(y_test, y_score)
        hd_metrics = _classification_metrics_payload(y_test, y_pred, y_score)
    else:
        roc_payload = _high_demand_roc(y_train, y_test, y_pred)
        hd_metrics = _high_demand_classification(y_train, y_test, y_pred)

    return {
        "task": task,
        "learning_curve_r2": lc_r2,
        "learning_curve_rmse": lc_rmse,
        "learning_curve_mae": lc_mae,
        "cross_validation": cv_scores,
        "residuals_vs_pred": {"points": points, "x_label": "Predicted", "y_label": "Residual (actual - pred)"},
        "feature_importance": importance,
        "high_demand_roc": roc_payload,
        "high_demand_metrics": hd_metrics,
        "shared_nonshared": _shared_nonshared_payload(df),
    }


def _learning_curve(pipe: Pipeline, X, y, cv: int, points: int, scoring, label: str, negate: bool = False) -> Dict[str, Any]:
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


def _cv_scores(pipe: Pipeline, X, y, cv: int, task: str = "regression") -> Dict[str, Any]:
    try:
        splitter = KFold(n_splits=int(max(2, min(10, cv))), shuffle=True, random_state=42)
        if task == "classification":
            acc = cross_val_score(pipe, X, y, cv=splitter, scoring="accuracy")
            f1_weighted = cross_val_score(pipe, X, y, cv=splitter, scoring="f1_weighted")
            precision_weighted = cross_val_score(
                pipe, X, y, cv=splitter, scoring=make_scorer(precision_score, average="weighted", zero_division=0)
            )
            out = {
                "folds": int(splitter.get_n_splits()),
                "accuracy_scores": [float(x) for x in np.asarray(acc, dtype=float).tolist()],
                "accuracy_mean": float(np.nanmean(acc)),
                "accuracy_std": float(np.nanstd(acc)),
                "f1_weighted_scores": [float(x) for x in np.asarray(f1_weighted, dtype=float).tolist()],
                "f1_weighted_mean": float(np.nanmean(f1_weighted)),
                "f1_weighted_std": float(np.nanstd(f1_weighted)),
                "precision_weighted_scores": [float(x) for x in np.asarray(precision_weighted, dtype=float).tolist()],
                "precision_weighted_mean": float(np.nanmean(precision_weighted)),
                "precision_weighted_std": float(np.nanstd(precision_weighted)),
            }
        else:
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
        return {"folds": int(cv)}


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


def _feature_importance(pipe: Pipeline, X_test, y_test, raw_feature_columns: List[str], cfg: DiagnosticsConfig, task: str = "regression") -> Dict[str, Any]:
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
                scoring="accuracy" if task == "classification" else "r2",
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


def _prediction_scores(pipe: Pipeline, X_test) -> Optional[np.ndarray]:
    if hasattr(pipe, "predict_proba"):
        try:
            score = pipe.predict_proba(X_test)
            if isinstance(score, np.ndarray) and score.ndim == 2 and score.shape[1] == 2:
                return np.asarray(score[:, 1], dtype=float)
            return np.asarray(score, dtype=float)
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        try:
            return np.asarray(pipe.decision_function(X_test), dtype=float)
        except Exception:
            pass
    return None


def _classification_roc(y_true, y_score) -> Dict[str, Any]:
    try:
        if y_score is None:
            return {"available": False}
        scores = np.asarray(y_score, dtype=float)
        if scores.ndim > 1:
            return {"available": False}
        fpr, tpr, _ = roc_curve(y_true, scores)
        score = float(auc(fpr, tpr))
        pts = [{"x": float(a), "y": float(b)} for a, b in zip(fpr.tolist(), tpr.tolist())]
        return {"available": True, "auc": score, "points": pts}
    except Exception:
        return {"available": False}


def _classification_metrics_payload(y_true, y_pred, y_score) -> Dict[str, Any]:
    try:
        acc = float(accuracy_score(y_true, y_pred))
        precision_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()
        decision_mean = None
        if y_score is not None:
            score_arr = np.asarray(y_score, dtype=float)
            if score_arr.ndim == 1:
                decision_mean = float(np.nanmean(score_arr))
        return {
            "available": True,
            "accuracy": acc,
            "precision_micro": precision_micro,
            "precision_weighted": precision_weighted,
            "recall_micro": recall_micro,
            "recall_weighted": recall_weighted,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "decision_mean": decision_mean,
            "confusion_matrix": cm,
        }
    except Exception:
        return {"available": False}


def _shared_nonshared_payload(df: pd.DataFrame) -> Dict[str, Any]:
    pairs = _detect_demand_breakdown_pairs(df)
    if not pairs:
        return {"available": False}
    a, b, label_a, label_b = pairs[0]
    sa = pd.to_numeric(df[a], errors="coerce").dropna()
    sb = pd.to_numeric(df[b], errors="coerce").dropna()
    if sa.empty or sb.empty:
        return {"available": False}
    size = int(min(len(sa), len(sb), 300))
    labels = [f"Obs {i + 1}" for i in range(size)]
    return {
        "available": True,
        "left_label": label_a,
        "right_label": label_b,
        "left_total": float(sa.sum()),
        "right_total": float(sb.sum()),
        "left_avg": float(sa.mean()),
        "right_avg": float(sb.mean()),
        "labels": labels,
        "left_values": [float(v) for v in sa.iloc[:size].to_list()],
        "right_values": [float(v) for v in sb.iloc[:size].to_list()],
    }
