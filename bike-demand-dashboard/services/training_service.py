import os
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from services.preprocess_service import build_preprocessor, split_xy
from services.model_registry import available_models as registry_available_models
from services.model_registry import get_estimator
from utils.metrics_utils import regression_metrics


def available_models() -> Dict[str, str]:
    # Return usable/installed models only (including optional ones if installed).
    # UI can still display the full catalog (including unavailable) via /api/models/catalog.
    return registry_available_models(task="regression")


def train_model(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    test_size: float,
    scale_numeric: bool,
    saved_models_folder: str,
    params: Dict[str, Any] | None = None,
    cv_folds: int | None = None,
    cross_validate: bool | None = None,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, Any]]:
    X, y = split_xy(df, target_column)
    # Defensive: ensure target has no NaN/inf before training.
    # If this triggers, it's almost always a dataset/target selection issue.
    if not np.isfinite(y).all():
        nan_count = int(np.isnan(y).sum())
        inf_count = int((~np.isfinite(y)).sum() - nan_count)
        msg = f"Target column '{target_column}' contains invalid values"
        parts = []
        if nan_count:
            parts.append(f"{nan_count} missing (NaN)")
        if inf_count:
            parts.append(f"{inf_count} infinite")
        msg += f" ({', '.join(parts)}). Choose a numeric target column and re-upload/clean the dataset."
        raise ValueError(msg)

    feature_columns = list(X.columns)
    ignored_columns = list(getattr(X, "attrs", {}).get("ignored_columns", []))
    # Align y to X's index for safe sampling/CV.
    try:
        y_series = pd.Series(y, index=X.index)
    except Exception:
        y_series = None

    preprocessor, preprocess_schema = build_preprocessor(df, feature_columns, scale_numeric=scale_numeric)
    model = get_estimator(model_name, params=params or {}, task="regression")
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # Optional cross-validation (quick model sanity check)
    n_rows = int(X.shape[0])
    cv_k = int(cv_folds or 5)
    cv_k = max(2, min(10, cv_k))
    do_cv = bool(cross_validate) if cross_validate is not None else (n_rows <= 8000)
    cv_summary: Dict[str, Any] = {"enabled": False}
    if do_cv and n_rows >= 200:
        try:
            # Keep CV runtime bounded on large datasets
            max_cv_rows = 5000
            if n_rows > max_cv_rows:
                X_cv = X.sample(n=max_cv_rows, random_state=42)
                if y_series is not None:
                    y_cv = y_series.loc[X_cv.index].to_numpy(dtype=float)
                else:
                    y_cv = y[:max_cv_rows]
            else:
                X_cv = X
                y_cv = y_series.to_numpy(dtype=float) if y_series is not None else y

            splitter = KFold(n_splits=cv_k, shuffle=True, random_state=42)
            r2_scores = cross_val_score(pipeline, X_cv, y_cv, cv=splitter, scoring="r2")
            # sklearn provides neg RMSE scorer in recent versions
            rmse_scores = None
            try:
                rmse_scores = -cross_val_score(pipeline, X_cv, y_cv, cv=splitter, scoring="neg_root_mean_squared_error")
            except Exception:
                rmse_scores = None

            cv_summary = {
                "enabled": True,
                "folds": int(cv_k),
                "rows_used": int(getattr(X_cv, "shape", [len(y_cv)])[0]),
                "r2_scores": [float(x) for x in np.asarray(r2_scores, dtype=float).tolist()],
                "r2_mean": float(np.nanmean(r2_scores)),
                "r2_std": float(np.nanstd(r2_scores)),
            }
            if rmse_scores is not None:
                cv_summary.update(
                    {
                        "rmse_scores": [float(x) for x in np.asarray(rmse_scores, dtype=float).tolist()],
                        "rmse_mean": float(np.nanmean(rmse_scores)),
                        "rmse_std": float(np.nanstd(rmse_scores)),
                    }
                )
        except Exception:
            cv_summary = {"enabled": False}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42
    )

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time_sec = float(time.time() - t0)
    y_pred = pipeline.predict(X_test)

    n_features = _safe_feature_count(pipeline, fallback=len(feature_columns))
    metrics = regression_metrics(y_test, y_pred, n_features=n_features)

    # store small samples for charts (avoid huge payloads)
    sample_size = int(min(250, len(y_test)))
    sample_idx = np.linspace(0, len(y_test) - 1, num=sample_size, dtype=int) if sample_size else []
    chart_payload = {
        "y_true": [float(y_test[i]) for i in sample_idx],
        "y_pred": [float(y_pred[i]) for i in sample_idx],
        "residuals": [float(y_test[i] - y_pred[i]) for i in sample_idx],
    }

    meta = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "ignored_columns": ignored_columns,
        "preprocess_schema": preprocess_schema,
        "chart_payload": chart_payload,
        "feature_hints": build_feature_hints(df, feature_columns),
        "training": {
            "train_time_sec": train_time_sec,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "test_size": float(test_size),
            "scale_numeric": bool(scale_numeric),
            "params": params or {},
            "cross_validation": cv_summary,
        },
    }

    # Save as a pipeline (preprocess + model)
    os.makedirs(saved_models_folder, exist_ok=True)
    return pipeline, metrics, meta


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)


def build_feature_hints(df: pd.DataFrame, feature_columns):
    hints = {}
    for col in feature_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce")
            hints[col] = {
                "type": "numeric",
                "min": float(np.nanmin(s.to_numpy())) if s.notna().any() else None,
                "max": float(np.nanmax(s.to_numpy())) if s.notna().any() else None,
                "mean": float(np.nanmean(s.to_numpy())) if s.notna().any() else None,
            }
        else:
            values = series.dropna().astype(str).value_counts().head(30).index.tolist()
            hints[col] = {"type": "categorical", "values": values}
    return hints


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _safe_feature_count(pipeline: Pipeline, fallback: int) -> int:
    try:
        pre = pipeline.named_steps["preprocess"]
        # works after fit
        if hasattr(pre, "get_feature_names_out"):
            return len(pre.get_feature_names_out())
    except Exception:
        pass
    return int(fallback)
