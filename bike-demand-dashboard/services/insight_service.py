from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.dataset_service import coerce_datetime_column, infer_column_types


def build_insights(df: pd.DataFrame, target_column: Optional[str], model_meta: Optional[Dict[str, Any]] = None) -> Dict:
    numeric_columns, categorical_columns, datetime_columns = infer_column_types(df)
    target = target_column if target_column in df.columns else _pick_target(numeric_columns)
    time_col = _pick_time_column(datetime_columns)

    insights: List[str] = []

    if target:
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if not s.empty:
            insights.append(f"Demand range: {float(s.min()):.1f} to {float(s.max()):.1f} (median {float(s.median()):.1f}).")

    if time_col and target:
        tmp = df[[time_col, target]].copy()
        tmp[time_col] = coerce_datetime_column(tmp, time_col)
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            tmp["hour"] = tmp[time_col].dt.hour
            by_hour = tmp.groupby("hour")[target].mean().sort_values(ascending=False)
            peak_hour = int(by_hour.index[0])
            insights.append(f"Peak demand hour (avg): {peak_hour:02d}:00.")

    temp_col = _pick_temperature_column(numeric_columns)
    if target and temp_col and temp_col != target:
        tmp = df[[temp_col, target]].copy()
        tmp[temp_col] = pd.to_numeric(tmp[temp_col], errors="coerce")
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) >= 20:
            corr = float(tmp[temp_col].corr(tmp[target]))
            if abs(corr) >= 0.2:
                direction = "increases" if corr > 0 else "decreases"
                insights.append(f"As `{temp_col}` rises, demand generally {direction} (corr {corr:.2f}).")

    if model_meta:
        perf = model_meta.get("performance_label")
        if perf:
            insights.append(f"Model performance looks {perf}.")
        factors = model_meta.get("top_factors") or []
        if factors:
            insights.append("Top influencing factors: " + ", ".join(factors[:5]) + ".")

    return {"target_column": target, "time_column": time_col, "insights": insights}


def model_quality_label(r2_score: float) -> str:
    if r2_score >= 0.85:
        return "excellent"
    if r2_score >= 0.7:
        return "good"
    if r2_score >= 0.5:
        return "average"
    return "poor"


def extract_top_factors(pipeline, raw_feature_columns, top_k: int = 8):
    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocess"]
        feature_names = None
        if hasattr(pre, "get_feature_names_out"):
            feature_names = list(pre.get_feature_names_out())
        if feature_names is None:
            feature_names = list(raw_feature_columns)

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.ravel(np.array(model.coef_, dtype=float))
            importances = np.abs(coef)

        if importances is None or len(importances) != len(feature_names):
            return []

        idx = np.argsort(importances)[::-1][:top_k]
        return [str(feature_names[i]) for i in idx]
    except Exception:
        return []


def extract_factor_importances(pipeline, raw_feature_columns, top_k: int = 12):
    """
    Returns a list of {feature, importance} sorted descending.
    Importance is model-dependent (tree importances or abs(linear coefficients)) and normalized.
    """
    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocess"]
        feature_names = None
        if hasattr(pre, "get_feature_names_out"):
            feature_names = list(pre.get_feature_names_out())
        if feature_names is None:
            feature_names = list(raw_feature_columns)

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.ravel(np.array(model.coef_, dtype=float))
            importances = np.abs(coef)

        if importances is None or len(importances) != len(feature_names):
            return []

        importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(importances))
        if not np.isfinite(total) or total <= 0:
            total = 1.0
        norm = importances / total
        idx = np.argsort(norm)[::-1][:top_k]
        out = [{"feature": str(feature_names[i]), "importance": float(norm[i])} for i in idx]
        return out
    except Exception:
        return []

def _pick_time_column(datetime_columns):
    for p in ["datetime", "date", "timestamp", "time"]:
        for c in datetime_columns:
            if p in c.lower():
                return c
    return datetime_columns[0] if datetime_columns else None


def _pick_temperature_column(numeric_columns):
    for p in ["temp", "temperature"]:
        for c in numeric_columns:
            if p in c.lower():
                return c
    return numeric_columns[0] if numeric_columns else None


def _pick_target(numeric_columns):
    for p in ["count", "demand", "rides", "cnt", "target", "y"]:
        for c in numeric_columns:
            if c.lower() == p or p in c.lower():
                return c
    return numeric_columns[0] if numeric_columns else None
