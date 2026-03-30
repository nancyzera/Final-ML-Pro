from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from services.dataset_service import read_dataset
from utils.helpers import json_loads


def build_ai_context(*, dataset_row, model_row) -> Dict[str, Any]:
    meta = json_loads(getattr(model_row, "meta_json", None), default={})
    dataset_path = getattr(dataset_row, "filepath", None)
    df = read_dataset(dataset_path) if dataset_path else None

    target = (getattr(dataset_row, "target_column", None) or meta.get("target_column") or "").strip()
    top_factors = meta.get("top_factors") or []
    factor_importances = meta.get("factor_importances") or []
    ignored_columns = meta.get("ignored_columns") or []

    correlations = []
    if df is not None and target and target in df.columns:
        try:
            y = pd.to_numeric(df[target], errors="coerce")
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            for c in numeric_cols:
                if c == target:
                    continue
                x = pd.to_numeric(df[c], errors="coerce")
                tmp = pd.concat([x, y], axis=1).dropna()
                if len(tmp) < 40:
                    continue
                corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
                if corr == corr:
                    correlations.append({"feature": str(c), "corr": corr})
            correlations.sort(key=lambda d: abs(d["corr"]), reverse=True)
            correlations = correlations[:10]
        except Exception:
            correlations = []

    return {
        "dataset": {
            "id": getattr(dataset_row, "id", None),
            "filename": getattr(dataset_row, "filename", ""),
            "rows_count": getattr(dataset_row, "rows_count", None),
            "columns_count": getattr(dataset_row, "columns_count", None),
            "missing_values": getattr(dataset_row, "missing_values", None),
            "target_column": target,
            "ignored_columns": ignored_columns,
        },
        "model": {
            "id": getattr(model_row, "id", None),
            "model_name": getattr(model_row, "model_name", ""),
            "r2_score": float(getattr(model_row, "r2_score", 0.0) or 0.0),
            "adjusted_r2": float(getattr(model_row, "adjusted_r2", 0.0) or 0.0),
            "mae": float(getattr(model_row, "mae", 0.0) or 0.0),
            "mse": float(getattr(model_row, "mse", 0.0) or 0.0),
            "rmse": float(getattr(model_row, "rmse", 0.0) or 0.0),
        },
        "factors": {"top_factors": top_factors, "factor_importances": factor_importances, "correlations": correlations},
        "training": meta.get("training") or {},
    }


def build_prompt(*, ctx: Dict[str, Any], locale_context: str = "") -> str:
    ds = ctx.get("dataset") or {}
    m = ctx.get("model") or {}
    factors = ctx.get("factors") or {}
    training = ctx.get("training") or {}

    loc = (locale_context or "").strip()
    if not loc:
        loc = "the target city/region in the dataset"

    # Keep it deterministic and dashboard-friendly
    return f"""
You are a senior data scientist. Write a concise, executive-ready narrative (no bullet lists, no markdown).

Goal:
1) Summarize model performance and whether it is good/average/poor.
2) Explain which factors influence bike demand and how (directional reasoning).
3) Provide 2 short operational recommendations for improving data/model.
4) Mention any caveats (correlation ≠ causation, seasonality, missing values, etc).

Constraints:
- Plain paragraphs only. No bullet points. No hashtags. No JSON.
- Keep it readable for non-technical stakeholders.
- If you lack a location field, assume the deployment context is {loc}.

Dataset:
filename={ds.get('filename')}
rows={ds.get('rows_count')} columns={ds.get('columns_count')} missing_values={ds.get('missing_values')}
target_column={ds.get('target_column')}
ignored_feature_columns={ds.get('ignored_columns')}

Model:
name={m.get('model_name')}
R2={m.get('r2_score'):.3f} adjusted_R2={m.get('adjusted_r2'):.3f}
MAE={m.get('mae'):.3f} MSE={m.get('mse'):.3f} RMSE={m.get('rmse'):.3f}
train_meta={training}

Model factors (if available):
top_factors={factors.get('top_factors')}
factor_importances={factors.get('factor_importances')}
correlations={factors.get('correlations')}
""".strip()


def quality_label(r2: float) -> str:
    try:
        r2 = float(r2)
    except Exception:
        return "unknown"
    if r2 >= 0.80:
        return "good"
    if r2 >= 0.60:
        return "average"
    return "poor"

