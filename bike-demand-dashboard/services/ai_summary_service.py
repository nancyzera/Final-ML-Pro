from __future__ import annotations

import re
from typing import Any, Dict, List

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
- Do not answer with yes/no only. Write at least 4 complete sentences.
- Ground every claim in the provided metrics, factors, or correlations.
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


def is_low_signal_response(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return True
    if len(s) < 40:
        return True
    lowered = re.sub(r"[^a-z]+", " ", s.lower()).strip()
    if lowered in {"yes", "no", "okay", "ok", "sure", "true", "false"}:
        return True
    tokens = [tok for tok in lowered.split() if tok]
    if len(tokens) <= 4:
        return True
    return False


def build_local_summary(*, ctx: Dict[str, Any], locale_context: str = "") -> str:
    ds = ctx.get("dataset") or {}
    m = ctx.get("model") or {}
    factors = ctx.get("factors") or {}
    training = ctx.get("training") or {}
    task = str(training.get("task") or "regression")
    loc = (locale_context or "").strip() or "the operating area"

    sentences: List[str] = []
    model_name = str(m.get("model_name") or "This model")
    target = str(ds.get("target_column") or "demand")
    rows = ds.get("rows_count")
    cols = ds.get("columns_count")
    missing = ds.get("missing_values")

    if task == "classification":
        accuracy = float(training.get("cross_validation", {}).get("accuracy_mean") or 0.0)
        support = training.get("cross_validation", {}).get("rows_used") or rows
        sentences.append(
            f"{model_name} is being used as a classification model for {target}, based on about {int(support) if support else 'the available'} records, and its recent validation accuracy is {accuracy:.3f}."
        )
    else:
        r2 = float(m.get("r2_score") or 0.0)
        rmse = float(m.get("rmse") or 0.0)
        quality = quality_label(r2)
        shape = f"{int(rows)} rows and {int(cols)} columns" if rows and cols else "the uploaded dataset"
        sentences.append(
            f"{model_name} was trained on {shape} and currently shows {quality} predictive strength with R² {r2:.3f} and RMSE {rmse:.3f} for {target}."
        )

    if missing is not None:
        sentences.append(
            f"The dataset contains {int(missing)} missing values, so data quality is good enough for modeling but still worth monitoring before making planning decisions in {loc}."
        )

    top_factors = factors.get("top_factors") or []
    corrs = factors.get("correlations") or []
    if top_factors:
        sentences.append(
            f"The strongest model signals are coming from {', '.join(str(x) for x in top_factors[:4])}, which suggests these variables are driving most of the variation in observed demand."
        )

    if corrs:
        strongest = corrs[0]
        feature = strongest.get("feature") or "the strongest feature"
        corr = float(strongest.get("corr") or 0.0)
        direction = "higher" if corr > 0 else "lower"
        sentences.append(
            f"The clearest linear relationship in the raw data is {feature}, where {direction} values are associated with {target} changes with correlation {corr:.2f}; this is useful for planning, although it should not be treated as proof of causation."
        )

    cv = training.get("cross_validation") or {}
    if cv.get("enabled"):
        if task == "classification":
            sentences.append(
                f"Cross-validation is enabled, which means the classification scores are being checked across multiple folds instead of a single split, so the reported performance is more stable than a one-off test result."
            )
        else:
            r2_mean = cv.get("r2_mean")
            r2_std = cv.get("r2_std")
            if r2_mean is not None and r2_std is not None:
                sentences.append(
                    f"Cross-validation shows an average R² of {float(r2_mean):.3f} with spread {float(r2_std):.3f}, so we can judge whether this model is consistent or still sensitive to the train-test split."
                )

    sentences.append(
        f"The practical next step is to collect cleaner seasonal, weather, and calendar signals for {loc}, then compare whether the current feature set keeps the error low during peak-demand periods."
    )
    return " ".join(s.strip() for s in sentences if s and str(s).strip())


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
