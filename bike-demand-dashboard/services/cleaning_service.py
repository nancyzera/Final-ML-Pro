from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from services.dataset_service import infer_column_types


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safe, beginner-friendly cleaning to improve downstream ML robustness:
    - Drop fully-empty columns
    - Drop 'Unnamed: ...' index columns
    - Strip whitespace in string columns
    - Normalize common missing tokens to NaN
    - Convert numeric-looking object columns to numeric
    - Convert datetime-looking object columns to datetime
    - Drop fully-empty rows and duplicate rows
    Returns cleaned dataframe + a cleaning report.
    """
    before_rows, before_cols = int(df.shape[0]), int(df.shape[1])
    report: Dict[str, Any] = {
        "before": {"rows": before_rows, "cols": before_cols},
        "dropped_columns": [],
        "converted_columns": {},
        "filled_missing": [],
        "filled_missing_total": 0,
        "dropped_rows": {"empty": 0, "duplicates": 0},
        "missing_values_before": int(df.isna().sum().sum()),
    }

    out = df.copy()

    # Drop "Unnamed: 0" style columns
    unnamed = [c for c in out.columns if str(c).strip().lower().startswith("unnamed")]
    if unnamed:
        report["dropped_columns"].extend([str(c) for c in unnamed])
        out = out.drop(columns=unnamed, errors="ignore")

    # Drop fully-empty columns
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        report["dropped_columns"].extend([str(c) for c in empty_cols])
        out = out.drop(columns=empty_cols, errors="ignore")

    # Replace common missing tokens
    missing_tokens = {"", "na", "n/a", "null", "none", "-", "nan"}
    for col in out.columns:
        if out[col].dtype == object:
            s = out[col].astype(str)
            # Keep real NaNs (string conversion makes "nan"), so operate on original
            raw = out[col]
            cleaned = raw.where(~raw.astype(str).str.strip().str.lower().isin(missing_tokens), np.nan)
            out[col] = cleaned

    # Strip whitespace in object columns
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip().replace({"nan": np.nan})

    # Try type conversions for object columns
    numeric_cols, categorical_cols, datetime_cols = infer_column_types(out)
    for col in list(out.columns):
        if col in datetime_cols and out[col].dtype == object:
            try:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                report["converted_columns"][str(col)] = "datetime"
            except Exception:
                pass
        elif col in numeric_cols and out[col].dtype == object:
            try:
                conv = pd.to_numeric(out[col], errors="coerce")
                # only accept if conversion keeps most values
                if conv.notna().mean() >= 0.85:
                    out[col] = conv
                    report["converted_columns"][str(col)] = "numeric"
            except Exception:
                pass

    # Drop fully-empty rows
    before = int(out.shape[0])
    out = out.dropna(how="all")
    report["dropped_rows"]["empty"] = int(before - out.shape[0])

    # Drop duplicate rows
    before = int(out.shape[0])
    out = out.drop_duplicates()
    report["dropped_rows"]["duplicates"] = int(before - out.shape[0])

    # Auto-fill missing values (imputation) to create a merged, ML-ready dataset CSV.
    # Note: The ML pipeline also imputes; this step makes the saved cleaned CSV directly usable
    # for preview/exports and reduces user friction with missing values.
    num_cols, cat_cols, dt_cols = infer_column_types(out)
    filled_total = 0

    # Numeric -> median
    for col in num_cols:
        try:
            s = pd.to_numeric(out[col], errors="coerce")
            miss = int(s.isna().sum())
            if miss <= 0:
                continue
            med = float(np.nanmedian(s.to_numpy(dtype=float))) if s.notna().any() else 0.0
            out[col] = s.fillna(med)
            filled_total += miss
            report["filled_missing"].append({"column": str(col), "strategy": "median", "filled": miss, "value": med})
        except Exception:
            continue

    # Datetime -> forward fill then backward fill
    for col in dt_cols:
        try:
            s = out[col]
            miss = int(pd.isna(s).sum())
            if miss <= 0:
                continue
            if not pd.api.types.is_datetime64_any_dtype(s):
                s = pd.to_datetime(s, errors="coerce")
            s2 = s.ffill().bfill()
            out[col] = s2
            filled = int(pd.isna(s).sum() - pd.isna(s2).sum())
            if filled > 0:
                filled_total += filled
                report["filled_missing"].append({"column": str(col), "strategy": "ffill_bfill", "filled": filled})
        except Exception:
            continue

    # Categorical -> mode (fallback "Unknown")
    for col in cat_cols:
        try:
            s = out[col]
            miss = int(pd.isna(s).sum())
            if miss <= 0:
                continue
            non = s.dropna().astype(str)
            fill_value = non.mode().iloc[0] if not non.empty else "Unknown"
            out[col] = s.fillna(fill_value)
            filled_total += miss
            report["filled_missing"].append({"column": str(col), "strategy": "mode", "filled": miss, "value": str(fill_value)})
        except Exception:
            continue

    report["filled_missing_total"] = int(filled_total)
    report["missing_values_after"] = int(out.isna().sum().sum())
    report["after"] = {"rows": int(out.shape[0]), "cols": int(out.shape[1])}
    return out, report
