import os
from typing import Dict, List, Tuple

import pandas as pd
import warnings


def read_dataset(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        # Try common CSV defaults; fall back to python engine for odd delimiters
        try:
            # low_memory=False prevents pandas chunked inference that can trigger noisy DtypeWarning
            df = pd.read_csv(filepath, low_memory=False)
            return _normalize_columns(df)
        except Exception:
            df = pd.read_csv(filepath, engine="python", sep=None, low_memory=False)
            return _normalize_columns(df)
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath, engine="openpyxl")
        return _normalize_columns(df)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def dataset_preview(df: pd.DataFrame, n: int = 20) -> Dict:
    preview_df = df.head(n).copy()
    preview_df = _safe_for_json(preview_df)
    return {"rows": preview_df.to_dict(orient="records"), "columns": list(preview_df.columns)}


def dataset_summary(df: pd.DataFrame) -> Dict:
    rows_count, columns_count = int(df.shape[0]), int(df.shape[1])
    missing_values = int(df.isna().sum().sum())
    columns = list(df.columns)
    numeric_columns, categorical_columns, datetime_columns = infer_column_types(df)
    return {
        "rows_count": rows_count,
        "columns_count": columns_count,
        "missing_values": missing_values,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
    }


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
        else:
            # Attempt to parse as datetime if it looks like dates
            if _looks_datetime(series):
                datetime_columns.append(col)
            elif _looks_numeric(series):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

    return numeric_columns, categorical_columns, datetime_columns


def _looks_datetime(series: pd.Series) -> bool:
    if series.empty:
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.7


def _looks_numeric(series: pd.Series) -> bool:
    if series.empty:
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        numeric = pd.to_numeric(sample, errors="coerce")
    return numeric.notna().mean() >= 0.85


def coerce_datetime_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce")


def _safe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    # Convert NaN/NaT to None and datetime to ISO strings
    result = df.copy()
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            result[col] = result[col].where(result[col].notna(), None)
    return result


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column names are strings, stripped, and unique.
    Duplicate column names can cause pandas to return a DataFrame for df['col'],
    which breaks numeric conversion and chart logic.
    """
    cols = [str(c).strip() if c is not None else "" for c in df.columns]
    df = df.copy()
    df.columns = _make_unique(cols)
    return df


def _make_unique(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        base = c if c else "column"
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}__{seen[base]}")
    return out
