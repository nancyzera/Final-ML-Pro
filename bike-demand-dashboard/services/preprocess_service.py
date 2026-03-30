from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.dataset_service import infer_column_types


def build_preprocessor(
    df: pd.DataFrame,
    feature_columns: List[str],
    scale_numeric: bool = True,
) -> Tuple[ColumnTransformer, Dict]:
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    schema = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "scale_numeric": bool(scale_numeric),
    }
    return preprocessor, schema


def split_xy(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # If columns are duplicated, pandas returns a DataFrame for df[target_column].
    # This breaks downstream numeric coercion and can surface as confusing training errors.
    obj = df.loc[:, target_column]
    if isinstance(obj, pd.DataFrame):
        raise ValueError(
            f"Target column '{target_column}' is ambiguous because the dataset contains duplicate column names. "
            "Rename duplicate columns and re-upload, or pick the correct target column name."
        )

    y_raw = obj
    if pd.api.types.is_datetime64_any_dtype(y_raw):
        raise ValueError(
            f"Target column '{target_column}' is not suitable: it looks like a datetime/timestamp. "
            "Choose a numeric demand column (e.g., count/rides/demand)."
        )

    # Always coerce to numeric to ensure a consistent 1D float target.
    # This also normalizes pandas nullable numeric dtypes.
    y = pd.to_numeric(y_raw, errors="coerce")

    # Drop rows where target is missing/non-numeric to prevent scikit-learn errors (y cannot contain NaN)
    if len(y):
        mask = y.notna()
        dropped_target_rows = int((~mask).sum())
    else:
        mask = None
        dropped_target_rows = 0

    missing_ratio = float(dropped_target_rows / len(y)) if len(y) else 1.0
    if missing_ratio > 0.2:
        pct = missing_ratio * 100
        raise ValueError(
            f"Target column '{target_column}' is not suitable: {pct:.1f}% of values are missing or non-numeric. "
            "Choose a numeric demand column (e.g., count/rides/demand) or clean the dataset."
        )

    # Build features and align to the target mask (after dropping NaN targets)
    X = df.drop(columns=[target_column])
    if mask is not None and dropped_target_rows:
        X = X.loc[mask].copy()
        y = y.loc[mask]
        try:
            X.attrs["dropped_target_rows"] = dropped_target_rows
        except Exception:
            pass

    # very low variance target makes training meaningless
    try:
        std = float(np.nanstd(y.to_numpy(dtype=float)))
        if std == 0.0:
            raise ValueError(
                f"Target column '{target_column}' is not suitable: values have zero variance (all the same). "
                "Choose a different target."
            )
    except ValueError:
        raise
    except Exception:
        pass

    # Drop timestamp/datetime columns from features for better generalization and to avoid noisy leakage.
    # Keep them available for analytics charts (handled separately).
    ignored = _infer_timestamp_columns(df)
    if ignored:
        drop_cols = [c for c in ignored if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols, errors="ignore")
            # Store for UI summaries / training metadata (optional)
            try:
                X.attrs["ignored_columns"] = sorted(drop_cols)
            except Exception:
                pass

    y_arr = y.to_numpy(dtype=float)
    if y_arr.size == 0:
        raise ValueError(
            f"Target column '{target_column}' is not suitable: after removing missing/non-numeric targets, "
            "no rows remain for training. Choose a different target column."
        )
    if np.isnan(y_arr).any():
        # Defensive guard: should be prevented by the mask, but some edge cases can sneak through.
        bad = int(np.isnan(y_arr).sum())
        raise ValueError(
            f"Target column '{target_column}' still contains {bad} missing values after cleaning. "
            "Choose a numeric target column with fewer missing values."
        )

    return X, y_arr


def _infer_timestamp_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify timestamp/datetime columns that should not be used as ML features.
    Rules:
    - Any column inferred as datetime-like.
    - Common timestamp naming patterns (timestamp/datetime/date/time/*_at).
    - Numeric epoch-like time columns when name suggests time/date.
    """
    _numeric, _categorical, datetime_cols = infer_column_types(df)
    ignored = set(datetime_cols or [])

    for col in df.columns:
        if col in ignored:
            continue
        name = str(col).strip().lower()
        if not name:
            continue

        # Strong name matches
        if "timestamp" in name or "datetime" in name or name in {"timestamp", "datetime", "date", "time"} or name.endswith("_at"):
            ignored.add(col)
            continue

        # Epoch-like numeric timestamps (seconds/ms) when name hints time/date
        if (("date" in name) or ("time" in name)) and pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                med = float(np.nanmedian(s.to_numpy(dtype=float)))
                # ~2001-09-09 in seconds is 1e9; ms would be 1e12+
                if med >= 1e9:
                    ignored.add(col)

    return sorted(ignored)
