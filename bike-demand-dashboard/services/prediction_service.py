from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from services.training_service import load_pipeline


def predict_from_inputs(model_path: str, inputs: Dict[str, Any], feature_columns) -> Tuple[float, Dict]:
    pipeline = load_pipeline(model_path)

    missing = [c for c in feature_columns if c not in inputs]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    row = {c: inputs.get(c) for c in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    # Light coercion for numeric-like strings
    for col in X.columns:
        if X[col].dtype == object:
            val = X.loc[0, col]
            if isinstance(val, str) and val.strip() != "":
                maybe_num = _try_float(val)
                if maybe_num is not None:
                    X.loc[0, col] = maybe_num

    pred = float(np.ravel(pipeline.predict(X))[0])
    details = {"model_path": model_path}
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            arr = np.asarray(proba, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                details["predicted_probability"] = float(arr[0, 1])
            elif arr.ndim >= 1:
                details["predicted_probability"] = float(arr.ravel()[0])
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            score = np.asarray(pipeline.decision_function(X), dtype=float).ravel()
            if score.size:
                details["decision_score"] = float(score[0])
        except Exception:
            pass
    return pred, details


def _try_float(value: str):
    try:
        return float(value)
    except Exception:
        return None
