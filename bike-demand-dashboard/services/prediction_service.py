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
    return pred, details


def _try_float(value: str):
    try:
        return float(value)
    except Exception:
        return None

