import os

import joblib
from flask import Blueprint, current_app, jsonify, request
from sklearn.model_selection import train_test_split

from models.database import db
from models.dataset_model import UploadedDataset
from services.dataset_service import dataset_summary, read_dataset
from services.preprocess_service import build_preprocessor, split_xy
from utils.helpers import clamp, safe_float


preprocess_bp = Blueprint("preprocess_bp", __name__, url_prefix="/api/preprocess")


@preprocess_bp.post("/<int:dataset_id>")
def preprocess_dataset(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    body = request.get_json(silent=True) or {}
    target_column = (body.get("target_column") or dataset.target_column or "").strip()
    test_size = clamp(safe_float(body.get("test_size"), 0.2), 0.05, 0.5)
    scale_numeric = bool(body.get("scale_numeric", True))

    if not target_column:
        return jsonify({"success": False, "message": "target_column is required."}), 400

    try:
        df = read_dataset(dataset.filepath)
        X, y = split_xy(df, target_column)
        feature_columns = list(X.columns)
        ignored_columns = list(getattr(X, "attrs", {}).get("ignored_columns", []))
        dropped_target_rows = int(getattr(X, "attrs", {}).get("dropped_target_rows", 0) or 0)

        preprocessor, schema = build_preprocessor(df, feature_columns, scale_numeric=scale_numeric)

        # Fit preprocessor only to validate and save it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=42)
        preprocessor.fit(X_train)

        os.makedirs(current_app.config["SAVED_MODELS_FOLDER"], exist_ok=True)
        preprocess_path = os.path.join(current_app.config["SAVED_MODELS_FOLDER"], f"preprocess_dataset_{dataset.id}.joblib")
        joblib.dump(preprocessor, preprocess_path)

        dataset.target_column = target_column
        db.session.commit()

        summary = dataset_summary(df)
        preprocess_summary = {
            "target_column": target_column,
            "test_size": float(test_size),
            "scale_numeric": bool(scale_numeric),
            "feature_columns": feature_columns,
            "ignored_columns": ignored_columns,
            "dropped_target_rows": dropped_target_rows,
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "saved_preprocessor": preprocess_path,
            "schema": schema,
        }
        return jsonify({"success": True, "message": "Preprocessing configuration saved.", "data": {"summary": summary, "preprocess": preprocess_summary}})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
