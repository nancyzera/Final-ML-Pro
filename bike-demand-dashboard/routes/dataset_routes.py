import os
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from models.database import db
from models.dataset_model import UploadedDataset
from models.app_state import AppState
from models.dataset_artifact import DatasetArtifact
from services.dataset_service import dataset_preview, dataset_summary, infer_column_types, read_dataset
from services.cleaning_service import clean_dataset
from utils.file_utils import allowed_file, safe_filename
from models.trained_model import TrainedModel
from utils.helpers import json_dumps, json_loads


dataset_bp = Blueprint("dataset_bp", __name__, url_prefix="/api/datasets")


@dataset_bp.post("/upload")
def upload_dataset():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file provided."}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"success": False, "message": "No selected file."}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type. Use CSV or Excel."}), 400

        filename = safe_filename(file.filename)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        stored_name = f"{stamp}__{filename}"
        original_path = os.path.join(current_app.config["UPLOAD_FOLDER"], stored_name)
        file.save(original_path)

        df_original = read_dataset(original_path)
        df_clean, cleaning_report = clean_dataset(df_original)

        cleaned_name = f"{stamp}__cleaned__{os.path.splitext(filename)[0]}.csv"
        cleaned_path = os.path.join(current_app.config["UPLOAD_FOLDER"], cleaned_name)
        df_clean.to_csv(cleaned_path, index=False)

        summary = dataset_summary(df_clean)
        preview = dataset_preview(df_clean)

        dataset = UploadedDataset(
            filename=filename,
            filepath=cleaned_path,
            rows_count=summary["rows_count"],
            columns_count=summary["columns_count"],
            missing_values=summary["missing_values"],
        )
        db.session.add(dataset)
        db.session.commit()

        # Store artifacts + cleaning report
        db.session.add(DatasetArtifact(dataset_id=dataset.id, artifact_type="original", path=original_path))
        db.session.add(DatasetArtifact(dataset_id=dataset.id, artifact_type="cleaned", path=cleaned_path))
        db.session.add(DatasetArtifact(dataset_id=dataset.id, artifact_type="cleaning_report", meta_json=json_dumps(cleaning_report)))
        db.session.commit()

        # Make this the active dataset
        state = AppState.query.get(1) or AppState(id=1)
        state.active_dataset_id = dataset.id
        db.session.add(state)
        db.session.commit()

        return jsonify(
            {
                "success": True,
                "message": "Dataset uploaded successfully.",
                "data": {
                    "dataset": _dataset_to_dict(dataset),
                    "summary": summary,
                    "preview": preview,
                    "cleaning_report": cleaning_report,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@dataset_bp.get("")
def list_datasets():
    datasets = UploadedDataset.query.order_by(UploadedDataset.uploaded_at.desc()).all()
    return jsonify({"success": True, "data": [_dataset_to_dict(d) for d in datasets]})


@dataset_bp.get("/<int:dataset_id>/preview")
def preview_dataset(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    try:
        df = read_dataset(dataset.filepath)
        preview = dataset_preview(df)
        return jsonify({"success": True, "data": {"dataset": _dataset_to_dict(dataset), "preview": preview}})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@dataset_bp.get("/<int:dataset_id>/summary")
def summary_dataset(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    try:
        df = read_dataset(dataset.filepath)
        summary = dataset_summary(df)
        report_row = (
            DatasetArtifact.query.filter_by(dataset_id=dataset.id, artifact_type="cleaning_report")
            .order_by(DatasetArtifact.created_at.desc())
            .first()
        )
        cleaning_report = json_loads(report_row.meta_json, default=None) if report_row else None
        return jsonify(
            {"success": True, "data": {"dataset": _dataset_to_dict(dataset), "summary": summary, "cleaning_report": cleaning_report}}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@dataset_bp.get("/<int:dataset_id>/profile")
def dataset_profile(dataset_id: int):
    """
    Lightweight scan for training: separates numeric/categorical/datetime, estimates one-hot dimensionality,
    and identifies timestamp-like columns that will be ignored during training.
    """
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    try:
        import numpy as np
        import pandas as pd

        df = read_dataset(dataset.filepath)
        numeric_cols, categorical_cols, datetime_cols = infer_column_types(df)

        approx_dim = int(len(numeric_cols))
        cat_card = {}
        for c in categorical_cols:
            try:
                n = int(df[c].dropna().astype(str).nunique())
            except Exception:
                n = 0
            cat_card[str(c)] = int(n)
            approx_dim += int(min(n, 50)) if n > 0 else 0

        ignored = set(datetime_cols or [])
        for col in df.columns:
            if col in ignored:
                continue
            name = str(col).strip().lower()
            if not name:
                continue
            if "timestamp" in name or "datetime" in name or name in {"timestamp", "datetime", "date", "time"} or name.endswith("_at"):
                ignored.add(col)
                continue
            if (("date" in name) or ("time" in name)) and pd.api.types.is_numeric_dtype(df[col]):
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().any():
                    med = float(np.nanmedian(s.to_numpy(dtype=float)))
                    if med >= 1e9:
                        ignored.add(col)

        return jsonify(
            {
                "success": True,
                "data": {
                    "rows_count": int(df.shape[0]),
                    "columns_count": int(df.shape[1]),
                    "numeric_count": int(len(numeric_cols)),
                    "categorical_count": int(len(categorical_cols)),
                    "datetime_count": int(len(datetime_cols)),
                    "numeric_columns": numeric_cols[:40],
                    "categorical_columns": categorical_cols[:40],
                    "datetime_columns": datetime_cols[:40],
                    "categorical_cardinality": cat_card,
                    "approx_feature_dim": int(approx_dim),
                    "ignored_timestamp_columns": sorted(ignored),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@dataset_bp.get("/<int:dataset_id>/status")
def dataset_status(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    preprocess_path = os.path.join(current_app.config["SAVED_MODELS_FOLDER"], f"preprocess_dataset_{dataset.id}.joblib")
    models_count = TrainedModel.query.filter_by(dataset_id=dataset.id).count()
    best_model = TrainedModel.query.filter_by(dataset_id=dataset.id).order_by(TrainedModel.r2_score.desc()).first()
    return jsonify(
        {
            "success": True,
            "data": {
                "dataset_id": dataset.id,
                "has_target": bool(dataset.target_column),
                "has_preprocessor": os.path.exists(preprocess_path),
                "preprocess_path": preprocess_path if os.path.exists(preprocess_path) else None,
                "models_count": int(models_count),
                "best_model_id": best_model.id if best_model else None,
            },
        }
    )


@dataset_bp.post("/<int:dataset_id>/target")
def set_target(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    body = request.get_json(silent=True) or {}
    target_column = (body.get("target_column") or "").strip()
    if not target_column:
        return jsonify({"success": False, "message": "target_column is required."}), 400
    try:
        df = read_dataset(dataset.filepath)
        if target_column not in df.columns:
            return jsonify({"success": False, "message": "Target column not found in dataset."}), 400
        dataset.target_column = target_column
        db.session.commit()

        state = AppState.query.get(1) or AppState(id=1)
        state.active_dataset_id = dataset.id
        db.session.add(state)
        db.session.commit()

        return jsonify({"success": True, "message": "Target column saved.", "data": _dataset_to_dict(dataset)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@dataset_bp.delete("/<int:dataset_id>")
def delete_dataset(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    try:
        # Remove associated artifacts on disk (best-effort)
        artifacts = DatasetArtifact.query.filter_by(dataset_id=dataset.id).all()
        for a in artifacts:
            if a.path:
                try:
                    if os.path.exists(a.path):
                        os.remove(a.path)
                except Exception:
                    pass

        preprocess_path = os.path.join(current_app.config["SAVED_MODELS_FOLDER"], f"preprocess_dataset_{dataset.id}.joblib")
        try:
            if os.path.exists(preprocess_path):
                os.remove(preprocess_path)
        except Exception:
            pass

        # Remove saved model files for this dataset
        models = TrainedModel.query.filter_by(dataset_id=dataset.id).all()
        for m in models:
            try:
                if m.model_path and os.path.exists(m.model_path):
                    os.remove(m.model_path)
            except Exception:
                pass

        # Update state if needed
        state = AppState.query.get(1)
        if state:
            if state.active_dataset_id == dataset.id:
                state.active_dataset_id = None
            if state.active_model_id and any(m.id == state.active_model_id for m in models):
                state.active_model_id = None

        # Delete DB row (cascades models + predictions)
        db.session.delete(dataset)
        db.session.commit()

        # Pick a new active dataset/model if empty
        state = AppState.query.get(1)
        if state:
            if state.active_dataset_id is None:
                latest = UploadedDataset.query.order_by(UploadedDataset.uploaded_at.desc()).first()
                state.active_dataset_id = latest.id if latest else None
            if state.active_model_id is None and state.active_dataset_id:
                best = (
                    TrainedModel.query.filter_by(dataset_id=state.active_dataset_id)
                    .order_by(TrainedModel.r2_score.desc())
                    .first()
                )
                state.active_model_id = best.id if best else None
            db.session.commit()

        return jsonify({"success": True, "message": "Dataset deleted."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


def _dataset_to_dict(d: UploadedDataset):
    return {
        "id": d.id,
        "filename": d.filename,
        "filepath": d.filepath,
        "target_column": d.target_column,
        "rows_count": d.rows_count,
        "columns_count": d.columns_count,
        "missing_values": d.missing_values,
        "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
    }
