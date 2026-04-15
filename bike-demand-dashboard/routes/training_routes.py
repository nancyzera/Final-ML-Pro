import os
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from models.database import db
from models.dataset_model import UploadedDataset
from models.app_state import AppState
from models.trained_model import TrainedModel
from services.dataset_service import read_dataset
from services.insight_service import extract_factor_importances, extract_top_factors, model_quality_label
from services.diagnostic_service import build_model_diagnostics
from services.model_registry import available_models as registry_available_models
from services.model_registry import get_model_task
from services.model_registry import get_catalog
from services.training_service import save_pipeline, train_model
from utils.helpers import clamp, json_dumps, json_loads, safe_float


training_bp = Blueprint("training_bp", __name__, url_prefix="/api")


@training_bp.get("/models/available")
def list_available_models():
    task = (request.args.get("task") or "").strip().lower()
    if task not in {"regression", "classification"}:
        task = None
    return jsonify({"success": True, "data": registry_available_models(task=task)})


@training_bp.get("/models/catalog")
def models_catalog():
    # Full catalog (includes optional/unavailable models + param schemas)
    return jsonify({"success": True, "data": get_catalog()})


@training_bp.post("/train")
def train():
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    model_name = (body.get("model_name") or "").strip()
    target_column = (body.get("target_column") or "").strip()
    test_size = clamp(safe_float(body.get("test_size"), 0.2), 0.05, 0.5)
    scale_numeric = bool(body.get("scale_numeric", True))
    params = body.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    cv_folds = body.get("cv_folds")
    try:
        cv_folds = int(cv_folds) if cv_folds is not None else None
    except Exception:
        cv_folds = None
    cross_validate = body.get("cross_validate")
    if cross_validate is not None:
        cross_validate = bool(cross_validate)
    task = (body.get("task") or "").strip().lower()

    if not dataset_id:
        return jsonify({"success": False, "message": "dataset_id is required."}), 400
    if not model_name:
        return jsonify({"success": False, "message": "model_name is required."}), 400

    dataset = UploadedDataset.query.get_or_404(int(dataset_id))
    if not target_column:
        target_column = dataset.target_column or ""
    if not target_column:
        return jsonify({"success": False, "message": "target_column is required."}), 400

    try:
        model_task = task if task in {"regression", "classification"} else get_model_task(model_name)
        df = read_dataset(dataset.filepath)
        pipeline, metrics, meta = train_model(
            df=df,
            target_column=target_column,
            model_name=model_name,
            test_size=float(test_size),
            scale_numeric=scale_numeric,
            saved_models_folder=current_app.config["SAVED_MODELS_FOLDER"],
            params=params,
            cv_folds=cv_folds,
            cross_validate=cross_validate,
            task=model_task,
        )

        is_classification = model_task == "classification"
        primary_score = float(metrics.get("accuracy", 0.0) if is_classification else metrics.get("r2_score", 0.0))
        secondary_score = float(metrics.get("f1_weighted", 0.0) if is_classification else metrics.get("rmse", 0.0))

        # Create DB record first to get id for filepath
        trained = TrainedModel(
            dataset_id=dataset.id,
            model_name=model_name,
            r2_score=primary_score,
            adjusted_r2=secondary_score,
            mae=float(metrics.get("mae", 0.0)),
            mse=float(metrics.get("mse", 0.0)),
            rmse=float(metrics.get("rmse", 0.0)),
            model_path="",
            trained_at=datetime.utcnow(),
        )
        db.session.add(trained)
        db.session.commit()

        model_path = os.path.join(current_app.config["SAVED_MODELS_FOLDER"], f"model_{trained.id}.joblib")
        save_pipeline(pipeline, model_path)

        top_factors = extract_top_factors(pipeline, meta.get("feature_columns", []), top_k=8)
        factor_importances = extract_factor_importances(pipeline, meta.get("feature_columns", []), top_k=12)
        performance_label = model_quality_label(primary_score)
        meta.update(
            {
                "metrics": metrics,
                "top_factors": top_factors,
                "factor_importances": factor_importances,
                "performance_label": performance_label,
                "display_metrics": {
                    "primary_name": "Accuracy" if is_classification else "R²",
                    "primary_value": primary_score,
                    "secondary_name": "F1 weighted" if is_classification else "RMSE",
                    "secondary_value": secondary_score,
                },
            }
        )

        trained.model_path = model_path
        trained.meta_json = json_dumps(meta)
        db.session.commit()

        dataset.target_column = target_column
        db.session.commit()

        state = AppState.query.get(1) or AppState(id=1)
        state.active_dataset_id = dataset.id
        state.active_model_id = trained.id
        db.session.add(state)
        db.session.commit()

        return jsonify(
            {
                "success": True,
                "message": "Model trained successfully.",
                "data": {"model": _model_to_dict(trained), "meta": meta},
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@training_bp.get("/models")
def list_models():
    models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
    return jsonify({"success": True, "data": [_model_to_dict(m) for m in models]})


@training_bp.get("/models/<int:model_id>/metrics")
def model_metrics(model_id: int):
    model = TrainedModel.query.get_or_404(model_id)
    meta = json_loads(model.meta_json, default={})
    return jsonify({"success": True, "data": {"model": _model_to_dict(model), "meta": meta}})


@training_bp.get("/models/<int:model_id>/diagnostics")
def model_diagnostics(model_id: int):
    model = TrainedModel.query.get_or_404(model_id)
    dataset = UploadedDataset.query.get_or_404(model.dataset_id)
    meta = json_loads(model.meta_json, default={})
    target = dataset.target_column or meta.get("target_column")
    if not target:
        return jsonify({"success": False, "message": "Dataset has no target column set."}), 400

    try:
        df = read_dataset(dataset.filepath)
        training_meta = meta.get("training") or {}
        test_size = float(training_meta.get("test_size", 0.2))
        scale_numeric = bool(training_meta.get("scale_numeric", True))
        task = str(meta.get("task") or "regression")
        payload = build_model_diagnostics(
            df=df,
            target_column=target,
            model_name=model.model_name,
            test_size=test_size,
            scale_numeric=scale_numeric,
            task=task,
            task_details=training_meta.get("task_details") or {},
        )
        return jsonify({"success": True, "data": {"model": _model_to_dict(model), "diagnostics": payload}})
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


def _model_to_dict(m: TrainedModel):
    meta = json_loads(m.meta_json, default={})
    display = meta.get("display_metrics") or {}
    metrics = meta.get("metrics") or {}
    return {
        "id": m.id,
        "dataset_id": m.dataset_id,
        "model_name": m.model_name,
        "task": meta.get("task") or "regression",
        "r2_score": m.r2_score,
        "adjusted_r2": m.adjusted_r2,
        "mae": m.mae,
        "mse": m.mse,
        "rmse": m.rmse,
        "primary_metric_name": display.get("primary_name"),
        "primary_metric_value": display.get("primary_value", m.r2_score),
        "secondary_metric_name": display.get("secondary_name"),
        "secondary_metric_value": display.get("secondary_value", m.adjusted_r2),
        "accuracy": metrics.get("accuracy"),
        "precision_micro": metrics.get("precision_micro"),
        "precision_weighted": metrics.get("precision_weighted"),
        "f1_micro": metrics.get("f1_micro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "decision_mean": metrics.get("decision_mean"),
        "roc_auc": metrics.get("roc_auc"),
        "model_path": m.model_path,
        "trained_at": m.trained_at.isoformat() if m.trained_at else None,
    }
