from typing import Optional

from flask import Blueprint, jsonify

from models.dataset_model import UploadedDataset
from models.prediction_history import PredictionHistory
from models.trained_model import TrainedModel
from services.chart_service import build_charts_payload
from services.dataset_service import read_dataset
from services.insight_service import build_insights
from utils.helpers import json_loads


dashboard_bp = Blueprint("dashboard_bp", __name__, url_prefix="/api/dashboard")


@dashboard_bp.get("/stats")
def dashboard_stats():
    total_datasets = UploadedDataset.query.count()
    total_models = TrainedModel.query.count()
    total_predictions = PredictionHistory.query.count()

    best_model = TrainedModel.query.order_by(TrainedModel.r2_score.desc()).first()
    latest_model = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).first()
    latest_dataset = UploadedDataset.query.order_by(UploadedDataset.uploaded_at.desc()).first()

    best_r2 = float(best_model.r2_score) if best_model else None
    current_rmse = float(latest_model.rmse) if latest_model else None
    latest_training_date = latest_model.trained_at.isoformat() if latest_model and latest_model.trained_at else None

    recent_activity = []
    for d in UploadedDataset.query.order_by(UploadedDataset.uploaded_at.desc()).limit(3).all():
        recent_activity.append({"type": "dataset", "title": f"Uploaded {d.filename}", "at": d.uploaded_at.isoformat()})
    for m in TrainedModel.query.order_by(TrainedModel.trained_at.desc()).limit(3).all():
        recent_activity.append(
            {"type": "model", "title": f"Trained {m.model_name} (R² {m.r2_score:.3f})", "at": m.trained_at.isoformat()}
        )
    for p in PredictionHistory.query.order_by(PredictionHistory.predicted_at.desc()).limit(3).all():
        recent_activity.append(
            {"type": "prediction", "title": f"Prediction {p.predicted_value:.2f}", "at": p.predicted_at.isoformat()}
        )
    recent_activity = sorted(recent_activity, key=lambda x: x["at"], reverse=True)[:8]

    return jsonify(
        {
            "success": True,
            "data": {
                "total_uploaded_datasets": total_datasets,
                "total_trained_models": total_models,
                "best_r2_score": best_r2,
                "current_rmse": current_rmse,
                "total_predictions_made": total_predictions,
                "latest_training_date": latest_training_date,
                "latest_dataset": _dataset_brief(latest_dataset),
                "best_model": _model_brief(best_model),
                "recent_activity": recent_activity,
            },
        }
    )


@dashboard_bp.get("/charts/<int:dataset_id>")
def dataset_charts(dataset_id: int):
    dataset = UploadedDataset.query.get_or_404(dataset_id)
    df = read_dataset(dataset.filepath)
    payload = build_charts_payload(df, target_column=dataset.target_column)

    # Add best-model chart payload if present
    best_model = TrainedModel.query.filter_by(dataset_id=dataset.id).order_by(TrainedModel.r2_score.desc()).first()
    if best_model and best_model.meta_json:
        meta = json_loads(best_model.meta_json, default={})
        payload["model_charts"] = meta.get("chart_payload") or {}
        payload["best_model"] = _model_brief(best_model)
    else:
        payload["model_charts"] = {}
        payload["best_model"] = None

    return jsonify({"success": True, "data": payload})


@dashboard_bp.get("/insights/<int:model_id>")
def model_insights(model_id: int):
    model = TrainedModel.query.get_or_404(model_id)
    dataset = UploadedDataset.query.get_or_404(model.dataset_id)
    df = read_dataset(dataset.filepath)
    meta = json_loads(model.meta_json, default={})

    model_meta = {
        "performance_label": meta.get("performance_label"),
        "top_factors": meta.get("top_factors"),
    }
    insights = build_insights(df, dataset.target_column or meta.get("target_column"), model_meta=model_meta)
    # Add simple correlation analysis for numeric features vs target (helps explain influence)
    target = insights.get("target_column")
    correlations = []
    if target and target in df.columns:
        try:
            import pandas as pd

            y = pd.to_numeric(df[target], errors="coerce")
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            for c in numeric_cols:
                if c == target:
                    continue
                x = pd.to_numeric(df[c], errors="coerce")
                tmp = pd.concat([x, y], axis=1).dropna()
                if len(tmp) < 20:
                    continue
                corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
                if corr == corr:  # not NaN
                    correlations.append({"feature": c, "corr": corr})
            correlations.sort(key=lambda d: abs(d["corr"]), reverse=True)
            correlations = correlations[:8]
        except Exception:
            correlations = []

    return jsonify(
        {
            "success": True,
            "data": {
                "model": _model_brief(model),
                "dataset": _dataset_brief(dataset),
                "insights": insights,
                "top_factors": meta.get("top_factors") or [],
                "factor_importances": meta.get("factor_importances") or [],
                "correlations": correlations,
            },
        }
    )


def _dataset_brief(d: Optional[UploadedDataset]):
    if not d:
        return None
    return {"id": d.id, "filename": d.filename, "target_column": d.target_column, "uploaded_at": d.uploaded_at.isoformat()}


def _model_brief(m: Optional[TrainedModel]):
    if not m:
        return None
    return {
        "id": m.id,
        "dataset_id": m.dataset_id,
        "model_name": m.model_name,
        "r2_score": m.r2_score,
        "rmse": m.rmse,
        "trained_at": m.trained_at.isoformat() if m.trained_at else None,
    }
