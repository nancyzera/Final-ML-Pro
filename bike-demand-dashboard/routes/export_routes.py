import csv
import io
from datetime import datetime

from flask import Blueprint, Response, jsonify

from models.dataset_model import UploadedDataset
from services.dataset_service import read_dataset
from services.report_service import build_report_context_from_model, generate_html_report
from models.ai_summary import AiSummary

from models.prediction_history import PredictionHistory
from models.trained_model import TrainedModel
from utils.helpers import json_loads


export_bp = Blueprint("export_bp", __name__, url_prefix="/api/export")


@export_bp.get("/predictions")
def export_predictions():
    preds = PredictionHistory.query.order_by(PredictionHistory.predicted_at.desc()).all()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "model_id", "predicted_value", "predicted_at", "input_data_json"])
    for p in preds:
        writer.writerow([p.id, p.model_id, p.predicted_value, p.predicted_at.isoformat(), p.input_data])

    filename = f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@export_bp.get("/models")
def export_models():
    models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "dataset_id", "model_name", "r2_score", "adjusted_r2", "mae", "mse", "rmse", "trained_at"])
    for m in models:
        writer.writerow([m.id, m.dataset_id, m.model_name, m.r2_score, m.adjusted_r2, m.mae, m.mse, m.rmse, m.trained_at.isoformat()])

    filename = f"models_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@export_bp.get("/datasets")
def export_datasets():
    datasets = UploadedDataset.query.order_by(UploadedDataset.uploaded_at.desc()).all()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["id", "filename", "target_column", "rows_count", "columns_count", "missing_values", "uploaded_at"])
    for d in datasets:
        writer.writerow([d.id, d.filename, d.target_column, d.rows_count, d.columns_count, d.missing_values, d.uploaded_at.isoformat()])

    filename = f"datasets_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@export_bp.get("/report/<int:model_id>")
def export_html_report(model_id: int):
    model = TrainedModel.query.get_or_404(model_id)
    dataset = UploadedDataset.query.get_or_404(model.dataset_id)

    df = read_dataset(dataset.filepath)
    ctx = build_report_context_from_model(model, dataset)
    latest_ai = (
        AiSummary.query.filter_by(model_id=model.id, provider="gemini")
        .order_by(AiSummary.created_at.desc())
        .first()
    )
    ai_text = latest_ai.response_text if latest_ai else None

    html = generate_html_report(
        dataset_df=df,
        dataset_dict=ctx["dataset"],
        model_dict=ctx["model"],
        model_meta=ctx["meta"],
        ai_summary_text=ai_text,
    )
    filename = f"bike_demand_report_model_{model.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    return Response(
        html,
        mimetype="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
