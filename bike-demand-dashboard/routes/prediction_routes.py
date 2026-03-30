from flask import Blueprint, jsonify, request

from models.database import db
from models.prediction_history import PredictionHistory
from models.trained_model import TrainedModel
from services.prediction_service import predict_from_inputs
from utils.helpers import json_dumps, json_loads


prediction_bp = Blueprint("prediction_bp", __name__, url_prefix="/api")


@prediction_bp.post("/predict/<int:model_id>")
def predict(model_id: int):
    model = TrainedModel.query.get_or_404(model_id)
    body = request.get_json(silent=True) or {}
    inputs = body.get("inputs") or {}

    meta = json_loads(model.meta_json, default={})
    feature_columns = meta.get("feature_columns") or []
    if not feature_columns:
        return jsonify({"success": False, "message": "Model metadata missing feature columns."}), 500

    try:
        predicted_value, _details = predict_from_inputs(model.model_path, inputs, feature_columns=feature_columns)
        record = PredictionHistory(model_id=model.id, input_data=json_dumps(inputs), predicted_value=float(predicted_value))
        db.session.add(record)
        db.session.commit()
        return jsonify(
            {
                "success": True,
                "message": "Prediction generated.",
                "data": {"predicted_value": float(predicted_value), "prediction": _prediction_to_dict(record)},
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400


@prediction_bp.get("/predictions")
def list_predictions():
    preds = PredictionHistory.query.order_by(PredictionHistory.predicted_at.desc()).limit(200).all()
    return jsonify({"success": True, "data": [_prediction_to_dict(p) for p in preds]})


def _prediction_to_dict(p: PredictionHistory):
    return {
        "id": p.id,
        "model_id": p.model_id,
        "input_data": json_loads(p.input_data, default={}),
        "predicted_value": p.predicted_value,
        "predicted_at": p.predicted_at.isoformat() if p.predicted_at else None,
    }

