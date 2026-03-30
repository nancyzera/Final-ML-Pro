from flask import Blueprint, jsonify, request

from models.app_state import AppState
from models.database import db


state_bp = Blueprint("state_bp", __name__, url_prefix="/api/state")


def _get_state() -> AppState:
    state = AppState.query.get(1)
    if not state:
        state = AppState(id=1, active_dataset_id=None, active_model_id=None)
        db.session.add(state)
        db.session.commit()
    return state


@state_bp.get("")
def get_state():
    state = _get_state()
    return jsonify(
        {
            "success": True,
            "data": {
                "active_dataset_id": state.active_dataset_id,
                "active_model_id": state.active_model_id,
            },
        }
    )


@state_bp.post("")
def set_state():
    body = request.get_json(silent=True) or {}
    state = _get_state()

    if "active_dataset_id" in body:
        state.active_dataset_id = body.get("active_dataset_id")
    if "active_model_id" in body:
        state.active_model_id = body.get("active_model_id")

    db.session.commit()
    return jsonify(
        {
            "success": True,
            "message": "State updated.",
            "data": {"active_dataset_id": state.active_dataset_id, "active_model_id": state.active_model_id},
        }
    )

