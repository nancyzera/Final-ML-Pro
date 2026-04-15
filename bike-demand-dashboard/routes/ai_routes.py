from datetime import datetime

from flask import Blueprint, current_app, jsonify, request

from models.ai_summary import AiSummary
from models.database import db
from models.dataset_model import UploadedDataset
from models.trained_model import TrainedModel
from services.ai_summary_service import build_ai_context, build_local_summary, build_prompt, is_low_signal_response
from services.gemini_service import GeminiError, generate_text
from services.gemini_service import list_models as gemini_list_models


ai_bp = Blueprint("ai_bp", __name__, url_prefix="/api/ai")


@ai_bp.get("/status")
def status():
    enabled = True
    has_api = bool(current_app.config.get("GEMINI_API_KEY"))
    return jsonify(
        {
            "success": True,
            "data": {
                "enabled": enabled,
                "provider": "gemini" if has_api else "local",
                "model": current_app.config.get("GEMINI_MODEL") or "gemini-2.5-flash",
                "fallback_mode": not has_api,
            },
        }
    )


@ai_bp.get("/models")
def models():
    """
    List Gemini models available to this API key (filtered to those supporting generateContent).
    """
    api_key = current_app.config.get("GEMINI_API_KEY") or ""
    if not api_key:
        return jsonify({"success": False, "message": "Gemini is not configured. Set GEMINI_API_KEY in .env."}), 400
    try:
        j = gemini_list_models(api_key=api_key)
        out = []
        for m in (j.get("models") or []):
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            out.append(
                {
                    "name": m.get("name"),
                    "displayName": m.get("displayName"),
                    "description": m.get("description"),
                    "methods": methods,
                }
            )
        return jsonify({"success": True, "data": out})
    except GeminiError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@ai_bp.post("/summary/<int:model_id>")
def summarize_model(model_id: int):
    """
    Generate (or re-generate) a narrative model-performance summary and factor explanation using Gemini.
    """
    model = TrainedModel.query.get_or_404(model_id)
    dataset = UploadedDataset.query.get_or_404(model.dataset_id)
    body = request.get_json(silent=True) or {}
    locale_context = (body.get("context") or "").strip()
    force = bool(body.get("force", False))

    ctx = build_ai_context(dataset_row=dataset, model_row=model)
    prompt = build_prompt(ctx=ctx, locale_context=locale_context or "Kigali, Rwanda")

    # Return cached summary unless forced, but only when it matches this prompt and is not low-signal.
    if not force:
        cached = (
            AiSummary.query.filter_by(model_id=model.id, provider="gemini", prompt=prompt)
            .order_by(AiSummary.created_at.desc())
            .first()
        )
        if cached:
            cached_text = str(cached.response_text or "").strip()
            if is_low_signal_response(cached_text):
                db.session.delete(cached)
                db.session.commit()
            else:
                return jsonify(
                    {
                        "success": True,
                        "message": "AI summary loaded.",
                        "data": {"summary_text": cached.response_text, "created_at": cached.created_at.isoformat()},
                    }
                )

    api_key = current_app.config.get("GEMINI_API_KEY") or ""
    gem_model = current_app.config.get("GEMINI_MODEL") or "gemini-2.5-flash"
    local_summary = build_local_summary(ctx=ctx, locale_context=locale_context or "Kigali, Rwanda")

    if not api_key:
        row = AiSummary(model_id=model.id, provider="gemini", model_name="local-summary", prompt=prompt, response_text=local_summary)
        db.session.add(row)
        db.session.commit()
        return jsonify(
            {
                "success": True,
                "message": "Local summary generated.",
                "data": {"summary_text": local_summary, "created_at": row.created_at.isoformat()},
            }
        )

    try:
        text = generate_text(api_key=api_key, model=gem_model, prompt=prompt)
        if is_low_signal_response(text):
            text = local_summary
        row = AiSummary(model_id=model.id, provider="gemini", model_name=gem_model, prompt=prompt, response_text=text)
        db.session.add(row)
        db.session.commit()
        return jsonify(
            {
                "success": True,
                "message": "AI summary generated.",
                "data": {"summary_text": text, "created_at": row.created_at.isoformat()},
            }
        )
    except GeminiError as e:
        row = AiSummary(model_id=model.id, provider="gemini", model_name=f"{gem_model} (fallback)", prompt=prompt, response_text=local_summary)
        db.session.add(row)
        db.session.commit()
        return jsonify(
            {
                "success": True,
                "message": f"Gemini was unavailable, so a local summary was generated instead. {e}",
                "data": {"summary_text": local_summary, "created_at": row.created_at.isoformat()},
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
