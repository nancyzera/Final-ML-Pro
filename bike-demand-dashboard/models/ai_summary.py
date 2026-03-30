from datetime import datetime

from models.database import db


class AiSummary(db.Model):
    __tablename__ = "ai_summaries"

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey("trained_models.id"), nullable=False, index=True)

    provider = db.Column(db.String(50), nullable=False, default="gemini")
    model_name = db.Column(db.String(120), nullable=True)

    prompt = db.Column(db.Text, nullable=False)
    response_text = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

