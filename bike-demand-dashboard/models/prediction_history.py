from datetime import datetime

from models.database import db


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey("trained_models.id"), nullable=False, index=True)

    input_data = db.Column(db.Text, nullable=False)  # JSON string
    predicted_value = db.Column(db.Float, nullable=False)
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    model = db.relationship("TrainedModel", back_populates="predictions")

