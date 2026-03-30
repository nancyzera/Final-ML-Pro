from datetime import datetime

from models.database import db


class TrainedModel(db.Model):
    __tablename__ = "trained_models"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("uploaded_datasets.id"), nullable=False, index=True)

    model_name = db.Column(db.String(255), nullable=False)
    r2_score = db.Column(db.Float, nullable=False, default=0.0)
    adjusted_r2 = db.Column(db.Float, nullable=False, default=0.0)
    mae = db.Column(db.Float, nullable=False, default=0.0)
    mse = db.Column(db.Float, nullable=False, default=0.0)
    rmse = db.Column(db.Float, nullable=False, default=0.0)

    model_path = db.Column(db.String(1024), nullable=False)
    meta_json = db.Column(db.Text, nullable=True)  # stores feature schema, samples, etc.

    trained_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    dataset = db.relationship("UploadedDataset", back_populates="trained_models")
    predictions = db.relationship("PredictionHistory", back_populates="model", cascade="all, delete-orphan")

