from datetime import datetime

from models.database import db


class UploadedDataset(db.Model):
    __tablename__ = "uploaded_datasets"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(1024), nullable=False)
    target_column = db.Column(db.String(255), nullable=True)
    rows_count = db.Column(db.Integer, nullable=False, default=0)
    columns_count = db.Column(db.Integer, nullable=False, default=0)
    missing_values = db.Column(db.Integer, nullable=False, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    trained_models = db.relationship("TrainedModel", back_populates="dataset", cascade="all, delete-orphan")
    artifacts = db.relationship("DatasetArtifact", backref="dataset", cascade="all, delete-orphan")
