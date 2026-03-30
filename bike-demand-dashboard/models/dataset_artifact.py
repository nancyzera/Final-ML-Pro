from datetime import datetime

from models.database import db


class DatasetArtifact(db.Model):
    __tablename__ = "dataset_artifacts"

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("uploaded_datasets.id"), nullable=False, index=True)

    artifact_type = db.Column(db.String(50), nullable=False)  # original | cleaned | cleaning_report
    path = db.Column(db.String(1024), nullable=True)
    meta_json = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

