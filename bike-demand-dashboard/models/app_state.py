from models.database import db


class AppState(db.Model):
    __tablename__ = "app_state"

    id = db.Column(db.Integer, primary_key=True)  # always 1
    active_dataset_id = db.Column(db.Integer, nullable=True)
    active_model_id = db.Column(db.Integer, nullable=True)

