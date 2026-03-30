import os
import time

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from dotenv import load_dotenv

from models.database import db
from routes.dataset_routes import dataset_bp
from routes.preprocess_routes import preprocess_bp
from routes.training_routes import training_bp
from routes.prediction_routes import prediction_bp
from routes.dashboard_routes import dashboard_bp
from routes.export_routes import export_bp
from routes.state_routes import state_bp
from routes.ai_routes import ai_bp


def create_app():
    # Load .env BEFORE importing config so env-based config values are present.
    load_dotenv(override=True)

    # Import config after dotenv load (Config reads env vars at import time).
    import importlib
    import config as config_module

    importlib.reload(config_module)
    Config = config_module.Config

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(Config)
    # Cache-busting for static assets during local development
    app.config["APP_VERSION"] = os.getenv("APP_VERSION") or str(int(time.time()))

    # Ensure folders exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["SAVED_MODELS_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EXPORTS_FOLDER"], exist_ok=True)

    CORS(app)
    db.init_app(app)

    # Ensure all models are imported before create_all()
    from models import dataset_model  # noqa: F401
    from models import trained_model  # noqa: F401
    from models import prediction_history  # noqa: F401
    from models import app_state  # noqa: F401
    from models import dataset_artifact  # noqa: F401
    from models import ai_summary  # noqa: F401

    with app.app_context():
        db.create_all()

    # API blueprints
    app.register_blueprint(dataset_bp)
    app.register_blueprint(preprocess_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(state_bp)
    app.register_blueprint(ai_bp)

    @app.get("/api/health")
    def health():
        return {"success": True, "data": {"status": "ok"}}

    @app.errorhandler(404)
    def handle_404(err):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "message": "API route not found."}), 404
        return err

    @app.errorhandler(405)
    def handle_405(err):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "message": "Method not allowed."}), 405
        return err

    @app.errorhandler(500)
    def handle_500(err):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "message": "Internal server error."}), 500
        return err

    # UI routes
    @app.get("/")
    def home():
        return redirect(url_for("ui_dashboard"))

    @app.get("/dashboard")
    def ui_dashboard():
        return render_template("dashboard.html", active="Dashboard")

    @app.get("/upload")
    def ui_upload():
        return render_template("upload.html", active="Upload Dataset")

    @app.get("/preprocess")
    def ui_preprocess():
        return render_template("preprocess.html", active="Data Preprocessing")

    @app.get("/train")
    def ui_train():
        return render_template("train.html", active="Model Training")

    @app.get("/performance")
    def ui_performance():
        return render_template("performance.html", active="Model Performance")

    @app.get("/predictions")
    def ui_predictions():
        return render_template("predictions.html", active="Predictions")

    @app.get("/analytics")
    def ui_analytics():
        return render_template("analytics.html", active="Analytics")

    @app.get("/reports")
    def ui_reports():
        return render_template("reports.html", active="Reports")

    @app.get("/settings")
    def ui_settings():
        return render_template("settings.html", active="Settings")

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "127.0.0.1")
    app.run(host=host, port=int(os.getenv("PORT", "5000")), debug=True)
