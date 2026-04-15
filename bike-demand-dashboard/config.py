import os


def _env(key: str, default: str = "") -> str:
    """
    Read environment variable but treat empty strings as missing.
    This allows `.env` to contain keys with empty values without breaking defaults.
    """
    v = os.getenv(key)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


class Config:
    SECRET_KEY = _env("SECRET_KEY", "dev-secret-change-me")

    # SQLite default for local MVP
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DB_PATH = _env("SQLITE_PATH", os.path.join(BASE_DIR, "app.db"))
    SQLALCHEMY_DATABASE_URI = _env("DATABASE_URL", f"sqlite:///{DB_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Storage
    UPLOAD_FOLDER = _env("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
    SAVED_MODELS_FOLDER = _env("SAVED_MODELS_FOLDER", os.path.join(BASE_DIR, "saved_models"))
    EXPORTS_FOLDER = _env("EXPORTS_FOLDER", os.path.join(BASE_DIR, "exports"))

    MAX_CONTENT_LENGTH = int(_env("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))  # 50MB

    # App behavior
    JSON_SORT_KEYS = False
    # Disable aggressive caching for static assets (useful for local dev)
    SEND_FILE_MAX_AGE_DEFAULT = int(_env("SEND_FILE_MAX_AGE_DEFAULT", "0"))

    # Optional AI summaries (Gemini)
    GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
    # Use a "latest" alias by default; the exact set of supported models can vary by account/region.
    GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
