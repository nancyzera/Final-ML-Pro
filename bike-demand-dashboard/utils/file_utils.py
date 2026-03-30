import os
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def safe_filename(original_name: str) -> str:
    return secure_filename(original_name or "dataset")


def ensure_within_folder(path: str, folder: str) -> str:
    # Prevent path traversal issues when later serving files
    folder_abs = os.path.abspath(folder)
    path_abs = os.path.abspath(path)
    if not path_abs.startswith(folder_abs + os.sep) and path_abs != folder_abs:
        raise ValueError("Invalid file path.")
    return path_abs

