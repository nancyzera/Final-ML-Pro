import json
from datetime import datetime
from typing import Any, Dict


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def json_loads(text: str, default=None):
    if not text:
        return default if default is not None else {}
    try:
        return json.loads(text)
    except Exception:
        return default if default is not None else {}


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def pick_first(items):
    for item in items:
        if item:
            return item
    return None


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

