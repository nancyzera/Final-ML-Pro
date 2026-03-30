from __future__ import annotations

from typing import Any, Dict, Optional


class GeminiError(RuntimeError):
    pass


def _strip_model_prefix(name: str) -> str:
    name = (name or "").strip()
    if name.startswith("models/"):
        return name.split("models/", 1)[1]
    return name


def list_models(*, api_key: str, timeout_sec: int = 25) -> Dict[str, Any]:
    api_key = (api_key or "").strip()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not configured.")
    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise GeminiError(f"Missing dependency 'requests': {e}")

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        res = requests.get(url, params={"key": api_key}, timeout=timeout_sec, headers={"Accept": "application/json"})
    except Exception as e:
        raise GeminiError(str(e))
    if res.status_code >= 400:
        raise GeminiError(f"Gemini API error (HTTP {res.status_code}).")
    try:
        return res.json()
    except Exception:
        raise GeminiError("Gemini returned a non-JSON response.")


def _supports_generate_content(model_obj: Dict[str, Any]) -> bool:
    methods = model_obj.get("supportedGenerationMethods") or []
    return "generateContent" in methods


def _pick_model_from_list(models_json: Dict[str, Any], preferred: str) -> str:
    preferred = _strip_model_prefix(preferred)
    models = models_json.get("models") or []
    # 1) preferred exact match
    for m in models:
        name = _strip_model_prefix(m.get("name") or "")
        if name == preferred and _supports_generate_content(m):
            return name
    # 2) prefer "flash" for speed/cost
    flash = [m for m in models if _supports_generate_content(m) and "flash" in str(m.get("name", "")).lower()]
    if flash:
        return _strip_model_prefix(flash[0].get("name") or "")
    # 3) any model supporting generateContent
    supported = [m for m in models if _supports_generate_content(m)]
    if supported:
        return _strip_model_prefix(supported[0].get("name") or "")
    # 4) fallback to preferred (even if unsupported)
    return preferred or "gemini-1.5-flash-latest"


def generate_text(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.3,
    timeout_sec: int = 45,
) -> str:
    """
    Calls Google Generative Language API (Gemini) to generate text.
    This project keeps the integration optional and fails gracefully if API key is missing.
    """
    api_key = (api_key or "").strip()
    model = (model or "").strip()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not configured.")
    if not model:
        model = "gemini-1.5-flash"

    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise GeminiError(f"Missing dependency 'requests': {e}")

    model = _strip_model_prefix(model) or "gemini-1.5-flash-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature)},
    }

    try:
        res = requests.post(
            url,
            params=params,
            json=payload,
            timeout=timeout_sec,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
    except Exception as e:
        raise GeminiError(str(e))

    if res.status_code >= 400:
        # Try to surface the API's explanation (invalid key, quota, etc.)
        detail = ""
        try:
            j = res.json()
            err = j.get("error") or {}
            msg = err.get("message") or ""
            status = err.get("status") or ""
            detail = f"{status}: {msg}".strip(": ").strip()
        except Exception:
            detail = (res.text or "").strip()
        if res.status_code == 404 and "NOT_FOUND" in (detail or ""):
            # Resolve a supported model dynamically and retry once.
            try:
                models_json = list_models(api_key=api_key, timeout_sec=min(25, int(timeout_sec)))
                picked = _pick_model_from_list(models_json, model)
                if picked and picked != model:
                    retry_url = f"https://generativelanguage.googleapis.com/v1beta/models/{picked}:generateContent"
                    res2 = requests.post(
                        retry_url,
                        params=params,
                        json=payload,
                        timeout=timeout_sec,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                    if res2.status_code < 400:
                        data = res2.json()
                        candidates = data.get("candidates") or []
                        c0 = candidates[0]
                        content = c0.get("content") or {}
                        parts = content.get("parts") or []
                        text = (parts[0] or {}).get("text") or ""
                        text = str(text).strip()
                        if text:
                            return text
            except Exception:
                pass

        if detail:
            detail = detail.replace("\n", " ").strip()[:400]
            raise GeminiError(f"Gemini API error (HTTP {res.status_code}): {detail}")
        raise GeminiError(f"Gemini API error (HTTP {res.status_code}).")

    data: Optional[Dict[str, Any]] = None
    try:
        data = res.json()
    except Exception:
        raise GeminiError("Gemini returned a non-JSON response.")

    # candidates[0].content.parts[0].text
    try:
        candidates = data.get("candidates") or []
        c0 = candidates[0]
        content = c0.get("content") or {}
        parts = content.get("parts") or []
        text = (parts[0] or {}).get("text") or ""
        text = str(text).strip()
        if not text:
            raise GeminiError("Gemini returned an empty response.")
        return text
    except GeminiError:
        raise
    except Exception:
        raise GeminiError("Could not parse Gemini response.")
