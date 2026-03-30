import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402


def main() -> int:
    app = create_app()
    client = app.test_client()

    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    r = client.get("/api/health")
    check("GET /api/health", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get("/api/dashboard/stats")
    check("GET /api/dashboard/stats", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    # UI routes should render HTML (not JSON) and return 200
    for path in [
        "/dashboard",
        "/upload",
        "/preprocess",
        "/train",
        "/performance",
        "/predictions",
        "/analytics",
        "/reports",
        "/settings",
    ]:
        r = client.get(path)
        ok = r.status_code == 200 and (not r.is_json) and (b"<!doctype html" in (r.data or b"")[:200].lower())
        check(f"GET {path}", ok)

    r = client.get("/api/state")
    check("GET /api/state", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    csv_path = Path("example_data/bike_demand_example.csv")
    with csv_path.open("rb") as f:
        data = {"file": (io.BytesIO(f.read()), "bike_demand_example.csv")}
        r = client.post("/api/datasets/upload", data=data, content_type="multipart/form-data")
    check("POST /api/datasets/upload", r.status_code == 200 and r.is_json and r.json.get("success") is True)
    dataset_id = r.json["data"]["dataset"]["id"]

    r = client.get("/api/datasets")
    check("GET /api/datasets", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get(f"/api/datasets/{dataset_id}/summary")
    check("GET /api/datasets/<id>/summary", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get(f"/api/datasets/{dataset_id}/preview")
    check("GET /api/datasets/<id>/preview", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    target = "count"
    r = client.post(f"/api/datasets/{dataset_id}/target", json={"target_column": target})
    check("POST /api/datasets/<id>/target", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.post(f"/api/preprocess/{dataset_id}", json={"target_column": target, "test_size": 0.2, "scale_numeric": True})
    check("POST /api/preprocess/<dataset_id>", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get("/api/models/available")
    check("GET /api/models/available", r.status_code == 200 and r.is_json and r.json.get("success") is True)
    available = (r.json.get("data") or {}).keys() if r.is_json else []

    # Train a few representative models to catch wiring / serialization issues.
    # Keep this short to avoid long runtimes on CI/slow machines.
    candidate_models = [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest Regressor",
    ]
    model_ids = []
    for model_name in candidate_models:
        if model_name not in available:
            check(f"POST /api/train ({model_name})", False, "Model not listed in /api/models/available")
            continue
        r = client.post(
            "/api/train",
            json={
                "dataset_id": dataset_id,
                "target_column": target,
                "model_name": model_name,
                "test_size": 0.2,
                "scale_numeric": True,
            },
        )
        check(f"POST /api/train ({model_name})", r.status_code == 200 and r.is_json and r.json.get("success") is True)
        if r.status_code == 200 and r.is_json and r.json.get("success") is True:
            model_ids.append(r.json["data"]["model"]["id"])

    model_id = model_ids[0]

    r = client.get("/api/models")
    check("GET /api/models", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get(f"/api/models/{model_id}/metrics")
    check("GET /api/models/<id>/metrics", r.status_code == 200 and r.is_json and r.json.get("success") is True)
    meta = r.json["data"].get("meta") or {}

    r = client.get(f"/api/models/{model_id}/diagnostics")
    check("GET /api/models/<id>/diagnostics", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    feature_columns = meta.get("feature_columns") or []
    hints = meta.get("feature_hints") or {}
    inputs = {}
    for col in feature_columns:
        hint = hints.get(col) or {}
        if hint.get("type") == "categorical":
            inputs[col] = (hint.get("values") or [""])[0]
        else:
            inputs[col] = hint.get("mean") if hint.get("mean") is not None else 0

    r = client.post(f"/api/predict/{model_id}", json={"inputs": inputs})
    check("POST /api/predict/<model_id>", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get("/api/predictions")
    check("GET /api/predictions", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get(f"/api/dashboard/charts/{dataset_id}")
    check("GET /api/dashboard/charts/<dataset_id>", r.status_code == 200 and r.is_json and r.json.get("success") is True)
    if r.status_code == 200 and r.is_json and r.json.get("success") is True:
        payload = r.json.get("data") or {}
        # Ensure we have a rich set of charts (base + extended/dynamic)
        extra = payload.get("extra_numeric_scatters") or []
        extra2 = payload.get("extra_charts") or []
        charts_count = len([k for k in payload.keys() if k and k not in {"best_model", "model_charts"}])
        ok = charts_count >= 20 or (charts_count + len(extra) + len(extra2)) >= 20
        check("Charts payload >= 20", ok, f"charts_count={charts_count} extra={len(extra)} extra2={len(extra2)}")

    r = client.get(f"/api/dashboard/insights/{model_id}")
    check("GET /api/dashboard/insights/<model_id>", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    r = client.get("/api/export/predictions")
    check("GET /api/export/predictions", r.status_code == 200 and r.headers.get("Content-Type", "").startswith("text/csv"))

    r = client.get("/api/export/models")
    check("GET /api/export/models", r.status_code == 200 and r.headers.get("Content-Type", "").startswith("text/csv"))

    r = client.get("/api/export/datasets")
    check("GET /api/export/datasets", r.status_code == 200 and r.headers.get("Content-Type", "").startswith("text/csv"))

    for mid in model_ids[:2] if model_ids else [model_id]:
        r = client.get(f"/api/export/report/{mid}")
        check(
            f"GET /api/export/report/<model_id> ({mid})",
            r.status_code == 200 and r.headers.get("Content-Type", "").startswith("text/html"),
        )

    r = client.delete(f"/api/datasets/{dataset_id}")
    check("DELETE /api/datasets/<id>", r.status_code == 200 and r.is_json and r.json.get("success") is True)

    failed = [x for x in results if not x[1]]
    for name, ok, _detail in results:
        print(f"- {'OK ' if ok else 'FAIL'} {name}")
    print(f"TOTAL={len(results)} OK={len(results)-len(failed)} FAIL={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
