import os
import base64
import io
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from services.chart_service import build_charts_payload
from services.dataset_service import infer_column_types
from services.insight_service import build_insights
from services.training_service import load_pipeline
from utils.helpers import json_loads


def generate_html_report(
    dataset_df: pd.DataFrame,
    dataset_dict: Dict[str, Any],
    model_dict: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
    ai_summary_text: Optional[str] = None,
) -> str:
    charts = build_charts_payload(dataset_df, target_column=dataset_dict.get("target_column"))

    # Model charts (if available)
    model_charts = {}
    top_factors: List[str] = []
    factor_importances = []
    if model_dict and model_meta:
        model_charts = (model_meta.get("chart_payload") or {}) if isinstance(model_meta, dict) else {}
        top_factors = model_meta.get("top_factors") or []
        factor_importances = model_meta.get("factor_importances") or []

    target = charts.get("detected", {}).get("target_column")
    numeric_cols, categorical_cols, _dt_cols = infer_column_types(dataset_df)

    correlations = []
    if target and target in dataset_df.columns:
        y = pd.to_numeric(dataset_df[target], errors="coerce")
        for col in numeric_cols:
            if col == target:
                continue
            x = pd.to_numeric(dataset_df[col], errors="coerce")
            tmp = pd.concat([x, y], axis=1).dropna()
            if len(tmp) < 20:
                continue
            corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
            if not np.isnan(corr):
                correlations.append((col, corr))
        correlations.sort(key=lambda t: abs(t[1]), reverse=True)
        correlations = correlations[:8]

    # Build report images
    figures = []
    figures.append(("Demand over time", _fig_line(charts.get("demand_over_time"))))
    figures.append(("Demand by category", _fig_bar(charts.get("demand_by_category_bar"))))
    figures.append(("Temperature vs demand", _fig_scatter(charts.get("temp_vs_demand_scatter"))))
    figures.append(("Demand distribution", _fig_hist(charts.get("demand_histogram"))))
    figures.append(("Outliers (box plot)", _fig_box(charts.get("demand_boxplot"))))
    figures.append(("Correlation heatmap", _fig_heatmap(dataset_df, numeric_cols[:12])))
    figures.append(("Category share", _fig_pie(charts.get("category_share_pie"))))
    figures.append(("Cumulative demand trend", _fig_line(charts.get("cumulative_demand_area"), fill=True)))

    if model_charts:
        figures.append(("Actual vs Predicted", _fig_actual_pred(model_charts)))
        figures.append(("Residual distribution", _fig_residuals(model_charts)))

    # Insights
    insights = build_insights(dataset_df, dataset_dict.get("target_column"), model_meta={"top_factors": top_factors, "performance_label": (model_meta or {}).get("performance_label")})
    insight_lines = insights.get("insights") or []

    metrics_html = ""
    if model_dict:
        metrics_html = f"""
        <div class="grid">
          <div class="kpi"><div class="k">R²</div><div class="v">{model_dict.get('r2_score', 0):.3f}</div></div>
          <div class="kpi"><div class="k">Adj R²</div><div class="v">{model_dict.get('adjusted_r2', 0):.3f}</div></div>
          <div class="kpi"><div class="k">MAE</div><div class="v">{model_dict.get('mae', 0):.3f}</div></div>
          <div class="kpi"><div class="k">MSE</div><div class="v">{model_dict.get('mse', 0):.3f}</div></div>
          <div class="kpi"><div class="k">RMSE</div><div class="v">{model_dict.get('rmse', 0):.3f}</div></div>
        </div>
        """

    corr_html = ""
    if correlations:
        rows = "".join([f"<tr><td>{_esc(col)}</td><td>{corr:+.2f}</td></tr>" for col, corr in correlations])
        corr_html = f"""
        <h3>Top Correlations (numeric)</h3>
        <p class="muted">Correlation is a quick signal (not causation). Higher absolute values indicate stronger linear association with demand.</p>
        <table class="table">
          <thead><tr><th>Feature</th><th>corr(feature, demand)</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    factors_html = ""
    if factor_importances:
        rows = "".join([f"<tr><td>{_esc(i.get('feature'))}</td><td>{float(i.get('importance',0))*100:.1f}%</td></tr>" for i in factor_importances[:12]])
        factors_html = f"""
        <h3>Top Influencing Factors (model-based)</h3>
        <p class="muted">These factors come from the trained model (tree importances or linear coefficients) and are normalized for readability.</p>
        <table class="table">
          <thead><tr><th>Feature</th><th>Relative importance</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
    elif top_factors:
        rows = "".join([f"<tr><td>{i+1}</td><td>{_esc(f)}</td></tr>" for i, f in enumerate(top_factors[:12])])
        factors_html = f"""
        <h3>Top Influencing Factors (model-based)</h3>
        <p class="muted">Top drivers extracted from the trained model metadata.</p>
        <table class="table">
          <thead><tr><th>#</th><th>Factor</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    figs_html = ""
    for title, fig in figures:
        if fig is None:
            continue
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        figs_html += f"""
        <div class="card">
          <div class="card-h">{_esc(title)}</div>
          <img class="img" src="data:image/png;base64,{b64}" alt="{_esc(title)}"/>
          <div class="card-p muted">{_explain_chart(title, target)}</div>
        </div>
        """

    step_html = """
    <div class="steps">
      <div class="step"><div class="n">1</div><div><div class="t">Upload dataset</div><div class="m">CSV/Excel → preview and select target.</div></div></div>
      <div class="step"><div class="n">2</div><div><div class="t">Preprocess</div><div class="m">Missing values, encoding, scaling, train/test split.</div></div></div>
      <div class="step"><div class="n">3</div><div><div class="t">Train model</div><div class="m">Try multiple regressors and compare metrics.</div></div></div>
      <div class="step"><div class="n">4</div><div><div class="t">Evaluate</div><div class="m">R², MAE, MSE, RMSE + residual analysis.</div></div></div>
      <div class="step"><div class="n">5</div><div><div class="t">Predict</div><div class="m">Enter features → demand prediction + history.</div></div></div>
      <div class="step"><div class="n">6</div><div><div class="t">Report</div><div class="m">Download this report with charts + explanations.</div></div></div>
    </div>
    """

    if insight_lines:
        insight_html = "".join([f"<div class='ins'>{_esc(x)}</div>" for x in insight_lines])
        insight_html = f"<div class='ins-wrap'>{insight_html}</div>"
    else:
        insight_html = "<div class='muted'>No insights yet.</div>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bike Demand Report</title>
  <style>
    :root{{--primary:#4f46e5;}}
    body{{font-family:Inter, Arial, sans-serif; background:#f6f7ff; color:#0f172a; margin:0;}}
    .wrap{{max-width:1100px; margin:0 auto; padding:24px;}}
    .hero{{background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:16px; padding:18px 18px;}}
    .title{{font-size:22px; font-weight:800; margin:0;}}
    .muted{{color:rgba(15,23,42,0.62);}}
    .meta{{display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; font-size:13px;}}
    .pill{{background:#f1f5f9; border:1px solid rgba(2,6,23,0.08); padding:6px 10px; border-radius:999px;}}
    .grid{{display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:14px;}}
    .kpi{{background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:14px; padding:10px;}}
    .k{{font-size:12px; color:rgba(15,23,42,0.62);}}
    .v{{font-size:18px; font-weight:800; margin-top:4px;}}
    h3{{margin:18px 0 8px;}}
    .card{{background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:16px; padding:14px; margin-top:12px;}}
    .card-h{{font-weight:800; margin-bottom:10px;}}
    .img{{width:100%; border-radius:12px; border:1px solid rgba(2,6,23,0.08);}}
    .card-p{{margin-top:10px; font-size:13px; line-height:1.45;}}
    .table{{width:100%; border-collapse:collapse; font-size:13px;}}
    .table th,.table td{{border-bottom:1px solid rgba(2,6,23,0.08); padding:8px; text-align:left;}}
    .steps{{display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px;}}
    .step{{background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:14px; padding:10px; display:flex; gap:10px;}}
    .step .n{{width:28px; height:28px; border-radius:10px; display:grid; place-items:center; background:var(--primary); color:#fff; font-weight:900;}}
    .step .t{{font-weight:800; font-size:13px;}}
    .step .m{{font-size:12px; color:rgba(15,23,42,0.62); margin-top:2px;}}
    .ins-wrap{{display:grid; gap:8px; margin-top:12px;}}
    .ins{{background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:14px; padding:10px 12px; font-size:13px; line-height:1.45;}}
    .ai{{white-space:pre-wrap; background:#fff; border:1px solid rgba(2,6,23,0.10); border-radius:16px; padding:14px 16px; margin-top:10px; line-height:1.55;}}
    @media (max-width: 980px){{.grid{{grid-template-columns:repeat(2,1fr);}} .steps{{grid-template-columns:1fr;}}}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="title">Bike Demand Prediction Report</div>
      <div class="meta">
        <div class="pill">Dataset: <b>{_esc(dataset_dict.get('filename',''))}</b></div>
        <div class="pill">Target: <b>{_esc(dataset_dict.get('target_column') or target or '')}</b></div>
        <div class="pill">Rows: <b>{int(dataset_dict.get('rows_count') or dataset_df.shape[0])}</b></div>
        <div class="pill">Columns: <b>{int(dataset_dict.get('columns_count') or dataset_df.shape[1])}</b></div>
        {f"<div class='pill'>Model: <b>{_esc(model_dict.get('model_name',''))}</b></div>" if model_dict else ""}
      </div>
      {metrics_html}
    </div>

    {f"<h3>AI Summary</h3><div class='muted'>Generated using Gemini. Review critically and validate with domain context.</div><div class='ai'>{_esc(ai_summary_text)}</div>" if ai_summary_text else ""}

    <h3>Workflow</h3>
    <p class="muted">Follow these steps to build a reliable demand predictor.</p>
    {step_html}

    <h3>Key Insights</h3>
    {insight_html}

    {factors_html}
    {corr_html}

    <h3>Analytics Charts</h3>
    <p class="muted">Each chart below is generated from your dataset and includes a short explanation.</p>
    {figs_html}
  </div>
</body>
</html>
"""
    return html


def build_report_context_from_model(model_row, dataset_row) -> Dict[str, Any]:
    meta = json_loads(getattr(model_row, "meta_json", None), default={})
    dataset_dict = {
        "id": dataset_row.id,
        "filename": dataset_row.filename,
        "target_column": dataset_row.target_column or meta.get("target_column"),
        "rows_count": dataset_row.rows_count,
        "columns_count": dataset_row.columns_count,
        "missing_values": dataset_row.missing_values,
        "uploaded_at": dataset_row.uploaded_at.isoformat() if dataset_row.uploaded_at else None,
    }
    model_dict = {
        "id": model_row.id,
        "dataset_id": model_row.dataset_id,
        "model_name": model_row.model_name,
        "r2_score": model_row.r2_score,
        "adjusted_r2": model_row.adjusted_r2,
        "mae": model_row.mae,
        "mse": model_row.mse,
        "rmse": model_row.rmse,
        "trained_at": model_row.trained_at.isoformat() if model_row.trained_at else None,
    }
    return {"dataset": dataset_dict, "model": model_dict, "meta": meta}


def _fig_to_base64(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("ascii")


def _fig_line(payload: Optional[Dict[str, Any]], fill: bool = False):
    if not payload:
        return None
    labels = payload.get("labels") or []
    values = payload.get("values") or []
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(values, color="#4f46e5", linewidth=2)
    if fill:
        ax.fill_between(range(len(values)), values, color="#4f46e5", alpha=0.12)
    ax.set_title("Demand trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand")
    ax.grid(True, alpha=0.18)
    ax.set_xticks(np.linspace(0, len(labels) - 1, num=min(8, len(labels))).astype(int))
    ax.set_xticklabels([labels[i] for i in ax.get_xticks().astype(int)], rotation=0, fontsize=8)
    return fig


def _fig_bar(payload: Optional[Dict[str, Any]]):
    if not payload:
        return None
    labels = payload.get("labels") or []
    values = payload.get("values") or []
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(labels, values, color="#4f46e5", alpha=0.85)
    ax.set_title("Average demand by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Avg demand")
    ax.grid(True, axis="y", alpha=0.18)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    return fig


def _fig_scatter(payload: Optional[Dict[str, Any]]):
    if not payload:
        return None
    points = payload.get("points") or []
    if not points:
        return None
    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.scatter(x, y, s=14, alpha=0.55, color="#4f46e5")
    ax.set_title("Feature vs demand")
    ax.set_xlabel(payload.get("x_label") or "Feature")
    ax.set_ylabel(payload.get("y_label") or "Demand")
    ax.grid(True, alpha=0.18)
    return fig


def _fig_hist(payload: Optional[Dict[str, Any]]):
    if not payload:
        return None
    labels = payload.get("labels") or []
    values = payload.get("values") or []
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(len(values)), values, color="#4f46e5", alpha=0.85)
    ax.set_title("Demand distribution (histogram)")
    ax.set_xlabel("Bins")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.18)
    return fig


def _fig_box(payload: Optional[Dict[str, Any]]):
    if not payload or payload.get("median") is None:
        return None
    data = [[payload["min"], payload["q1"], payload["median"], payload["q3"], payload["max"]]]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    # matplotlib bxp wants dict format
    ax.bxp(
        [
            {
                "med": payload["median"],
                "q1": payload["q1"],
                "q3": payload["q3"],
                "whislo": payload["min"],
                "whishi": payload["max"],
                "fliers": [],
            }
        ],
        showfliers=False,
    )
    ax.set_title("Demand outliers (box plot)")
    ax.set_xticks([])
    ax.set_ylabel("Demand")
    ax.grid(True, axis="y", alpha=0.18)
    return fig


def _fig_heatmap(df: pd.DataFrame, cols: List[str]):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    sns.heatmap(corr, ax=ax, cmap="Blues", vmin=-1, vmax=1, center=0, linewidths=0.5)
    ax.set_title("Correlation heatmap")
    return fig


def _fig_pie(payload: Optional[Dict[str, Any]]):
    if not payload:
        return None
    labels = payload.get("labels") or []
    values = payload.get("values") or []
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.pie(values, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("Category share")
    return fig


def _fig_actual_pred(model_charts: Dict[str, Any]):
    y_true = model_charts.get("y_true") or []
    y_pred = model_charts.get("y_pred") or []
    if not y_true or not y_pred:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.scatter(y_true, y_pred, s=18, alpha=0.65, color="#4f46e5")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.18)
    return fig


def _fig_residuals(model_charts: Dict[str, Any]):
    residuals = model_charts.get("residuals") or []
    if not residuals:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.hist(residuals, bins=18, color="#4f46e5", alpha=0.85)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.18)
    return fig


def _explain_chart(title: str, target: Optional[str]) -> str:
    t = (target or "demand")
    if "Demand over time" in title:
        return f"Shows how {t} changes over time. Use it to spot peaks, seasonality, and trend shifts."
    if "Demand by category" in title:
        return f"Compares average {t} across a selected category (station/season/weather). High bars indicate higher typical demand."
    if "Temperature" in title:
        return f"Shows the relationship between temperature and {t}. A strong upward/downward pattern suggests weather influence."
    if "distribution" in title.lower():
        return f"Shows how {t} values are distributed. Skewed distributions may benefit from transformations or robust models."
    if "Outliers" in title:
        return f"Highlights extreme {t} values. Outliers can dominate error metrics and may require cleaning or robust modeling."
    if "Correlation heatmap" in title:
        return f"Shows correlations between numeric features. Use it to detect redundant features and understand relationships with {t}."
    if "Category share" in title:
        return "Shows how frequent each category is in the dataset. Rare categories can be harder for models to learn."
    if "Cumulative" in title:
        return f"Shows cumulative {t} over time for a big-picture view of growth and sustained demand."
    if "Actual vs Predicted" in title:
        return "Points closer to the diagonal mean better predictions. Large deviations indicate under/over prediction patterns."
    if "Residual" in title:
        return "Residuals should be centered near 0. Wide or skewed residuals suggest missing features or non-linear patterns."
    return "Chart generated from your dataset to support understanding and modeling decisions."


def _esc(text: Any) -> str:
    s = "" if text is None else str(text)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )
