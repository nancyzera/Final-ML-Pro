from typing import Dict, Optional

import numpy as np
import pandas as pd

from services.dataset_service import coerce_datetime_column, infer_column_types


def build_charts_payload(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict:
    numeric_columns, categorical_columns, datetime_columns = infer_column_types(df)

    # pick target
    chosen_target = target_column if target_column in df.columns else _pick_target(numeric_columns)
    if chosen_target is None:
        chosen_target = numeric_columns[0] if numeric_columns else None

    time_col = _pick_time_column(df, datetime_columns)

    payload = {"detected": {"time_column": time_col, "target_column": chosen_target}}

    if chosen_target:
        payload["demand_histogram"] = histogram_series(_col_series(df, chosen_target))
        payload["demand_boxplot"] = boxplot_stats(_col_series(df, chosen_target))

    payload["correlation_heatmap"] = correlation_matrix(df, numeric_columns)

    if time_col and chosen_target:
        payload["demand_over_time"] = time_series(df, time_col, chosen_target)
        payload["cumulative_demand_area"] = cumulative_series(df, time_col, chosen_target)
    elif chosen_target:
        payload["demand_over_time"] = index_time_series(df, chosen_target)
        payload["cumulative_demand_area"] = index_cumulative(df, chosen_target)

    cat_for_bar = _pick_category_column(df, categorical_columns)
    if cat_for_bar and chosen_target and cat_for_bar != chosen_target:
        payload["demand_by_category_bar"] = category_bar(df, cat_for_bar, chosen_target)
        payload["category_share_pie"] = category_pie(df, cat_for_bar)

    temp_col = _pick_temperature_column(df, numeric_columns)
    if temp_col and chosen_target and temp_col != chosen_target:
        payload["temp_vs_demand_scatter"] = scatter_xy(df, temp_col, chosen_target)

    humidity_col = _pick_humidity_column(df, numeric_columns)
    if humidity_col and chosen_target and humidity_col != chosen_target:
        payload["humidity_vs_demand_scatter"] = scatter_xy(df, humidity_col, chosen_target)

    if time_col and chosen_target:
        payload["demand_by_hour_bar"] = demand_by_hour(df, time_col, chosen_target)
        payload["demand_by_weekday_bar"] = demand_by_weekday(df, time_col, chosen_target)
        payload["demand_by_month_bar"] = demand_by_month(df, time_col, chosen_target)

    work_col = _pick_workingday_column(df)
    if work_col and chosen_target and work_col != chosen_target:
        payload["demand_by_workingday_bar"] = category_bar(df, work_col, chosen_target, max_cats=5)

    payload["missing_values_bar"] = missing_values_bar(df, max_cols=15)

    if chosen_target:
        payload["extra_numeric_scatters"] = extra_numeric_scatters(
            df,
            numeric_columns=numeric_columns,
            target_col=chosen_target,
            exclude_cols={chosen_target, temp_col, humidity_col},
            max_charts=10,
        )
        payload["extra_charts"] = extra_target_charts(
            df,
            target_col=chosen_target,
            time_col=time_col,
            numeric_columns=numeric_columns,
            max_points=500,
        )
        # If dataset contains shared/non-shared style demand columns (e.g., casual/registered),
        # add dedicated demand breakdown charts.
        try:
            shared = shared_nonshared_charts(df, time_col=time_col)
            if shared:
                payload["extra_charts"].extend(shared)
        except Exception:
            pass

    return payload


def shared_nonshared_charts(df: pd.DataFrame, time_col: Optional[str]) -> list:
    """
    Add charts when the dataset contains demand breakdown columns, such as:
    - casual vs registered (UCI Bike Sharing)
    - member vs casual / subscriber vs customer
    - shared vs non-shared
    """
    pairs = _detect_demand_breakdown_pairs(df)
    if not pairs:
        return []

    out = []
    for a, b, title_a, title_b in pairs[:2]:  # keep payload bounded
        # pie share
        try:
            sa = pd.to_numeric(_col_series(df, a), errors="coerce").dropna()
            sb = pd.to_numeric(_col_series(df, b), errors="coerce").dropna()
            if sa.empty or sb.empty:
                continue
            out.append(
                {
                    "kind": "pie",
                    "name": f"share_{a}_{b}",
                    "title": f"Demand share: {title_a} vs {title_b}",
                    "labels": [title_a, title_b],
                    "values": [float(sa.sum()), float(sb.sum())],
                }
            )
        except Exception:
            pass

        if time_col:
            # time series for each
            try:
                ts_a = time_series(df, time_col, a, max_points=400)
                ts_b = time_series(df, time_col, b, max_points=400)
                if ts_a.get("labels") and ts_b.get("labels"):
                    out.append(
                        {
                            "kind": "line",
                            "name": f"{a}_over_time",
                            "title": f"{title_a} demand over time",
                            "labels": ts_a["labels"],
                            "values": ts_a["values"],
                            "xTitle": "Time",
                            "yTitle": title_a,
                        }
                    )
                    out.append(
                        {
                            "kind": "line",
                            "name": f"{b}_over_time",
                            "title": f"{title_b} demand over time",
                            "labels": ts_b["labels"],
                            "values": ts_b["values"],
                            "xTitle": "Time",
                            "yTitle": title_b,
                        }
                    )
            except Exception:
                pass

            # by hour for each
            try:
                bh_a = demand_by_hour(df, time_col, a)
                bh_b = demand_by_hour(df, time_col, b)
                if bh_a.get("labels") and bh_b.get("labels"):
                    out.append(
                        {
                            "kind": "bar",
                            "name": f"{a}_by_hour",
                            "title": f"Avg {title_a} demand by hour",
                            "labels": bh_a["labels"],
                            "values": bh_a["values"],
                            "xTitle": "Hour",
                            "yTitle": f"Avg {title_a}",
                        }
                    )
                    out.append(
                        {
                            "kind": "bar",
                            "name": f"{b}_by_hour",
                            "title": f"Avg {title_b} demand by hour",
                            "labels": bh_b["labels"],
                            "values": bh_b["values"],
                            "xTitle": "Hour",
                            "yTitle": f"Avg {title_b}",
                        }
                    )
            except Exception:
                pass

    return out


def _detect_demand_breakdown_pairs(df: pd.DataFrame) -> list:
    cols = {str(c).strip().lower(): c for c in df.columns}

    candidates = []
    # common bike-sharing schema
    if "registered" in cols and "casual" in cols:
        candidates.append((cols["registered"], cols["casual"], "Registered", "Casual"))

    # member vs casual
    for a, b, la, lb in [
        ("member", "casual", "Member", "Casual"),
        ("subscriber", "customer", "Subscriber", "Customer"),
        ("shared", "non_shared", "Shared", "Non-shared"),
        ("shared", "nonshared", "Shared", "Non-shared"),
    ]:
        if a in cols and b in cols:
            candidates.append((cols[a], cols[b], la, lb))

    # heuristic: any two numeric columns containing keywords
    if not candidates:
        keywords_a = ["registered", "member", "subscriber", "shared"]
        keywords_b = ["casual", "customer", "non_shared", "nonshared", "non-shared"]
        a_col = None
        b_col = None
        for k in keywords_a:
            for name, orig in cols.items():
                if k in name:
                    a_col = orig
                    break
            if a_col:
                break
        for k in keywords_b:
            for name, orig in cols.items():
                if k in name:
                    b_col = orig
                    break
            if b_col:
                break
        if a_col and b_col and a_col != b_col:
            candidates.append((a_col, b_col, str(a_col), str(b_col)))

    # Keep only pairs that look numeric-ish
    out = []
    for a, b, la, lb in candidates:
        try:
            sa = pd.to_numeric(_col_series(df, a), errors="coerce")
            sb = pd.to_numeric(_col_series(df, b), errors="coerce")
            if sa.notna().mean() < 0.6 or sb.notna().mean() < 0.6:
                continue
            out.append((a, b, la, lb))
        except Exception:
            continue
    return out


def extra_target_charts(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    numeric_columns,
    max_points: int = 500,
) -> list:
    """
    Generate additional charts that do not depend on having many extra features.
    This is used to reliably exceed 20 charts on the Analytics page across datasets.
    """
    out = []
    y = pd.to_numeric(_col_series(df, target_col), errors="coerce").dropna()
    if y.empty:
        return out

    # Downsample for stable payload size
    y_ds = _downsample(y, max_points=max_points)
    labels = [str(i) for i in y_ds.index.tolist()]
    values = [float(v) for v in y_ds.to_numpy()]

    def add_line(name: str, title: str, vals: list, x_title: str = "Index", y_title: str = target_col):
        out.append(
            {"kind": "line", "name": name, "title": title, "labels": labels, "values": vals, "xTitle": x_title, "yTitle": y_title}
        )

    def add_bar(name: str, title: str, lab: list, vals: list, x_title: str = "Bucket", y_title: str = "Value"):
        out.append(
            {"kind": "bar", "name": name, "title": title, "labels": lab, "values": vals, "xTitle": x_title, "yTitle": y_title}
        )

    def add_scatter(name: str, title: str, points: list, x_label: str, y_label: str):
        out.append({"kind": "scatter", "name": name, "title": title, "points": points, "xLabel": x_label, "yLabel": y_label})

    # Rolling mean / median / std on downsampled series
    s = pd.Series(values)
    for w in (7, 30):
        rm = s.rolling(window=min(w, len(s)), min_periods=max(2, min(5, len(s)))).mean().bfill().tolist()
        add_line(f"rolling_mean_{w}", f"Rolling mean ({w})", [float(v) if v == v else 0.0 for v in rm])
    med = s.rolling(window=min(15, len(s)), min_periods=max(2, min(5, len(s)))).median().bfill().tolist()
    add_line("rolling_median_15", "Rolling median (15)", [float(v) if v == v else 0.0 for v in med])
    st = s.rolling(window=min(15, len(s)), min_periods=max(2, min(5, len(s)))).std().fillna(0.0).tolist()
    add_line("rolling_std_15", "Rolling volatility (std, 15)", [float(v) if v == v else 0.0 for v in st], y_title=f"{target_col} (std)")

    # Lag scatter (y(t-1) vs y(t))
    if len(values) >= 30:
        xs = values[:-1]
        ys = values[1:]
        pts = [{"x": float(a), "y": float(b)} for a, b in zip(xs, ys)]
        if len(pts) > 600:
            pts = list(pd.DataFrame(pts).sample(n=600, random_state=42).to_dict(orient="records"))
        add_scatter("lag1_scatter", f"Lag scatter: {target_col}(t-1) vs {target_col}(t)", pts, f"{target_col}(t-1)", f"{target_col}(t)")

    # First-difference histogram (as bar chart)
    if len(values) >= 20:
        diffs = np.diff(np.array(values, dtype=float))
        hist = histogram_series(pd.Series(diffs), bins=25)
        add_bar("diff_hist", f"Change distribution: Δ {target_col}", hist["labels"], hist["values"], x_title="Δ bin", y_title="Count")

    # Quantiles bar
    q = np.percentile(y.to_numpy(), [5, 10, 25, 50, 75, 90, 95]).tolist()
    q_labels = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
    add_bar("quantiles", f"{target_col} quantiles", q_labels, [float(v) for v in q], x_title="Quantile", y_title=target_col)

    # Top/bottom slices
    topn = min(12, len(y))
    top = y.sort_values(ascending=False).head(topn)
    add_bar("top_values", f"Top {topn} {target_col} values", [str(i) for i in top.index.tolist()], [float(v) for v in top.to_numpy()], x_title="Row", y_title=target_col)
    bot = y.sort_values(ascending=True).head(topn)
    add_bar("bottom_values", f"Bottom {topn} {target_col} values", [str(i) for i in bot.index.tolist()], [float(v) for v in bot.to_numpy()], x_title="Row", y_title=target_col)

    # Correlation-to-target bar (numeric only)
    corrs = []
    try:
        for c in numeric_columns:
            if c == target_col or c not in df.columns:
                continue
            xs = pd.to_numeric(_col_series(df, c), errors="coerce")
            tmp = pd.concat([xs, pd.to_numeric(_col_series(df, target_col), errors="coerce")], axis=1).dropna()
            if tmp.shape[0] < 30:
                continue
            corr = float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))
            if corr == corr:
                corrs.append((abs(corr), corr, c))
        corrs.sort(reverse=True, key=lambda t: t[0])
        corrs = corrs[:12]
        if corrs:
            add_bar(
                "corr_to_target",
                f"Top correlations vs {target_col}",
                [c for _a, _corr, c in corrs],
                [float(corr) for _a, corr, _c in corrs],
                x_title="Feature",
                y_title="Correlation",
            )
    except Exception:
        pass

    # If time exists, add seasonal groupings
    if time_col:
        try:
            tmp = pd.DataFrame({time_col: _col_series(df, time_col), target_col: _col_series(df, target_col)})
            tmp[time_col] = coerce_datetime_column(tmp, time_col)
            tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
            tmp = tmp.dropna()
            if not tmp.empty:
                tmp["day"] = tmp[time_col].dt.day
                grp = tmp.groupby("day")[target_col].mean().sort_index()
                grp = grp.head(31)
                add_bar("by_day_of_month", f"Avg {target_col} by day of month", [str(i) for i in grp.index.tolist()], [float(v) for v in grp.to_numpy()], "Day", f"Avg {target_col}")
                tmp["week"] = tmp[time_col].dt.isocalendar().week.astype(int)
                grp2 = tmp.groupby("week")[target_col].mean().sort_index().head(20)
                add_bar("by_week", f"Avg {target_col} by week (sample)", [str(i) for i in grp2.index.tolist()], [float(v) for v in grp2.to_numpy()], "Week", f"Avg {target_col}")
        except Exception:
            pass

    return out


def histogram_series(series: pd.Series, bins: int = 25) -> Dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"labels": [], "values": []}
    counts, edges = np.histogram(s.to_numpy(), bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": counts.tolist()}


def boxplot_stats(series: pd.Series) -> Dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    q1, med, q3 = np.percentile(s.to_numpy(), [25, 50, 75])
    return {"min": float(np.min(s)), "q1": float(q1), "median": float(med), "q3": float(q3), "max": float(np.max(s))}


def correlation_matrix(df: pd.DataFrame, numeric_columns, max_cols: int = 12) -> Dict:
    cols = [c for c in numeric_columns if c in df.columns][:max_cols]
    if len(cols) < 2:
        return {"labels": [], "matrix": []}
    corr = df[cols].corr(numeric_only=True).fillna(0.0)
    matrix = []
    for i, row in enumerate(cols):
        for j, col in enumerate(cols):
            matrix.append({"x": j, "y": i, "v": float(corr.loc[row, col])})
    return {"labels": cols, "matrix": matrix}


def time_series(df: pd.DataFrame, time_col: str, target_col: str, max_points: int = 500) -> Dict:
    tmp = pd.DataFrame({time_col: _col_series(df, time_col), target_col: _col_series(df, target_col)})
    tmp[time_col] = coerce_datetime_column(tmp, time_col)
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna().sort_values(time_col)
    if tmp.empty:
        return {"labels": [], "values": []}
    # Resample to hourly/daily depending on span
    span_days = (tmp[time_col].max() - tmp[time_col].min()).days if tmp.shape[0] else 0
    if span_days > 60:
        grp = tmp.set_index(time_col)[target_col].resample("D").mean()
    else:
        grp = tmp.set_index(time_col)[target_col].resample("h").mean()
    grp = grp.dropna()
    grp = _downsample(grp, max_points=max_points)
    labels = [d.strftime("%Y-%m-%d %H:%M") for d in grp.index.to_pydatetime()]
    return {"labels": labels, "values": [float(v) for v in grp.to_numpy()]}


def cumulative_series(df: pd.DataFrame, time_col: str, target_col: str, max_points: int = 500) -> Dict:
    ts = time_series(df, time_col, target_col, max_points=max_points)
    values = np.cumsum(np.array(ts.get("values", []), dtype=float)).tolist()
    return {"labels": ts.get("labels", []), "values": values}


def index_time_series(df: pd.DataFrame, target_col: str, max_points: int = 500) -> Dict:
    s = pd.to_numeric(_col_series(df, target_col), errors="coerce").dropna()
    if s.empty:
        return {"labels": [], "values": []}
    s = _downsample(s, max_points=max_points)
    labels = [str(i) for i in s.index.tolist()]
    return {"labels": labels, "values": [float(v) for v in s.to_numpy()]}


def index_cumulative(df: pd.DataFrame, target_col: str, max_points: int = 500) -> Dict:
    ts = index_time_series(df, target_col, max_points=max_points)
    values = np.cumsum(np.array(ts.get("values", []), dtype=float)).tolist()
    return {"labels": ts.get("labels", []), "values": values}


def category_bar(df: pd.DataFrame, cat_col: str, target_col: str, max_cats: int = 20) -> Dict:
    tmp = pd.DataFrame({cat_col: _col_series(df, cat_col), target_col: _col_series(df, target_col)})
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp[cat_col] = tmp[cat_col].astype(str)
    grp = tmp.dropna().groupby(cat_col)[target_col].mean().sort_values(ascending=False).head(max_cats)
    return {"labels": grp.index.tolist(), "values": [float(v) for v in grp.to_numpy()]}


def category_pie(df: pd.DataFrame, cat_col: str, max_cats: int = 8) -> Dict:
    s = _col_series(df, cat_col).dropna().astype(str).value_counts()
    if s.empty:
        return {"labels": [], "values": []}
    top = s.head(max_cats)
    other = int(s.iloc[max_cats:].sum()) if len(s) > max_cats else 0
    labels = top.index.tolist() + (["Other"] if other > 0 else [])
    values = top.to_list() + ([other] if other > 0 else [])
    return {"labels": labels, "values": values}


def scatter_xy(df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 600) -> Dict:
    tmp = pd.DataFrame({x_col: _col_series(df, x_col), y_col: _col_series(df, y_col)})
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return {"points": [], "x_label": x_col, "y_label": y_col}
    tmp = tmp.sample(n=min(max_points, len(tmp)), random_state=42)
    points = [{"x": float(x), "y": float(y)} for x, y in zip(tmp[x_col].to_numpy(), tmp[y_col].to_numpy())]
    return {"points": points, "x_label": x_col, "y_label": y_col}


def _pick_time_column(df: pd.DataFrame, datetime_columns):
    preferred = ["datetime", "date", "timestamp", "time"]
    for p in preferred:
        for c in datetime_columns:
            if p in c.lower():
                return c
    return datetime_columns[0] if datetime_columns else None


def _pick_temperature_column(df: pd.DataFrame, numeric_columns):
    preferred = ["temp", "temperature"]
    for p in preferred:
        for c in numeric_columns:
            if p in c.lower():
                return c
    return numeric_columns[0] if numeric_columns else None


def _pick_humidity_column(df: pd.DataFrame, numeric_columns):
    preferred = ["humidity", "hum"]
    for p in preferred:
        for c in numeric_columns:
            if p == c.lower() or p in c.lower():
                return c
    return None


def _pick_workingday_column(df: pd.DataFrame):
    for c in df.columns:
        cl = str(c).lower()
        if cl in {"workingday", "workday"} or "workingday" in cl:
            return c
    return None


def _pick_category_column(df: pd.DataFrame, categorical_columns):
    # choose a column with manageable cardinality
    candidates = []
    for c in categorical_columns:
        n = df[c].dropna().astype(str).nunique()
        if 2 <= n <= 25:
            candidates.append((n, c))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else (categorical_columns[0] if categorical_columns else None)


def _pick_target(numeric_columns):
    preferred = ["count", "demand", "rides", "cnt", "target", "y"]
    for p in preferred:
        for c in numeric_columns:
            if c.lower() == p or p in c.lower():
                return c
    return None


def _downsample(series: pd.Series, max_points: int):
    if len(series) <= max_points:
        return series
    idx = np.linspace(0, len(series) - 1, num=max_points, dtype=int)
    return series.iloc[idx]


def demand_by_hour(df: pd.DataFrame, time_col: str, target_col: str) -> Dict:
    tmp = pd.DataFrame({time_col: _col_series(df, time_col), target_col: _col_series(df, target_col)})
    tmp[time_col] = coerce_datetime_column(tmp, time_col)
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return {"labels": [], "values": []}
    tmp["hour"] = tmp[time_col].dt.hour
    grp = tmp.groupby("hour")[target_col].mean().reindex(range(24)).fillna(0.0)
    return {"labels": [str(i) for i in grp.index.tolist()], "values": [float(v) for v in grp.to_numpy()]}


def demand_by_weekday(df: pd.DataFrame, time_col: str, target_col: str) -> Dict:
    tmp = pd.DataFrame({time_col: _col_series(df, time_col), target_col: _col_series(df, target_col)})
    tmp[time_col] = coerce_datetime_column(tmp, time_col)
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return {"labels": [], "values": []}
    tmp["weekday"] = tmp[time_col].dt.dayofweek  # 0=Mon
    grp = tmp.groupby("weekday")[target_col].mean().reindex(range(7)).fillna(0.0)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {"labels": labels, "values": [float(v) for v in grp.to_numpy()]}


def demand_by_month(df: pd.DataFrame, time_col: str, target_col: str) -> Dict:
    tmp = pd.DataFrame({time_col: _col_series(df, time_col), target_col: _col_series(df, target_col)})
    tmp[time_col] = coerce_datetime_column(tmp, time_col)
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return {"labels": [], "values": []}
    tmp["month"] = tmp[time_col].dt.month
    grp = tmp.groupby("month")[target_col].mean().reindex(range(1, 13)).fillna(0.0)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return {"labels": labels, "values": [float(v) for v in grp.to_numpy()]}


def missing_values_bar(df: pd.DataFrame, max_cols: int = 15) -> Dict:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(max_cols)
    if miss.empty:
        return {"labels": [], "values": []}
    return {"labels": [str(c) for c in miss.index.tolist()], "values": [int(v) for v in miss.to_numpy()]}


def extra_numeric_scatters(df: pd.DataFrame, numeric_columns, target_col: str, exclude_cols: set, max_charts: int = 10):
    cols = []
    for c in numeric_columns:
        if c in exclude_cols:
            continue
        cols.append(c)
    # prefer columns with decent variance
    scored = []
    for c in cols:
        try:
            s = pd.to_numeric(_col_series(df, c), errors="coerce")
            if s.dropna().shape[0] < 20:
                continue
            var = float(np.nanvar(s.to_numpy(dtype=float)))
            scored.append((var, c))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [c for _v, c in scored[:max_charts]]

    out = []
    for c in chosen:
        payload = scatter_xy(df, c, target_col)
        payload["name"] = f"{c} vs {target_col}"
        payload["x_col"] = c
        payload["y_col"] = target_col
        out.append(payload)
    return out


def _col_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    If a dataframe has duplicate column names, `df[col]` becomes a DataFrame.
    Always return a 1D Series by taking the first matching column.
    """
    obj = df.loc[:, col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj
