let charts = {};
let fullscreenChart = null;
let fullscreenTargetId = null;

function byId(id) {
  return document.getElementById(id);
}

function getChartCanvas(id) {
  const el = byId(id);
  if (!el) {
    console.warn(`Missing chart canvas: ${id}`);
    return null;
  }
  return el;
}

function chartTitle(title, summary) {
  return summary ? [title, summary] : title;
}

function chartAxisLabel(label) {
  return Array.isArray(label) ? (label[0] || '') : label;
}

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  return {
    primary: (css.getPropertyValue('--primary') || '').trim() || '#4f46e5',
    primaryRgb: (css.getPropertyValue('--primary-rgb') || '').trim() || '79,70,229',
  };
}

function withAlpha(color, alpha) {
  const c = String(color || '').trim();
  if (c.startsWith('#') && (c.length === 7 || c.length === 4)) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split('').map(ch => ch + ch).join('');
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return c;
}

function palette(n) {
  const { primary } = themeColors();
  const base = [primary, '#7453ff', '#1dd3b0', '#f59e0b', '#f43f5e', '#22c55e', '#38bdf8', '#a78bfa', '#fb7185', '#f97316'];
  const out = [];
  for (let i = 0; i < n; i++) out.push(base[i % base.length]);
  return out;
}

function destroyAll() {
  Object.values(charts).forEach(c => window.chartDestroy(c));
  charts = {};
}

function cloneChartConfig(chart) {
  return {
    type: chart.config.type,
    data: {
      labels: Array.isArray(chart.data?.labels) ? [...chart.data.labels] : chart.data?.labels,
      datasets: (chart.data?.datasets || []).map(ds => ({
        ...ds,
        data: Array.isArray(ds.data) ? ds.data.map(point => (
          point && typeof point === 'object' ? { ...point } : point
        )) : ds.data,
      })),
    },
    options: JSON.parse(JSON.stringify(chart.options || {})),
  };
}

function setupFullscreenButtons() {
  const modalEl = byId('analyticsFullscreenModal');
  if (!modalEl || typeof bootstrap === 'undefined') return;
  const modal = new bootstrap.Modal(modalEl);
  document.addEventListener('click', (event) => {
    const btn = event.target.closest('[data-chart-target]');
    if (!btn) return;
    const target = btn.getAttribute('data-chart-target');
    const source = charts[target];
    if (!source) return;
    fullscreenTargetId = target;
    const title = btn.closest('.card-body')?.querySelector('.h6')?.textContent || 'Chart';
    const titleEl = byId('analyticsFullscreenTitle');
    if (titleEl) titleEl.textContent = title;
    modal.show();
  });
  modalEl.addEventListener('shown.bs.modal', () => {
    if (!fullscreenTargetId) return;
    const source = charts[fullscreenTargetId];
    if (!source) return;
    if (fullscreenChart) {
      window.chartDestroy(fullscreenChart);
      fullscreenChart = null;
    }
    const ctx = getChartCanvas('analyticsFullscreenCanvas');
    if (!ctx) return;
    const cloned = cloneChartConfig(source);
    cloned.options = cloned.options || {};
    cloned.options.responsive = true;
    cloned.options.maintainAspectRatio = false;
    fullscreenChart = new Chart(ctx, cloned);
  });
  modalEl.addEventListener('hidden.bs.modal', () => {
    if (fullscreenChart) {
      window.chartDestroy(fullscreenChart);
      fullscreenChart = null;
    }
    fullscreenTargetId = null;
  });
}

async function loadDatasets() {
  const res = await window.apiGet('/api/datasets');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load datasets.');
    return;
  }
  const ds = res.data || [];
  const select = byId('analyticsDataset');
  if (!select) return;
  select.innerHTML = '';
  const state = await window.apiGet('/api/state');
  const activeId = state?.data?.active_dataset_id;
  if (ds.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No datasets yet — upload one first';
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  ds.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `#${d.id} • ${d.filename}`;
    select.appendChild(opt);
  });
  select.disabled = false;
  const chosen = activeId || ds[0].id;
  select.value = String(chosen);
  await loadChartsForDataset(chosen);
}

function buildLine(canvasId, labels, values, label, fill=false) {
  const { primary, primaryRgb } = themeColors();
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const axisLabel = chartAxisLabel(label);
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label, data: values, borderColor: primary, backgroundColor: `rgba(${primaryRgb},0.20)`, tension: 0.25, fill }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: label } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 }, title: { display: true, text: 'Time / Index' } },
        y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: axisLabel } }
      }
    }
  });
}

function buildBar(canvasId, labels, values, label, xTitle='Category', yTitle='Value') {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data: values, backgroundColor: 'rgba(116,83,255,0.65)' }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: label } },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: xTitle } },
        y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: yTitle } }
      }
    }
  });
}

function buildPie(canvasId, labels, values) {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const colors = palette(values.length);
  charts[canvasId] = new Chart(ctx, {
    type: 'pie',
    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
    options: { responsive: true, plugins: { tooltip: { enabled: true }, legend: { position: 'bottom' }, title: { display: true, text: 'Category share' } } }
  });
}

function buildScatter(canvasId, points, xLabel, yLabel) {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: `${yLabel} vs ${xLabel}`, data: points, backgroundColor: 'rgba(29,211,176,0.70)' }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: `${yLabel} vs ${xLabel}` } },
      scales: {
        x: { title: { display: true, text: xLabel }, grid: { color: 'rgba(255,255,255,0.08)' } },
        y: { title: { display: true, text: yLabel }, grid: { color: 'rgba(255,255,255,0.08)' } }
      }
    }
  });
}

function buildScatterWithTrend(canvasId, payload) {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const { primary, primaryRgb } = themeColors();
  const title = payload?.summary
    ? [`${payload.y_label || 'Y'} vs ${payload.x_label || 'X'}`, payload.summary]
    : `${payload.y_label || 'Y'} vs ${payload.x_label || 'X'}`;
  const datasets = [
    {
      label: `${payload?.y_label || 'Y'} vs ${payload?.x_label || 'X'}`,
      data: payload?.points || [],
      backgroundColor: 'rgba(29,211,176,0.70)',
    }
  ];
  if ((payload?.trend_points || []).length >= 2) {
    datasets.push({
      type: 'line',
      label: `Trend line (slope ${Number(payload?.slope || 0).toFixed(3)})`,
      data: payload.trend_points,
      parsing: false,
      borderColor: `rgba(${primaryRgb},0.95)`,
      backgroundColor: `rgba(${primaryRgb},0.10)`,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0,
    });
  }
  charts[canvasId] = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: datasets.length > 1 }, title: { display: true, text: title } },
      scales: {
        x: { title: { display: true, text: payload?.x_label || 'X' }, grid: { color: 'rgba(255,255,255,0.08)' } },
        y: { title: { display: true, text: payload?.y_label || 'Y' }, grid: { color: 'rgba(255,255,255,0.08)' } }
      }
    }
  });
}

function buildHeatmap(canvasId, labels, matrix) {
  const { primaryRgb } = themeColors();
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const min = -1, max = 1;
  charts[canvasId] = new Chart(ctx, {
    type: 'matrix',
    data: {
      datasets: [{
        label: 'Correlation',
        data: matrix,
        backgroundColor(c) {
          const v = c.raw.v;
          const alpha = Math.min(0.9, Math.max(0.1, Math.abs(v)));
          return v >= 0 ? `rgba(${primaryRgb},${alpha})` : `rgba(244,63,94,${alpha})`;
        },
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        width: (ctx) => (ctx.chart.chartArea || {}).width / (labels.length || 1) - 1,
        height: (ctx) => (ctx.chart.chartArea || {}).height / (labels.length || 1) - 1,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            title: (items) => {
              const it = items[0].raw;
              return `${labels[it.y]} × ${labels[it.x]}`;
            },
            label: (item) => `corr: ${item.raw.v.toFixed(2)}`
          }
        },
        legend: { display: false },
        title: { display: true, text: 'Correlation heatmap' }
      },
      scales: {
        x: { type: 'linear', ticks: { callback: (v) => labels[v] ?? '' }, min: 0, max: labels.length - 1, grid: { display: false } },
        y: { type: 'linear', ticks: { callback: (v) => labels[v] ?? '' }, min: 0, max: labels.length - 1, grid: { display: false } }
      }
    }
  });
}

function buildBoxPlot(canvasId, stats, label) {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const item = { min: stats.min, q1: stats.q1, median: stats.median, q3: stats.q3, max: stats.max };
  charts[canvasId] = new Chart(ctx, {
    type: 'boxplot',
    data: { labels: [label], datasets: [{ label: 'Box', data: [item], backgroundColor: 'rgba(245,158,11,0.35)', borderColor: 'rgba(245,158,11,0.9)' }] },
    options: { responsive: true, plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: `Box plot: ${label}` } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: label } } } }
  });
}

function buildRadar(canvasId, labels, values, label, summary='') {
  const { primary, primaryRgb } = themeColors();
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        borderColor: primary,
        backgroundColor: `rgba(${primaryRgb},0.18)`,
        pointBackgroundColor: primary,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, title: { display: true, text: chartTitle(label, summary) } },
      scales: { r: { beginAtZero: true } }
    }
  });
}

function buildPolarArea(canvasId, labels, values, label, summary='') {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'polarArea',
    data: {
      labels,
      datasets: [{ label, data: values, backgroundColor: palette(values.length).map(c => withAlpha(c, 0.65)) }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' }, title: { display: true, text: chartTitle(label, summary) } }
    }
  });
}

function buildBubble(canvasId, payload) {
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  charts[canvasId] = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [{
        label: `${payload?.size_label || 'Size'} intensity`,
        data: payload?.points || [],
        backgroundColor: 'rgba(56,189,248,0.45)',
        borderColor: 'rgba(56,189,248,0.95)',
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, title: { display: true, text: chartTitle(`${payload?.y_label || 'Y'} vs ${payload?.x_label || 'X'} bubble view`, payload?.summary || '') } },
      scales: {
        x: { title: { display: true, text: payload?.x_label || 'X' } },
        y: { title: { display: true, text: payload?.y_label || 'Y' } }
      }
    }
  });
}

function buildActualPred(canvasId, yTrue, yPred) {
  const { primaryRgb } = themeColors();
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return;
  const pts = (yTrue || []).map((y, i) => ({ x: y, y: (yPred || [])[i] ?? null })).filter(p => p.y != null);
  charts[canvasId] = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'Predicted vs Actual', data: pts, backgroundColor: `rgba(${primaryRgb},0.75)` }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: 'Actual vs Predicted' } },
      scales: {
        x: { title: { display: true, text: 'Actual' }, grid: { color: 'rgba(255,255,255,0.08)' } },
        y: { title: { display: true, text: 'Predicted' }, grid: { color: 'rgba(255,255,255,0.08)' } }
      }
    }
  });
}

function buildResidualHist(canvasId, residuals) {
  const bins = 18;
  if (!residuals || residuals.length === 0) return;
  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  residuals.forEach(r => {
    const idx = Math.max(0, Math.min(bins - 1, Math.floor((r - min) / step)));
    counts[idx] += 1;
  });
  const labels = counts.map((_, i) => `${(min + i * step).toFixed(1)}–${(min + (i + 1) * step).toFixed(1)}`);
  buildBar(canvasId, labels, counts, 'Residual distribution', 'Residual bin', 'Count');
}

function renderInsightCards(cards) {
  const wrap = byId('analyticsInsightCards');
  if (!wrap) return;
  const list = Array.isArray(cards) ? cards : [];
  if (!list.length) {
    wrap.innerHTML = '<div class="text-muted small">No clear analytical takeaways were detected for this dataset yet.</div>';
    return;
  }
  wrap.innerHTML = list.map((card) => `
    <div class="col-12 col-md-6 col-xl-4">
      <div class="panel-soft p-3 h-100">
        <div class="text-uppercase text-muted small mb-2" style="letter-spacing:.06em">${window.escapeHtml(card.title || 'Insight')}</div>
        <div>${window.escapeHtml(card.body || '')}</div>
      </div>
    </div>
  `).join('');
}

async function loadChartsForDataset(datasetId) {
  destroyAll();
  window.clearGlobalAlert();
  const res = await window.apiGet(`/api/dashboard/charts/${datasetId}`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load chart data.');
    return;
  }
  const d = res.data;
  renderInsightCards(d.story_cards || []);
  const line = d.demand_over_time || { labels: [], values: [] };
  buildLine('chartLineDemand', line.labels || [], line.values || [], chartTitle('Bike demand over time', line.summary), false);

  const pie = d.category_share_pie || { labels: [], values: [] };
  buildPie('chartPieShare', pie.labels || [], pie.values || []);
  if (charts.chartPieShare) charts.chartPieShare.options.plugins.title.text = chartTitle('Category share', pie.summary);

  const bar = d.demand_by_category_bar || { labels: [], values: [] };
  buildBar('chartBarCategory', bar.labels || [], bar.values || [], chartTitle('Demand by category', bar.summary), 'Category', 'Avg demand');

  const scatter = d.temp_vs_demand_scatter || { points: [], x_label: 'X', y_label: 'Y' };
  buildScatterWithTrend('chartScatterTemp', scatter);

  const hist = d.demand_histogram || { labels: [], values: [] };
  buildBar('chartHistDemand', hist.labels || [], hist.values || [], chartTitle('Demand distribution (histogram)', hist.summary), 'Demand bin', 'Count');

  const box = d.demand_boxplot || {};
  if (box && box.min != null) buildBoxPlot('chartBoxDemand', box, d.detected?.target_column || 'Demand');
  if (charts.chartBoxDemand && box.summary) charts.chartBoxDemand.options.plugins.title.text = chartTitle(`Box plot: ${d.detected?.target_column || 'Demand'}`, box.summary);

  const heat = d.correlation_heatmap || { labels: [], matrix: [] };
  if ((heat.labels || []).length > 1) buildHeatmap('chartHeatmap', heat.labels, heat.matrix || []);
  if (charts.chartHeatmap && heat.summary) charts.chartHeatmap.options.plugins.title.text = chartTitle('Correlation heatmap', heat.summary);

  const area = d.cumulative_demand_area || { labels: [], values: [] };
  buildLine('chartAreaCum', area.labels || [], area.values || [], chartTitle('Cumulative bike demand', area.summary), true);

  const byHour = d.demand_by_hour_bar || { labels: [], values: [] };
  buildBar('chartDemandHour', byHour.labels || [], byHour.values || [], chartTitle('Avg demand by hour', byHour.summary), 'Hour', 'Avg demand');

  const byWeekday = d.demand_by_weekday_bar || { labels: [], values: [] };
  buildBar('chartDemandWeekday', byWeekday.labels || [], byWeekday.values || [], chartTitle('Avg demand by weekday', byWeekday.summary), 'Weekday', 'Avg demand');

  const hum = d.humidity_vs_demand_scatter || { points: [], x_label: 'Humidity', y_label: 'Demand' };
  buildScatterWithTrend('chartScatterHumidity', hum);

  const work = d.demand_by_workingday_bar || { labels: [], values: [] };
  buildBar('chartWorkingday', work.labels || [], work.values || [], chartTitle('Demand by workingday', work.summary), 'Workingday', 'Avg demand');

  const month = d.demand_by_month_bar || { labels: [], values: [] };
  buildBar('chartDemandMonth', month.labels || [], month.values || [], chartTitle('Avg demand by month', month.summary), 'Month', 'Avg demand');

  const missing = d.missing_values_bar || { labels: [], values: [] };
  buildBar('chartMissing', missing.labels || [], missing.values || [], chartTitle('Missing values (top columns)', missing.summary), 'Column', 'Missing count');

  const radar = d.hourly_profile_radar || { labels: [], values: [], label: 'Hourly demand profile' };
  if ((radar.labels || []).length) buildRadar('chartRadarHour', radar.labels || [], radar.values || [], radar.label || 'Hourly demand profile', radar.summary || '');

  const polar = d.weekday_profile_polar || { labels: [], values: [], label: 'Weekday demand mix' };
  if ((polar.labels || []).length) buildPolarArea('chartPolarWeekday', polar.labels || [], polar.values || [], polar.label || 'Weekday demand mix', polar.summary || '');

  const bubble = d.temp_humidity_demand_bubble || { points: [] };
  if ((bubble.points || []).length) buildBubble('chartBubbleWeather', bubble);

  // Auto-generated extra charts (to exceed 20 graphs)
  const moreWrap = byId('moreCharts');
  if (moreWrap) {
    moreWrap.innerHTML = '';
    const blocks = [];
    const scatters = d.extra_numeric_scatters || [];
    const extraCharts = d.extra_charts || [];

    scatters.forEach((ex, idx) => {
      blocks.push({
        kind: 'scatter',
        title: `Scatter: ${ex.x_label || ex.x_col || 'Feature'} vs ${ex.y_label || ex.y_col || 'Demand'}`,
        render: (canvasId) => buildScatter(canvasId, ex.points || [], ex.x_label || ex.x_col || 'Feature', ex.y_label || ex.y_col || 'Demand'),
      });
    });

    extraCharts.forEach((ex) => {
      const kind = ex.kind || 'bar';
      blocks.push({
        kind,
        title: ex.title || ex.name || 'Chart',
        render: (canvasId) => {
          if (kind === 'line') {
            const { primary, primaryRgb } = themeColors();
            // reuse buildLine but allow custom axis titles
            const ctx = getChartCanvas(canvasId);
            if (!ctx) return;
            charts[canvasId] = new Chart(ctx, {
              type: 'line',
              data: {
                labels: ex.labels || [],
                datasets: [{
                  label: ex.title || ex.name || 'Series',
                  data: ex.values || [],
                  borderColor: primary,
                  backgroundColor: `rgba(${primaryRgb},0.18)`,
                  tension: 0.25,
                  fill: true
                }]
              },
              options: {
                responsive: true,
                plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: ex.title || '' } },
                scales: {
                  x: { grid: { display: false }, ticks: { maxTicksLimit: 8 }, title: { display: true, text: ex.xTitle || 'Index' } },
                  y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: ex.yTitle || 'Value' } }
                }
              }
            });
            return;
          }
          if (kind === 'scatter') {
            buildScatter(canvasId, ex.points || [], ex.xLabel || 'X', ex.yLabel || 'Y');
            return;
          }
          if (kind === 'pie') {
            buildPie(canvasId, ex.labels || [], ex.values || []);
            return;
          }
          buildBar(canvasId, ex.labels || [], ex.values || [], ex.title || 'Bar', ex.xTitle || 'Category', ex.yTitle || 'Value');
        }
      });
    });

    blocks.forEach((block, idx) => {
      const canvasId = `extra_${idx}`;
      const col = document.createElement('div');
      col.className = 'col-12 col-lg-6';
      col.innerHTML = `
        <div class="card card-soft">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <div class="h6 mb-0">${window.escapeHtml(block.title)}</div>
              <button class="btn btn-sm btn-outline-secondary" data-chart-target="${canvasId}">Full screen</button>
            </div>
            <canvas id="${canvasId}"></canvas>
          </div>
        </div>
      `;
      moreWrap.appendChild(col);
      block.render(canvasId);
    });
  }
}

const analyticsDataset = byId('analyticsDataset');
if (analyticsDataset) {
  analyticsDataset.addEventListener('change', async (e) => {
    const id = e.target.value;
    if (!id) return;
    await window.apiPostJson('/api/state', { active_dataset_id: Number(id) });
    await loadChartsForDataset(id);
  });
}

setupFullscreenButtons();
loadDatasets();
