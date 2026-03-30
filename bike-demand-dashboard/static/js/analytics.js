let charts = {};

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  return {
    primary: (css.getPropertyValue('--primary') || '').trim() || '#4f46e5',
    primaryRgb: (css.getPropertyValue('--primary-rgb') || '').trim() || '79,70,229',
  };
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

async function loadDatasets() {
  const res = await window.apiGet('/api/datasets');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load datasets.');
    return;
  }
  const ds = res.data || [];
  const select = document.getElementById('analyticsDataset');
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
  const ctx = document.getElementById(canvasId);
  charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label, data: values, borderColor: primary, backgroundColor: `rgba(${primaryRgb},0.20)`, tension: 0.25, fill }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: label } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8 }, title: { display: true, text: 'Time / Index' } },
        y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: label } }
      }
    }
  });
}

function buildBar(canvasId, labels, values, label, xTitle='Category', yTitle='Value') {
  const ctx = document.getElementById(canvasId);
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
  const ctx = document.getElementById(canvasId);
  const colors = palette(values.length);
  charts[canvasId] = new Chart(ctx, {
    type: 'pie',
    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
    options: { responsive: true, plugins: { tooltip: { enabled: true }, legend: { position: 'bottom' }, title: { display: true, text: 'Category share' } } }
  });
}

function buildScatter(canvasId, points, xLabel, yLabel) {
  const ctx = document.getElementById(canvasId);
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

function buildHeatmap(canvasId, labels, matrix) {
  const { primaryRgb } = themeColors();
  const ctx = document.getElementById(canvasId);
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
  const ctx = document.getElementById(canvasId);
  const item = { min: stats.min, q1: stats.q1, median: stats.median, q3: stats.q3, max: stats.max };
  charts[canvasId] = new Chart(ctx, {
    type: 'boxplot',
    data: { labels: [label], datasets: [{ label: 'Box', data: [item], backgroundColor: 'rgba(245,158,11,0.35)', borderColor: 'rgba(245,158,11,0.9)' }] },
    options: { responsive: true, plugins: { tooltip: { enabled: true }, legend: { display: false }, title: { display: true, text: `Box plot: ${label}` } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.08)' }, title: { display: true, text: label } } } }
  });
}

function buildActualPred(canvasId, yTrue, yPred) {
  const { primaryRgb } = themeColors();
  const ctx = document.getElementById(canvasId);
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

async function loadChartsForDataset(datasetId) {
  destroyAll();
  window.clearGlobalAlert();
  const res = await window.apiGet(`/api/dashboard/charts/${datasetId}`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load chart data.');
    return;
  }
  const d = res.data;
  const line = d.demand_over_time || { labels: [], values: [] };
  buildLine('chartLineDemand', line.labels || [], line.values || [], 'Bike demand over time', false);

  const pie = d.category_share_pie || { labels: [], values: [] };
  buildPie('chartPieShare', pie.labels || [], pie.values || []);

  const bar = d.demand_by_category_bar || { labels: [], values: [] };
  buildBar('chartBarCategory', bar.labels || [], bar.values || [], 'Demand by category', 'Category', 'Avg demand');

  const scatter = d.temp_vs_demand_scatter || { points: [], x_label: 'X', y_label: 'Y' };
  buildScatter('chartScatterTemp', scatter.points || [], scatter.x_label || 'Temperature', scatter.y_label || 'Demand');

  const hist = d.demand_histogram || { labels: [], values: [] };
  buildBar('chartHistDemand', hist.labels || [], hist.values || [], 'Demand distribution (histogram)', 'Demand bin', 'Count');

  const box = d.demand_boxplot || {};
  if (box && box.min != null) buildBoxPlot('chartBoxDemand', box, d.detected?.target_column || 'Demand');

  const heat = d.correlation_heatmap || { labels: [], matrix: [] };
  if ((heat.labels || []).length > 1) buildHeatmap('chartHeatmap', heat.labels, heat.matrix || []);

  const area = d.cumulative_demand_area || { labels: [], values: [] };
  buildLine('chartAreaCum', area.labels || [], area.values || [], 'Cumulative bike demand', true);

  const byHour = d.demand_by_hour_bar || { labels: [], values: [] };
  buildBar('chartDemandHour', byHour.labels || [], byHour.values || [], 'Avg demand by hour', 'Hour', 'Avg demand');

  const byWeekday = d.demand_by_weekday_bar || { labels: [], values: [] };
  buildBar('chartDemandWeekday', byWeekday.labels || [], byWeekday.values || [], 'Avg demand by weekday', 'Weekday', 'Avg demand');

  const hum = d.humidity_vs_demand_scatter || { points: [], x_label: 'Humidity', y_label: 'Demand' };
  buildScatter('chartScatterHumidity', hum.points || [], hum.x_label || 'Humidity', hum.y_label || 'Demand');

  const work = d.demand_by_workingday_bar || { labels: [], values: [] };
  buildBar('chartWorkingday', work.labels || [], work.values || [], 'Demand by workingday', 'Workingday', 'Avg demand');

  const month = d.demand_by_month_bar || { labels: [], values: [] };
  buildBar('chartDemandMonth', month.labels || [], month.values || [], 'Avg demand by month', 'Month', 'Avg demand');

  const missing = d.missing_values_bar || { labels: [], values: [] };
  buildBar('chartMissing', missing.labels || [], missing.values || [], 'Missing values (top columns)', 'Column', 'Missing count');

  // Auto-generated extra charts (to exceed 20 graphs)
  const moreWrap = document.getElementById('moreCharts');
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
            const ctx = document.getElementById(canvasId);
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
            <div class="h6 mb-2">${window.escapeHtml(block.title)}</div>
            <canvas id="${canvasId}"></canvas>
          </div>
        </div>
      `;
      moreWrap.appendChild(col);
      block.render(canvasId);
    });
  }
}

document.getElementById('analyticsDataset').addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) return;
  await window.apiPostJson('/api/state', { active_dataset_id: Number(id) });
  await loadChartsForDataset(id);
});

loadDatasets();
