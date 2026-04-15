let models = [];
let currentModelId = null;
const chartRefs = {};
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

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  return {
    primary: (css.getPropertyValue('--primary') || '').trim() || '#4f46e5',
    primaryRgb: (css.getPropertyValue('--primary-rgb') || '').trim() || '79,70,229',
    grid: (css.getPropertyValue('--grid') || '').trim() || 'rgba(2,6,23,0.08)',
  };
}

function metricValue(model, fallbackKey, fallback = 0) {
  return Number(model?.[fallbackKey] ?? fallback);
}

function registerChart(id, chart) {
  if (chartRefs[id]) window.chartDestroy(chartRefs[id]);
  chartRefs[id] = chart;
  return chart;
}

function renderModelsTable() {
  const table = byId('modelsTable');
  const columns = ['id', 'dataset_id', 'model_name', 'task', 'primary_metric', 'secondary_metric', 'f1_score', 'f1_micro', 'precision_weighted', 'trained_at'];
  const rows = models.map(m => ({
    id: m.id,
    dataset_id: m.dataset_id,
    model_name: m.model_name,
    task: m.task || 'regression',
    primary_metric: `${m.primary_metric_name || 'R²'} ${Number(m.primary_metric_value ?? m.r2_score ?? 0).toFixed(3)}`,
    secondary_metric: `${m.secondary_metric_name || 'RMSE'} ${Number(m.secondary_metric_value ?? m.adjusted_r2 ?? 0).toFixed(3)}`,
    f1_score: m.f1_weighted != null ? Number(m.f1_weighted).toFixed(3) : '—',
    f1_micro: m.f1_micro != null ? Number(m.f1_micro).toFixed(3) : '—',
    precision_weighted: m.precision_weighted != null ? Number(m.precision_weighted).toFixed(3) : '—',
    trained_at: m.trained_at ? new Date(m.trained_at).toLocaleString() : ''
  }));
  window.renderTable(table, columns, rows);
}

function renderMetricStrip(model, meta) {
  const wrap = byId('modelMetricStrip');
  if (!wrap) return;
  const mx = meta?.metrics || {};
  const cv = meta?.training?.cross_validation || {};
  const task = meta?.task || model?.task || 'regression';
  const items = task === 'classification'
    ? [
        { k: 'Accuracy', v: Number(mx.accuracy || 0).toFixed(3) },
        { k: 'Precision', v: Number(mx.precision_weighted || mx.precision_micro || 0).toFixed(3) },
        { k: 'F1 Score', v: Number(mx.f1_weighted || mx.f1_micro || 0).toFixed(3) },
        { k: 'Micro Avg', v: Number(mx.f1_micro || 0).toFixed(3) },
        { k: 'Weighted Avg', v: Number(mx.f1_weighted || 0).toFixed(3) },
        { k: 'Decision', v: Number(mx.decision_abs_mean || mx.decision_mean || 0).toFixed(3) },
        { k: 'AUC', v: mx.roc_auc != null ? Number(mx.roc_auc).toFixed(3) : '—' },
        { k: 'CV Accuracy', v: cv.enabled ? `${Number(cv.accuracy_mean || 0).toFixed(3)} ± ${Number(cv.accuracy_std || 0).toFixed(3)}` : '—' },
      ]
    : [
        { k: 'R²', v: Number(model?.r2_score ?? 0).toFixed(3) },
        { k: 'RMSE', v: Number(model?.rmse ?? 0).toFixed(3) },
        { k: 'MAE', v: Number(model?.mae ?? 0).toFixed(3) },
        { k: 'MAPE', v: mx.mape != null ? `${(Number(mx.mape) * 100).toFixed(1)}%` : '—' },
        { k: 'Explained Var', v: mx.explained_variance != null ? Number(mx.explained_variance).toFixed(3) : '—' },
        { k: 'Adjusted R²', v: Number(model?.adjusted_r2 ?? 0).toFixed(3) },
        { k: '10-Fold CV', v: cv.enabled ? `${Number(cv.r2_mean || 0).toFixed(3)} ± ${Number(cv.r2_std || 0).toFixed(3)}` : '—' },
      ];
  wrap.innerHTML = `
    <div class="row g-2">
      ${items.map(it => `
        <div class="col-6 col-lg-3">
          <div class="panel-soft p-3">
            <div class="text-uppercase text-muted small" style="letter-spacing:.06em">${window.escapeHtml(it.k)}</div>
            <div class="h5 mb-0">${window.escapeHtml(String(it.v))}</div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function populatePicker() {
  const picker = byId('modelPicker');
  if (!picker) return;
  picker.innerHTML = '';
  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No models trained yet';
    picker.appendChild(opt);
    picker.disabled = true;
    return;
  }
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    const scoreName = m.primary_metric_name || (m.task === 'classification' ? 'Accuracy' : 'R²');
    const scoreValue = Number(m.primary_metric_value ?? m.r2_score ?? 0).toFixed(3);
    opt.textContent = `#${m.id} • ${m.model_name} • ${scoreName} ${scoreValue}`;
    picker.appendChild(opt);
  });
  picker.disabled = false;
  picker.value = String(models[0].id);
}

function buildSeriesChart(canvasId, title, labels, values, yLabel, color) {
  const { grid } = themeColors();
  const ctx = getChartCanvas(canvasId);
  if (!ctx) return null;
  return registerChart(canvasId, new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: title,
        data: values,
        borderColor: color,
        backgroundColor: color.replace('0.85', '0.12').replace('0.8', '0.12'),
        tension: 0.25,
        fill: true,
        pointRadius: values.length > 100 ? 0 : 2
      }]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: title } },
      scales: {
        x: { title: { display: true, text: 'Sample' }, grid: { display: false } },
        y: { title: { display: true, text: yLabel }, grid: { color: grid } }
      }
    }
  }));
}

function renderActualPred(meta) {
  const payload = meta?.chart_payload || {};
  const labels = payload.actual_labels || [];
  const task = meta?.task || payload.task || 'regression';
  const actualTitle = task === 'classification' ? 'Actual class labels' : 'Actual demand';
  const predTitle = task === 'classification' ? 'Predicted class labels' : 'Predicted demand';
  buildSeriesChart('chartActualSeries', actualTitle, labels, payload.actual_series || [], task === 'classification' ? 'Class' : 'Demand', 'rgba(14,165,233,0.85)');
  buildSeriesChart('chartPredSeries', predTitle, labels, payload.predicted_series || [], task === 'classification' ? 'Class' : 'Demand', 'rgba(249,115,22,0.85)');
}

function renderLearningCurve(diag) {
  const { primary, primaryRgb, grid } = themeColors();
  const lc = diag?.learning_curve_r2 || {};
  const sizes = lc.train_sizes || [];
  const ctx = getChartCanvas('chartLearningCurve');
  if (!ctx) return;
  registerChart('chartLearningCurve', new Chart(ctx, {
    type: 'line',
    data: {
      labels: sizes.map(s => String(s)),
      datasets: [
        { label: `Train ${lc.label || 'Score'}`, data: lc.train_scores || [], borderColor: primary, backgroundColor: `rgba(${primaryRgb},0.10)`, tension: 0.25 },
        { label: `Validation ${lc.label || 'Score'}`, data: lc.val_scores || [], borderColor: '#16a34a', backgroundColor: 'rgba(22,163,74,0.10)', tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: lc.label ? `Training curve (${lc.label})` : 'Training curve' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: lc.label || 'Score' }, grid: { color: grid } }
      }
    }
  }));
}

function renderLearningCurveRmse(diag) {
  const { primaryRgb, grid } = themeColors();
  const lc = diag?.learning_curve_rmse || {};
  const ctx = getChartCanvas('chartLearningCurveRmse');
  if (!ctx) return;
  registerChart('chartLearningCurveRmse', new Chart(ctx, {
    type: 'line',
    data: {
      labels: (lc.train_sizes || []).map(s => String(s)),
      datasets: [
        { label: `Train ${lc.label || 'Score'}`, data: lc.train_scores || [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.10)', tension: 0.25 },
        { label: `Validation ${lc.label || 'Score'}`, data: lc.val_scores || [], borderColor: `rgba(${primaryRgb},0.78)`, backgroundColor: `rgba(${primaryRgb},0.08)`, tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: lc.label ? `Training curve (${lc.label})` : 'Training curve' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: lc.label || 'Score' }, grid: { color: grid } }
      }
    }
  }));
}

function renderLearningCurveMae(diag) {
  const { grid } = themeColors();
  const lc = diag?.learning_curve_mae || {};
  const ctx = getChartCanvas('chartLearningCurveMae');
  if (!ctx) return;
  registerChart('chartLearningCurveMae', new Chart(ctx, {
    type: 'line',
    data: {
      labels: (lc.train_sizes || []).map(s => String(s)),
      datasets: [
        { label: `Train ${lc.label || 'Score'}`, data: lc.train_scores || [], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.10)', tension: 0.25 },
        { label: `Validation ${lc.label || 'Score'}`, data: lc.val_scores || [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.08)', tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: lc.label ? `Training curve (${lc.label})` : 'Training curve' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: lc.label || 'Score' }, grid: { color: grid } }
      }
    }
  }));
}

function renderCvScores(diag) {
  const { grid } = themeColors();
  const cv = diag?.cross_validation || {};
  const isClassification = diag?.task === 'classification';
  const labels = isClassification
    ? (cv.accuracy_scores || []).map((_, i) => `Fold ${i + 1}`)
    : (cv.r2_scores || []).map((_, i) => `Fold ${i + 1}`);
  const datasets = isClassification
    ? [
        { label: 'Accuracy', data: cv.accuracy_scores || [], backgroundColor: 'rgba(22,163,74,0.60)' },
        { label: 'F1 weighted', data: cv.f1_weighted_scores || [], backgroundColor: 'rgba(249,115,22,0.55)' },
        { label: 'Precision weighted', data: cv.precision_weighted_scores || [], backgroundColor: 'rgba(14,165,233,0.50)' },
      ]
    : [
        { label: 'R²', data: cv.r2_scores || [], backgroundColor: 'rgba(22,163,74,0.60)' },
        { label: 'RMSE', data: cv.rmse_scores || [], backgroundColor: 'rgba(239,68,68,0.50)' },
      ];
  const ctx = getChartCanvas('chartCvScores');
  if (!ctx) return;
  registerChart('chartCvScores', new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: `Cross-validation scores (${cv.folds || 10} folds)` } },
      scales: { x: { grid: { display: false } }, y: { grid: { color: grid }, title: { display: true, text: 'Score' } } }
    }
  }));
}

function renderRocAuc(diag) {
  const { primaryRgb, grid } = themeColors();
  const roc = diag?.high_demand_roc || {};
  const pts = roc.points || [];
  const ctx = getChartCanvas('chartRocAuc');
  if (!ctx) return;
  registerChart('chartRocAuc', new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: `ROC (AUC ${Number(roc.auc || 0).toFixed(3)})`,
          data: pts,
          parsing: false,
          borderColor: `rgba(${primaryRgb},0.85)`,
          backgroundColor: `rgba(${primaryRgb},0.10)`,
          tension: 0.2,
          fill: true
        },
        {
          label: 'Random',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          parsing: false,
          borderColor: 'rgba(148,163,184,0.8)',
          borderDash: [5, 5],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: `ROC curve • AUC ${Number(roc.auc || 0).toFixed(3)}` } },
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate' }, grid: { color: grid } },
        y: { min: 0, max: 1, title: { display: true, text: 'True positive rate' }, grid: { color: grid } }
      }
    }
  }));
}

function renderHighDemandMetrics(diag) {
  const el = byId('highDemandMetrics');
  if (!el) return;
  const m = diag?.high_demand_metrics || {};
  if (!m.available) {
    el.textContent = 'Not available for this dataset/model.';
    return;
  }
  const cm = m.confusion_matrix || [[0, 0], [0, 0]];
  el.innerHTML = `
    <div class="row g-2">
      <div class="col-6"><span class="text-muted">Accuracy:</span> ${Number(m.accuracy || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Precision micro:</span> ${Number(m.precision_micro || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Precision weighted:</span> ${Number(m.precision_weighted || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Recall micro:</span> ${Number(m.recall_micro || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Recall weighted:</span> ${Number(m.recall_weighted || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">F1 micro:</span> ${Number(m.f1_micro || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">F1 weighted:</span> ${Number(m.f1_weighted || 0).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Decision:</span> ${m.decision_mean != null ? Number(m.decision_mean).toFixed(3) : '—'}</div>
    </div>
    <div class="mt-2">
      <div class="text-uppercase text-muted small" style="letter-spacing:.06em">Confusion matrix</div>
      <div class="table-responsive">
        <table class="table table-sm mb-0">
          <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
          <tbody>
            <tr><th class="text-muted">True 0</th><td>${cm[0]?.[0] ?? 0}</td><td>${cm[0]?.[1] ?? 0}</td></tr>
            <tr><th class="text-muted">True 1</th><td>${cm[1]?.[0] ?? 0}</td><td>${cm[1]?.[1] ?? 0}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderResidualVsPred(diag) {
  const { grid } = themeColors();
  const rp = diag?.residuals_vs_pred || {};
  const ctx = getChartCanvas('chartResidualVsPred');
  if (!ctx) return;
  registerChart('chartResidualVsPred', new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'Residuals', data: rp.points || [], backgroundColor: 'rgba(239,68,68,0.55)' }] },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: diag?.task === 'classification' ? 'Decision profile' : 'Residuals vs predicted' } },
      scales: {
        x: { title: { display: true, text: rp.x_label || 'Predicted' }, grid: { color: grid } },
        y: { title: { display: true, text: rp.y_label || 'Residual' }, grid: { color: grid } }
      }
    }
  }));
}

function renderImportance(diag) {
  const { primaryRgb, grid } = themeColors();
  const imp = diag?.feature_importance || {};
  const ctx = getChartCanvas('chartImportance');
  if (!ctx) return;
  registerChart('chartImportance', new Chart(ctx, {
    type: 'bar',
    data: { labels: imp.labels || [], datasets: [{ label: 'Importance', data: imp.values || [], backgroundColor: `rgba(${primaryRgb},0.70)` }] },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: `Feature importance (${imp.method || 'n/a'})` }, legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Feature' }, grid: { display: false }, ticks: { maxRotation: 25, minRotation: 0 } },
        y: { title: { display: true, text: 'Relative importance' }, grid: { color: grid } }
      }
    }
  }));
}

function renderSharedNonShared(diag) {
  const el = byId('sharedNonSharedPanel');
  if (!el) return;
  const data = diag?.shared_nonshared || {};
  if (!data.available) {
    el.textContent = 'No shared/non-shared breakdown detected in this dataset.';
    return;
  }
  el.innerHTML = `
    <div class="row g-3">
      <div class="col-12 col-lg-3">
        <div class="panel-soft p-3 h-100">
          <div class="text-muted small">${window.escapeHtml(data.left_label)}</div>
          <div class="h5 mb-1">${Number(data.left_total || 0).toFixed(1)}</div>
          <div class="small text-muted">Average ${Number(data.left_avg || 0).toFixed(2)}</div>
        </div>
      </div>
      <div class="col-12 col-lg-3">
        <div class="panel-soft p-3 h-100">
          <div class="text-muted small">${window.escapeHtml(data.right_label)}</div>
          <div class="h5 mb-1">${Number(data.right_total || 0).toFixed(1)}</div>
          <div class="small text-muted">Average ${Number(data.right_avg || 0).toFixed(2)}</div>
        </div>
      </div>
      <div class="col-12 col-lg-6">
        <canvas id="chartSharedNonShared"></canvas>
      </div>
    </div>
  `;
  const { grid } = themeColors();
  const ctx = getChartCanvas('chartSharedNonShared');
  if (!ctx) return;
  registerChart('chartSharedNonShared', new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels || [],
      datasets: [
        { label: data.left_label, data: data.left_values || [], borderColor: 'rgba(22,163,74,0.85)', backgroundColor: 'rgba(22,163,74,0.10)', tension: 0.25, fill: true, pointRadius: 0 },
        { label: data.right_label, data: data.right_values || [], borderColor: 'rgba(249,115,22,0.85)', backgroundColor: 'rgba(249,115,22,0.08)', tension: 0.25, fill: true, pointRadius: 0 },
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: `${data.left_label} vs ${data.right_label}` } },
      scales: {
        x: { title: { display: true, text: 'Sample' }, grid: { display: false } },
        y: { title: { display: true, text: 'Bike count' }, grid: { color: grid } }
      }
    }
  }));
}

function renderScoreComparison() {
  const ctx = getChartCanvas('chartScoreCompare');
  if (!ctx) return;
  const labels = models.map(m => `#${m.id} ${m.model_name}`);
  const primary = models.map(m => Number(m.primary_metric_value ?? m.r2_score ?? 0));
  const secondary = models.map(m => Number(m.secondary_metric_value ?? m.adjusted_r2 ?? 0));
  registerChart('chartScoreCompare', new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: models[0]?.primary_metric_name || 'Primary score', data: primary, backgroundColor: 'rgba(22,163,74,0.60)' },
        { label: models[0]?.secondary_metric_name || 'Secondary score', data: secondary, backgroundColor: 'rgba(239,68,68,0.55)' },
      ]
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: 'Model comparison' } },
      scales: { x: { grid: { display: false } }, y: { title: { display: true, text: 'Score' }, grid: { color: 'rgba(2,6,23,0.08)' } } }
    }
  }));
}

async function loadDiagnostics(modelId) {
  const res = await window.apiGet(`/api/models/${modelId}/diagnostics`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load diagnostics.');
    return;
  }
  const diag = res.data.diagnostics || {};
  renderLearningCurve(diag);
  renderLearningCurveRmse(diag);
  renderLearningCurveMae(diag);
  renderResidualVsPred(diag);
  renderImportance(diag);
  renderCvScores(diag);
  renderRocAuc(diag);
  renderHighDemandMetrics(diag);
  renderSharedNonShared(diag);
}

async function loadModelMetrics(modelId) {
  const res = await window.apiGet(`/api/models/${modelId}/metrics`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load model metrics.');
    return;
  }
  const meta = res.data.meta || {};
  const model = res.data.model || {};
  renderMetricStrip(model, meta);
  renderActualPred(meta);
}

async function loadModelInsights(modelId) {
  const el = byId('modelInsights');
  if (!el) return;
  const res = await window.apiGet(`/api/dashboard/insights/${modelId}`);
  if (!res.success) {
    el.textContent = res.message || 'Failed to load insights.';
    return;
  }
  const d = res.data;
  const insights = d.insights?.insights || [];
  const importances = d.factor_importances || [];
  const corrs = d.correlations || [];
  const impRows = importances.slice(0, 10).map(i => `<tr><td>${window.escapeHtml(i.feature)}</td><td>${(Number(i.importance) * 100).toFixed(1)}%</td></tr>`).join('');
  const corrRows = corrs.slice(0, 8).map(c => `<tr><td>${window.escapeHtml(c.feature)}</td><td>${Number(c.corr).toFixed(2)}</td></tr>`).join('');
  el.innerHTML = `
    <div class="row g-3">
      <div class="col-12 col-lg-5">
        <div class="small text-muted mb-2">Key takeaways</div>
        ${insights.length ? `<ul class="mb-0">${insights.map(x => `<li>${window.escapeHtml(x)}</li>`).join('')}</ul>` : `<div class="small text-muted">No insights available.</div>`}
      </div>
      <div class="col-12 col-lg-7">
        <div class="row g-3">
          <div class="col-12 col-md-6">
            <div class="small text-muted mb-2">Top model factors</div>
            <div class="table-responsive">
              <table class="table table-sm">
                <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
                <tbody>${impRows || `<tr><td colspan="2" class="text-muted">—</td></tr>`}</tbody>
              </table>
            </div>
          </div>
          <div class="col-12 col-md-6">
            <div class="small text-muted mb-2">Top correlations</div>
            <div class="table-responsive">
              <table class="table table-sm">
                <thead><tr><th>Feature</th><th>corr</th></tr></thead>
                <tbody>${corrRows || `<tr><td colspan="2" class="text-muted">—</td></tr>`}</tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function setupFullscreenButtons() {
  const modalEl = byId('chartFullscreenModal');
  if (!modalEl || typeof bootstrap === 'undefined') return;
  const modal = new bootstrap.Modal(modalEl);
  document.addEventListener('click', (event) => {
    const btn = event.target.closest('[data-chart-target]');
    if (!btn) return;
    const target = btn.getAttribute('data-chart-target');
    const source = chartRefs[target];
    if (!source) return;
    fullscreenTargetId = target;
    const title = btn.closest('.card-body, .p-3, .border')?.querySelector('.h6')?.textContent || 'Chart';
    const titleEl = byId('chartFullscreenTitle');
    if (titleEl) titleEl.textContent = title;
    modal.show();
  });
  modalEl.addEventListener('shown.bs.modal', () => {
    if (!fullscreenTargetId) return;
    const source = chartRefs[fullscreenTargetId];
    if (!source) return;
    if (fullscreenChart) {
      window.chartDestroy(fullscreenChart);
      fullscreenChart = null;
    }
    const ctx = getChartCanvas('chartFullscreenCanvas');
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

(async function setupAiSummary() {
  const btn = byId('aiGenerateBtn');
  if (!btn) return;
  const out = byId('aiSummary');
  const contextEl = byId('aiContext');
  async function refreshCached() {
    if (!out) return;
    if (!currentModelId) { out.textContent = 'Select a model, then click Generate.'; return; }
    const st = await window.apiGet('/api/ai/status');
    if (!st.success || !st.data?.enabled) {
      out.textContent = 'Gemini is not configured. Set GEMINI_API_KEY in .env to enable AI summaries.';
      return;
    }
    const cached = await window.apiPostJson(`/api/ai/summary/${currentModelId}`, { force: false });
    if (cached.success && cached.data?.summary_text) out.textContent = cached.data.summary_text;
    else out.textContent = 'No AI summary yet. Click Generate.';
  }
  btn.addEventListener('click', async () => {
    window.clearGlobalAlert();
    if (!currentModelId) {
      window.setGlobalAlert('warning', 'Select a model first.');
      return;
    }
    if (out) out.textContent = 'Generating AI summary…';
    const context = (contextEl?.value || '').trim();
    const res = await window.apiPostJson(`/api/ai/summary/${currentModelId}`, { context, force: true });
    if (!res.success) {
      window.setGlobalAlert('danger', res.message || 'AI summary failed.');
      if (out) out.textContent = res.message || 'AI summary failed.';
      return;
    }
    if (out) out.textContent = res.data?.summary_text || '—';
    window.setGlobalAlert('success', res.message || 'AI summary generated.');
  });
  window.__refreshAiSummary = refreshCached;
})();

(async () => {
  setupFullscreenButtons();
  window.clearGlobalAlert();
  const res = await window.apiGet('/api/models');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load models.');
    return;
  }
  models = (res.data || []).sort((a, b) => metricValue(b, 'primary_metric_value') - metricValue(a, 'primary_metric_value'));
  renderModelsTable();
  populatePicker();
  renderScoreComparison();
  if (models.length > 0) {
    const state = await window.apiGet('/api/state');
    const activeModelId = state?.data?.active_model_id;
    const chosen = activeModelId || models[0].id;
    currentModelId = Number(chosen);
    const picker = byId('modelPicker');
    if (picker) picker.value = String(chosen);
    await loadModelMetrics(chosen);
    await loadModelInsights(chosen);
    await loadDiagnostics(chosen);
    if (window.__refreshAiSummary) await window.__refreshAiSummary();
  }
})();

const modelPicker = byId('modelPicker');
if (modelPicker) {
  modelPicker.addEventListener('change', async (e) => {
    const id = e.target.value;
    if (!id) return;
    currentModelId = Number(id);
    await window.apiPostJson('/api/state', { active_model_id: Number(id) });
    await loadModelMetrics(id);
    await loadModelInsights(id);
    await loadDiagnostics(id);
    if (window.__refreshAiSummary) await window.__refreshAiSummary();
  });
}
