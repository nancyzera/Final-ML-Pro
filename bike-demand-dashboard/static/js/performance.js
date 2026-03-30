let models = [];
let chart1 = null;
let chart2 = null;
let chartLc = null;
let chartLcRmse = null;
let chartLcMae = null;
let chartRvP = null;
let chartImp = null;
let chartCompare = null;
let chartCv = null;
let chartRoc = null;
let currentModelId = null;

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  return {
    primary: (css.getPropertyValue('--primary') || '').trim() || '#4f46e5',
    primaryRgb: (css.getPropertyValue('--primary-rgb') || '').trim() || '79,70,229',
    grid: (css.getPropertyValue('--grid') || '').trim() || 'rgba(2,6,23,0.08)',
  };
}

function renderModelsTable() {
  const table = document.getElementById('modelsTable');
  const columns = ['id', 'dataset_id', 'model_name', 'r2_score', 'adjusted_r2', 'mae', 'mse', 'rmse', 'trained_at'];
  const rows = models.map(m => ({
    ...m,
    r2_score: Number(m.r2_score).toFixed(3),
    adjusted_r2: Number(m.adjusted_r2).toFixed(3),
    mae: Number(m.mae).toFixed(3),
    mse: Number(m.mse).toFixed(3),
    rmse: Number(m.rmse).toFixed(3),
    trained_at: m.trained_at ? new Date(m.trained_at).toLocaleString() : ''
  }));
  window.renderTable(table, columns, rows);
}

function renderMetricStrip(model, meta) {
  const wrap = document.getElementById('modelMetricStrip');
  if (!wrap) return;
  const m = model || {};
  const mx = meta?.metrics || {};
  const cv = meta?.training?.cross_validation || {};
  const items = [
    { k: 'R²', v: m.r2_score != null ? Number(m.r2_score).toFixed(3) : '—' },
    { k: 'RMSE', v: m.rmse != null ? Number(m.rmse).toFixed(3) : '—' },
    { k: 'MAE', v: m.mae != null ? Number(m.mae).toFixed(3) : '—' },
    { k: 'MAPE', v: (mx.mape != null) ? `${(Number(mx.mape) * 100).toFixed(1)}%` : '—' },
    { k: 'Explained Var', v: (mx.explained_variance != null) ? Number(mx.explained_variance).toFixed(3) : '—' },
    { k: `CV R²`, v: (cv.enabled && cv.r2_mean != null) ? `${Number(cv.r2_mean).toFixed(3)} ± ${Number(cv.r2_std || 0).toFixed(3)}` : '—' },
  ];
  wrap.innerHTML = `
    <div class="row g-2">
      ${items.map(it => `
        <div class="col-6 col-lg-2">
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
  const picker = document.getElementById('modelPicker');
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
    opt.textContent = `#${m.id} • ${m.model_name} • R² ${Number(m.r2_score).toFixed(3)}`;
    picker.appendChild(opt);
  });
  picker.disabled = false;
  picker.value = String(models[0].id);
}

function renderActualPred(meta) {
  const { primaryRgb, grid } = themeColors();
  const payload = meta?.chart_payload || {};
  const yTrue = payload.y_true || [];
  const yPred = payload.y_pred || [];
  const pts = yTrue.map((y, i) => ({ x: y, y: yPred[i] ?? null })).filter(p => p.y != null);
  const ctx = document.getElementById('chartActualPred');
  window.chartDestroy(chart1);
  chart1 = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'Pred vs Actual', data: pts, backgroundColor: `rgba(${primaryRgb},0.72)` }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true } },
      scales: {
        x: { title: { display: true, text: 'Actual' }, grid: { color: grid } },
        y: { title: { display: true, text: 'Predicted' }, grid: { color: grid } }
      }
    }
  });
}

function renderResiduals(meta) {
  const { grid } = themeColors();
  const payload = meta?.chart_payload || {};
  const residuals = payload.residuals || [];
  const bins = 18;
  if (residuals.length === 0) return;
  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  residuals.forEach(r => {
    const idx = Math.max(0, Math.min(bins - 1, Math.floor((r - min) / step)));
    counts[idx] += 1;
  });
  const labels = counts.map((_, i) => `${(min + i * step).toFixed(1)}–${(min + (i + 1) * step).toFixed(1)}`);
  const ctx = document.getElementById('chartResiduals');
  window.chartDestroy(chart2);
  chart2 = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Residuals', data: counts, backgroundColor: 'rgba(116,83,255,0.65)' }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true } },
      scales: { x: { grid: { display: false } }, y: { grid: { color: grid }, title: { display: true, text: 'Count' } } }
    }
  });
}

function renderLearningCurve(diag) {
  const { primary, primaryRgb, grid } = themeColors();
  const lc = diag?.learning_curve_r2 || {};
  const sizes = lc.train_sizes || [];
  const train = lc.train_scores || [];
  const val = lc.val_scores || [];
  const ctx = document.getElementById('chartLearningCurve');
  window.chartDestroy(chartLc);
  if (!sizes.length) {
    chartLc = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [] }, options: { plugins: { title: { display: true, text: lc.note || 'Learning curve not available' } } } });
    return;
  }
  chartLc = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sizes.map(s => String(s)),
      datasets: [
        { label: 'Train R²', data: train, borderColor: primary, backgroundColor: `rgba(${primaryRgb},0.10)`, tension: 0.25 },
        { label: 'Validation R²', data: val, borderColor: '#16a34a', backgroundColor: 'rgba(22,163,74,0.10)', tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Learning curve (R²)' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: 'R² score' }, grid: { color: grid } }
      }
    }
  });
}

function renderLearningCurveRmse(diag) {
  const { primaryRgb, grid } = themeColors();
  const lc = diag?.learning_curve_rmse || {};
  const sizes = lc.train_sizes || [];
  const train = lc.train_scores || [];
  const val = lc.val_scores || [];
  const ctx = document.getElementById('chartLearningCurveRmse');
  window.chartDestroy(chartLcRmse);
  if (!sizes.length) {
    chartLcRmse = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [] }, options: { plugins: { title: { display: true, text: lc.note || 'Learning curve not available' } } } });
    return;
  }
  chartLcRmse = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sizes.map(s => String(s)),
      datasets: [
        { label: 'Train RMSE', data: train, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.10)', tension: 0.25 },
        { label: 'Validation RMSE', data: val, borderColor: `rgba(${primaryRgb},0.78)`, backgroundColor: `rgba(${primaryRgb},0.08)`, tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Learning curve (RMSE)' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: 'RMSE' }, grid: { color: grid } }
      }
    }
  });
}

function renderLearningCurveMae(diag) {
  const { grid } = themeColors();
  const lc = diag?.learning_curve_mae || {};
  const sizes = lc.train_sizes || [];
  const train = lc.train_scores || [];
  const val = lc.val_scores || [];
  const ctx = document.getElementById('chartLearningCurveMae');
  window.chartDestroy(chartLcMae);
  if (!sizes.length) {
    chartLcMae = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [] }, options: { plugins: { title: { display: true, text: lc.note || 'Learning curve not available' } } } });
    return;
  }
  chartLcMae = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sizes.map(s => String(s)),
      datasets: [
        { label: 'Train MAE', data: train, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.10)', tension: 0.25 },
        { label: 'Validation MAE', data: val, borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.08)', tension: 0.25 },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Learning curve (MAE)' } },
      scales: {
        x: { title: { display: true, text: 'Training samples' }, grid: { display: false } },
        y: { title: { display: true, text: 'MAE' }, grid: { color: grid } }
      }
    }
  });
}

function renderCvScores(diag) {
  const { grid } = themeColors();
  const cv = diag?.cross_validation || {};
  const folds = cv.folds || 0;
  const r2 = cv.r2_scores || [];
  const rmse = cv.rmse_scores || [];
  const labels = r2.length ? r2.map((_, i) => `Fold ${i + 1}`) : (rmse.length ? rmse.map((_, i) => `Fold ${i + 1}`) : []);
  const ctx = document.getElementById('chartCvScores');
  window.chartDestroy(chartCv);
  if (!labels.length) {
    chartCv = new Chart(ctx, { type: 'bar', data: { labels: [], datasets: [] }, options: { plugins: { title: { display: true, text: 'CV scores not available' } } } });
    return;
  }
  chartCv = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: `R² (CV${folds || ''})`, data: r2, backgroundColor: 'rgba(22,163,74,0.60)' },
        { label: `RMSE (CV${folds || ''})`, data: rmse, backgroundColor: 'rgba(239,68,68,0.50)' },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Cross-validation scores' } },
      scales: { x: { grid: { display: false } }, y: { grid: { color: grid }, title: { display: true, text: 'Score' } } }
    }
  });
}

function renderRocAuc(diag) {
  const { primaryRgb, grid } = themeColors();
  const roc = diag?.high_demand_roc || {};
  const ctx = document.getElementById('chartRocAuc');
  window.chartDestroy(chartRoc);
  if (!roc.available || !(roc.points || []).length) {
    chartRoc = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [] }, options: { plugins: { title: { display: true, text: 'ROC not available' } } } });
    return;
  }
  const pts = roc.points || [];
  chartRoc = new Chart(ctx, {
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
      plugins: { tooltip: { enabled: true }, title: { display: true, text: `ROC curve (High demand) • AUC ${Number(roc.auc || 0).toFixed(3)}` } },
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate' }, grid: { color: grid } },
        y: { min: 0, max: 1, title: { display: true, text: 'True positive rate' }, grid: { color: grid } }
      }
    }
  });
}

function renderHighDemandMetrics(diag) {
  const el = document.getElementById('highDemandMetrics');
  if (!el) return;
  const m = diag?.high_demand_metrics || {};
  if (!m.available) {
    el.textContent = 'Not available for this dataset/model.';
    return;
  }
  const cm = m.confusion_matrix || [[0, 0], [0, 0]];
  el.innerHTML = `
    <div class="row g-2">
      <div class="col-6"><span class="text-muted">Threshold:</span> ${Number(m.threshold).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Accuracy:</span> ${Number(m.accuracy).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Precision:</span> ${Number(m.precision).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">Recall:</span> ${Number(m.recall).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">F1:</span> ${Number(m.f1).toFixed(3)}</div>
      <div class="col-6"><span class="text-muted">F1 (weighted):</span> ${Number(m.f1_weighted).toFixed(3)}</div>
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
  const pts = rp.points || [];
  const ctx = document.getElementById('chartResidualVsPred');
  window.chartDestroy(chartRvP);
  chartRvP = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label: 'Residuals', data: pts, backgroundColor: 'rgba(239,68,68,0.55)' }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Residuals vs Predicted' } },
      scales: {
        x: { title: { display: true, text: rp.x_label || 'Predicted' }, grid: { color: grid } },
        y: { title: { display: true, text: rp.y_label || 'Residual' }, grid: { color: grid } }
      }
    }
  });
}

function renderImportance(diag) {
  const { primaryRgb, grid } = themeColors();
  const imp = diag?.feature_importance || {};
  const labels = imp.labels || [];
  const values = imp.values || [];
  const ctx = document.getElementById('chartImportance');
  window.chartDestroy(chartImp);
  chartImp = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Importance', data: values, backgroundColor: `rgba(${primaryRgb},0.70)` }] },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: `Feature importance (${imp.method || 'n/a'})` }, legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Feature' }, grid: { display: false }, ticks: { maxRotation: 25, minRotation: 0 } },
        y: { title: { display: true, text: 'Relative importance' }, grid: { color: grid } }
      }
    }
  });
}

function renderScoreComparison() {
  const ctx = document.getElementById('chartScoreCompare');
  const labels = models.map(m => `#${m.id} ${m.model_name}`);
  const r2 = models.map(m => Number(m.r2_score));
  const rmse = models.map(m => Number(m.rmse));
  window.chartDestroy(chartCompare);
  chartCompare = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'R²', data: r2, backgroundColor: 'rgba(22,163,74,0.60)' },
        { label: 'RMSE', data: rmse, backgroundColor: 'rgba(239,68,68,0.55)' },
      ]
    },
    options: {
      responsive: true,
      plugins: { tooltip: { enabled: true }, title: { display: true, text: 'Model comparison (R² and RMSE)' } },
      scales: {
        x: { title: { display: true, text: 'Models' }, grid: { display: false } },
        y: { title: { display: true, text: 'Score' }, grid: { color: 'rgba(2,6,23,0.08)' } }
      }
    }
  });
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
  renderResiduals(meta);
}

async function loadModelInsights(modelId) {
  const el = document.getElementById('modelInsights');
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

(async function setupAiSummary() {
  const btn = document.getElementById('aiGenerateBtn');
  if (!btn) return;

  const out = document.getElementById('aiSummary');
  const contextEl = document.getElementById('aiContext');

  async function refreshCached() {
    if (!out) return;
    if (!currentModelId) { out.textContent = 'Select a model, then click Generate.'; return; }
    const st = await window.apiGet('/api/ai/status');
    if (!st.success || !st.data?.enabled) {
      out.textContent = 'Gemini is not configured. Set GEMINI_API_KEY in .env to enable AI summaries.';
      return;
    }
    // This endpoint returns cached summary when present (force=false)
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

  // expose for model selection changes
  window.__refreshAiSummary = refreshCached;
})();

(async () => {
  window.clearGlobalAlert();
  const res = await window.apiGet('/api/models');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load models.');
    return;
  }
  models = (res.data || []).sort((a, b) => Number(b.r2_score) - Number(a.r2_score));
  renderModelsTable();
  populatePicker();
  renderScoreComparison();
  if (models.length > 0) {
    const state = await window.apiGet('/api/state');
    const activeModelId = state?.data?.active_model_id;
    const chosen = activeModelId || models[0].id;
    currentModelId = Number(chosen);
    document.getElementById('modelPicker').value = String(chosen);
    await loadModelMetrics(chosen);
    await loadModelInsights(chosen);
    await loadDiagnostics(chosen);
    if (window.__refreshAiSummary) await window.__refreshAiSummary();
  }
})();

document.getElementById('modelPicker').addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) return;
  currentModelId = Number(id);
  await window.apiPostJson('/api/state', { active_model_id: Number(id) });
  await loadModelMetrics(id);
  await loadModelInsights(id);
  await loadDiagnostics(id);
  if (window.__refreshAiSummary) await window.__refreshAiSummary();
});
