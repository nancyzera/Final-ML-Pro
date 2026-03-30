let models = [];

function setReportLink(modelId) {
  const btn = document.getElementById('downloadReportBtn');
  if (!btn) return;
  btn.href = modelId ? `/api/export/report/${modelId}` : '#';
  btn.classList.toggle('disabled', !modelId);
}

function renderModelSummary(model, meta) {
  const el = document.getElementById('reportStats');
  const factors = (meta?.top_factors || []).slice(0, 8);
  el.innerHTML = `
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">Model</span><span>${window.escapeHtml(model.model_name)}</span></div>
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">R²</span><span>${Number(model.r2_score).toFixed(3)}</span></div>
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">Adj R²</span><span>${Number(model.adjusted_r2).toFixed(3)}</span></div>
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">MAE</span><span>${Number(model.mae).toFixed(3)}</span></div>
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">MSE</span><span>${Number(model.mse).toFixed(3)}</span></div>
    <div class="d-flex justify-content-between mb-2"><span class="text-muted">RMSE</span><span>${Number(model.rmse).toFixed(3)}</span></div>
    <div class="mt-3">
      <div class="small text-muted mb-1">Top influencing factors</div>
      <div class="small">${factors.length ? factors.map(window.escapeHtml).join(', ') : '—'}</div>
    </div>
    <div class="mt-3 small text-muted">Tip: download the HTML report to get charts + explanations.</div>
  `;
}

async function loadModels() {
  const res = await window.apiGet('/api/models');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load models.');
    return;
  }
  models = res.data || [];
  const select = document.getElementById('reportModelSelect');
  select.innerHTML = '';
  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No models trained yet';
    select.appendChild(opt);
    select.disabled = true;
    setReportLink(null);
    return;
  }

  // prefer active model if set
  const state = await window.apiGet('/api/state');
  const activeModelId = state?.data?.active_model_id;

  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `#${m.id} • ${m.model_name} • R² ${Number(m.r2_score).toFixed(3)}`;
    select.appendChild(opt);
  });
  select.disabled = false;
  select.value = String(activeModelId || models[0].id);
  await onModelChange();
}

async function onModelChange() {
  const id = document.getElementById('reportModelSelect').value;
  if (!id) return;
  setReportLink(id);
  const res = await window.apiGet(`/api/models/${id}/metrics`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load model metrics.');
    return;
  }
  renderModelSummary(res.data.model, res.data.meta || {});
  await window.apiPostJson('/api/state', { active_model_id: Number(id) });
}

document.getElementById('reportModelSelect')?.addEventListener('change', onModelChange);
loadModels();
