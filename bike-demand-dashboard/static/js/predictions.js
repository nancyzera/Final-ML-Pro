let models = [];
let selectedModel = null;
let modelMeta = null;

async function loadModels() {
  const res = await window.apiGet('/api/models');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load models.');
    return;
  }
  models = res.data || [];
  const select = document.getElementById('predModelSelect');
  select.innerHTML = '';
  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No models trained yet — train one first';
    select.appendChild(opt);
    select.disabled = true;
    document.getElementById('predictBtn').disabled = true;
    return;
  }
  const state = await window.apiGet('/api/state');
  const activeModelId = state?.data?.active_model_id;
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    const scoreName = m.primary_metric_name || (m.task === 'classification' ? 'Accuracy' : 'R²');
    const scoreValue = Number(m.primary_metric_value ?? m.r2_score ?? 0).toFixed(3);
    opt.textContent = `#${m.id} • ${m.model_name} • ${scoreName} ${scoreValue}`;
    select.appendChild(opt);
  });
  select.disabled = false;
  select.value = String(activeModelId || models[0].id);
  await onModelChanged();
}

function renderDynamicForm(meta) {
  const form = document.getElementById('predictionForm');
  form.innerHTML = '';
  const hints = meta?.feature_hints || {};
  const cols = meta?.feature_columns || [];
  cols.forEach(col => {
    const hint = hints[col] || {};
    const wrapper = document.createElement('div');
    wrapper.className = 'mb-2';
    const id = `f_${col}`;
    const label = document.createElement('label');
    label.className = 'form-label small text-muted';
    label.htmlFor = id;
    label.textContent = col;
    wrapper.appendChild(label);

    if (hint.type === 'categorical') {
      const select = document.createElement('select');
      select.className = 'form-select';
      select.id = id;
      const values = hint.values || [];
      const opt0 = document.createElement('option');
      opt0.value = '';
      opt0.textContent = 'Select…';
      select.appendChild(opt0);
      values.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
      });
      wrapper.appendChild(select);
    } else {
      const input = document.createElement('input');
      input.className = 'form-control';
      input.id = id;
      input.type = 'number';
      input.step = 'any';
      if (hint.min != null) input.placeholder = `e.g. ${Number(hint.mean ?? hint.min).toFixed(2)}`;
      wrapper.appendChild(input);
    }
    form.appendChild(wrapper);
  });
}

async function onModelChanged() {
  const modelId = document.getElementById('predModelSelect').value;
  if (!modelId) return;
  selectedModel = models.find(m => String(m.id) === String(modelId));
  await window.apiPostJson('/api/state', { active_model_id: Number(modelId) });
  const res = await window.apiGet(`/api/models/${modelId}/metrics`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load model metadata.');
    return;
  }
  modelMeta = res.data.meta || {};
  renderDynamicForm(modelMeta);
  document.getElementById('predictionResult').classList.add('d-none');
}

async function loadHistory() {
  const res = await window.apiGet('/api/predictions');
  if (!res.success) return;
  const rows = (res.data || []).map(p => ({
    id: p.id,
    model_id: p.model_id,
    predicted_value: Number(p.predicted_value).toFixed(3),
    predicted_at: p.predicted_at ? new Date(p.predicted_at).toLocaleString() : '',
    input_data: window.formatObjectSummary(p.input_data || {}, 7)
  }));
  const table = document.getElementById('predHistoryTable');
  window.renderTable(table, ['id', 'model_id', 'predicted_value', 'predicted_at', 'input_data'], rows);
}

document.getElementById('predModelSelect').addEventListener('change', onModelChanged);

document.getElementById('predictBtn').addEventListener('click', async () => {
  window.clearGlobalAlert();
  const modelId = document.getElementById('predModelSelect').value;
  if (!modelId) {
    window.setGlobalAlert('warning', 'Select a model first.');
    return;
  }
  const cols = modelMeta?.feature_columns || [];
  const inputs = {};
  for (const col of cols) {
    const el = document.getElementById(`f_${col}`);
    if (!el) continue;
    const val = el.value;
    inputs[col] = val;
  }
  const res = await window.apiPostJson(`/api/predict/${modelId}`, { inputs });
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Prediction failed.');
    return;
  }
  const card = document.getElementById('predictionResult');
  card.className = 'alert alert-success';
  if ((res.data.task || modelMeta?.task) === 'classification') {
    card.innerHTML = `
      <div class="fw-semibold">Predicted demand class</div>
      <div class="display-6">${window.escapeHtml(res.data.predicted_label || String(res.data.predicted_value))}</div>
      <div class="small text-muted mt-2">Class value: ${Number(res.data.predicted_value || 0).toFixed(0)}</div>
      ${res.data.predicted_probability != null ? `<div class="small text-muted">Probability: ${Number(res.data.predicted_probability).toFixed(3)}</div>` : ``}
      ${res.data.decision_score != null ? `<div class="small text-muted">Decision: ${Number(res.data.decision_score).toFixed(3)}</div>` : ``}
    `;
  } else {
    card.innerHTML = `<div class="fw-semibold">Predicted bike demand</div><div class="display-6">${Number(res.data.predicted_value).toFixed(2)}</div>`;
  }
  card.classList.remove('d-none');
  await loadHistory();
});

loadModels().then(loadHistory);
