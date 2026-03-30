let datasets = [];
let modelCatalog = [];
let currentParams = {}; // model_key -> {paramKey: value}

function setOutput(html) {
  document.getElementById('trainOutput').innerHTML = html;
}

function setScan(html) {
  const el = document.getElementById('datasetScan');
  if (el) el.innerHTML = html;
}

function setModelHint(text) {
  const el = document.getElementById('modelHint');
  if (el) el.textContent = text;
}

function renderParamsForm(modelName) {
  const wrap = document.getElementById('modelParams');
  if (!wrap) return;
  const meta = modelCatalog.find(m => m.name === modelName);
  if (!meta) {
    wrap.innerHTML = '<div class="text-muted">Select a model to configure parameters.</div>';
    return;
  }
  const specs = meta.params || [];
  if (!specs.length) {
    wrap.innerHTML = '<div class="text-muted">No parameters for this model.</div>';
    return;
  }

  const saved = currentParams[modelName] || {};
  const rows = specs.map((p) => {
    const key = p.key;
    const kind = p.kind;
    const id = `param_${key}`;
    const value = (saved[key] != null ? saved[key] : p.default);
    const help = p.help ? `<div class="form-text">${window.escapeHtml(p.help)}</div>` : '';

    if (kind === 'bool') {
      const checked = value ? 'checked' : '';
      return `
        <div class="form-check form-switch mb-2">
          <input class="form-check-input" type="checkbox" id="${id}" data-param-key="${window.escapeHtml(key)}" ${checked}>
          <label class="form-check-label" for="${id}">${window.escapeHtml(p.label || key)}</label>
          ${help}
        </div>
      `;
    }
    if (kind === 'select') {
      const opts = (p.options || []).map(o => `<option value="${window.escapeHtml(String(o))}" ${String(o) === String(value) ? 'selected' : ''}>${window.escapeHtml(String(o))}</option>`).join('');
      return `
        <div class="mb-2">
          <label class="form-label">${window.escapeHtml(p.label || key)}</label>
          <select class="form-select form-select-sm" id="${id}" data-param-key="${window.escapeHtml(key)}">${opts}</select>
          ${help}
        </div>
      `;
    }
    // numeric
    const min = (p.min != null) ? `min="${p.min}"` : '';
    const max = (p.max != null) ? `max="${p.max}"` : '';
    const step = (p.step != null) ? `step="${p.step}"` : '';
    return `
      <div class="mb-2">
        <label class="form-label">${window.escapeHtml(p.label || key)}</label>
        <input class="form-control form-control-sm" type="number" id="${id}" data-param-key="${window.escapeHtml(key)}" value="${window.escapeHtml(String(value))}" ${min} ${max} ${step}>
        ${help}
      </div>
    `;
  });

  wrap.innerHTML = `<div class="panel-soft p-2">${rows.join('')}</div>`;

  // Track changes so params persist when switching models
  wrap.querySelectorAll('[data-param-key]').forEach((el) => {
    const key = el.getAttribute('data-param-key');
    const save = () => {
      currentParams[modelName] = currentParams[modelName] || {};
      if (el.type === 'checkbox') currentParams[modelName][key] = el.checked;
      else currentParams[modelName][key] = el.value;
    };
    el.addEventListener('change', save);
    el.addEventListener('input', save);
  });
}

function renderCatalogCards(datasetProfile) {
  const wrap = document.getElementById('modelCatalog');
  if (!wrap) return;
  wrap.innerHTML = '';

  const rows = Number(datasetProfile?.rows_count || 0);
  const numericCount = Number(datasetProfile?.numeric_count || 0);
  const categoricalCount = Number(datasetProfile?.categorical_count || 0);
  const dtCount = Number(datasetProfile?.datetime_count || 0);
  const approxDim = Number(datasetProfile?.approx_feature_dim || 0);
  const ignored = datasetProfile?.ignored_timestamp_columns || [];

  const statusFor = (name) => {
    // "Works" means it can run; status is recommendation/feasibility.
    if (name === 'SVR (RBF)') {
      if (rows > 20000 || approxDim > 2500) return { state: 'disabled', label: 'Not suitable', cls: 'text-bg-light border' };
      if (rows > 8000 || approxDim > 1200) return { state: 'warn', label: 'Heavy', cls: 'text-bg-warning' };
      return { state: 'ok', label: 'Recommended', cls: 'text-bg-success' };
    }
    if (name === 'KNN Regressor') {
      if (rows > 50000 || approxDim > 3000) return { state: 'disabled', label: 'Not suitable', cls: 'text-bg-light border' };
      if (rows > 12000 || approxDim > 1500) return { state: 'warn', label: 'Heavy', cls: 'text-bg-warning' };
      return { state: 'ok', label: 'Recommended', cls: 'text-bg-success' };
    }
    if (name === 'Linear Regression' || name === 'Ridge Regression') {
      if (approxDim > 10000) return { state: 'warn', label: 'OK', cls: 'text-bg-warning' };
      return { state: 'ok', label: 'Recommended', cls: 'text-bg-success' };
    }
    // Tree/boosting
    if (name === 'Decision Tree Regressor') return { state: 'ok', label: 'OK', cls: 'text-bg-success' };
    if (name === 'Random Forest Regressor') return { state: 'ok', label: 'Recommended', cls: 'text-bg-success' };
    if (name === 'Gradient Boosting Regressor') return { state: 'ok', label: 'Recommended', cls: 'text-bg-success' };
    return { state: 'ok', label: 'OK', cls: 'text-bg-success' };
  };

  modelCatalog.forEach(m => {
    let st = statusFor(m.name);
    let disabled = st.state === 'disabled';

    if (m.task && m.task !== 'regression') {
      st = { state: 'disabled', label: 'Classification', cls: 'text-bg-light border' };
      disabled = true;
    }
    if (m.available === false) {
      st = { state: 'disabled', label: 'Not installed', cls: 'text-bg-light border' };
      disabled = true;
    }

    const col = document.createElement('div');
    col.className = 'col-12 col-md-6';
    const deps = (m.missing_deps || []).length ? `Missing: ${(m.missing_deps || []).map(window.escapeHtml).join(', ')}` : '';
    col.innerHTML = `
      <div class="p-3 rounded-4 border h-100 ${disabled ? 'opacity-50' : ''}">
        <div class="d-flex justify-content-between align-items-start gap-2">
          <div>
            <div class="fw-semibold">${window.escapeHtml(m.name)}</div>
            <div class="small text-muted">${window.escapeHtml(m.family || (m.task || 'Regression'))}</div>
          </div>
          <span class="badge ${st.cls}">${st.label}</span>
        </div>
        <div class="small mt-2"><span class="text-muted">Formula:</span> ${window.escapeHtml(m.formula || '—')}</div>
        <div class="small text-muted mt-1">${window.escapeHtml(m.notes || '')}</div>
        ${deps ? `<div class="small text-muted mt-1">${deps}</div>` : ''}
        <div class="d-flex gap-2 mt-2">
          <button class="btn btn-sm ${disabled ? 'btn-outline-secondary' : 'btn-outline-primary'}" ${disabled ? 'disabled' : ''} data-pick-model="${window.escapeHtml(m.name)}">
            Select
          </button>
        </div>
      </div>
    `;
    wrap.appendChild(col);
  });

  wrap.querySelectorAll('button[data-pick-model]').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.getAttribute('data-pick-model');
      const sel = document.getElementById('modelSelect');
      if (!sel) return;
      const meta = modelCatalog.find(x => x.name === name);
      if (meta?.task && meta.task !== 'regression') return;
      if (meta?.available === false) return;
      sel.value = name;
      setModelHint(meta ? `${meta.family} • ${meta.formula}` : 'Select a model to see details and formula.');
      renderParamsForm(name);
    });
  });

  // Show scan summary above catalog
  const scanBits = [
    `<div class="d-flex justify-content-between"><span class="text-muted">Rows</span><span class="fw-semibold">${rows}</span></div>`,
    `<div class="d-flex justify-content-between"><span class="text-muted">Numeric</span><span class="fw-semibold">${numericCount}</span></div>`,
    `<div class="d-flex justify-content-between"><span class="text-muted">Categorical</span><span class="fw-semibold">${categoricalCount}</span></div>`,
    `<div class="d-flex justify-content-between"><span class="text-muted">Datetime</span><span class="fw-semibold">${dtCount}</span></div>`,
    `<div class="d-flex justify-content-between"><span class="text-muted">Approx feature dim</span><span class="fw-semibold">${approxDim}</span></div>`,
  ];
  if (ignored.length) scanBits.push(`<div class="mt-2 small text-muted">Ignored timestamp columns for training: ${ignored.map(window.escapeHtml).join(', ')}</div>`);
  setScan(`<div class="small">${scanBits.join('')}</div>`);
}

async function loadTrainPage() {
  const dsRes = await window.apiGet('/api/datasets');
  if (!dsRes.success) {
    window.setGlobalAlert('danger', dsRes.message || 'Failed to load datasets.');
    return;
  }
  datasets = dsRes.data || [];
  const dsSelect = document.getElementById('trainDataset');
  dsSelect.innerHTML = '';
  const state = await window.apiGet('/api/state');
  const activeDatasetId = state?.data?.active_dataset_id;
  if (datasets.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No datasets yet — upload one first';
    dsSelect.appendChild(opt);
  } else {
    datasets.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.id;
      opt.textContent = `#${d.id} • ${d.filename}`;
      dsSelect.appendChild(opt);
    });
    dsSelect.value = String(activeDatasetId || datasets[0].id);
  }

  const modelsRes = await window.apiGet('/api/models/available');
  const modelSelect = document.getElementById('modelSelect');
  modelSelect.innerHTML = '';
  if (modelsRes.success) {
    Object.keys(modelsRes.data || {}).forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      modelSelect.appendChild(opt);
    });
  }

  const catalogRes = await window.apiGet('/api/models/catalog');
  modelCatalog = catalogRes.success ? (catalogRes.data || []) : [];
  modelSelect.onchange = () => {
    const name = modelSelect.value;
    const meta = modelCatalog.find(x => x.name === name);
    setModelHint(meta ? `${meta.family} • ${meta.formula}` : 'Select a model to see details and formula.');
    renderParamsForm(name);
  };
  if (modelSelect.value) renderParamsForm(modelSelect.value);

  await onTrainDatasetChanged();
}

async function onTrainDatasetChanged() {
  const datasetId = document.getElementById('trainDataset').value;
  if (!datasetId) return;
  await window.apiPostJson('/api/state', { active_dataset_id: Number(datasetId) });
  const res = await window.apiGet(`/api/datasets/${datasetId}/summary`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load summary.');
    return;
  }
  const summary = res.data.summary;
  const ds = datasets.find(d => String(d.id) === String(datasetId));

  const target = document.getElementById('trainTarget');
  target.innerHTML = '';
  const cols = (summary.numeric_columns || []);
  cols.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    target.appendChild(opt);
  });
  if (ds?.target_column && cols.includes(ds.target_column)) {
    target.value = ds.target_column;
  }

  // Dataset profile for model compatibility rendering
  const profileRes = await window.apiGet(`/api/datasets/${datasetId}/profile`);
  const profile = profileRes.success ? (profileRes.data || {}) : {};
  renderCatalogCards(profile);

  setOutput(`
    <div class="small">
      <div class="mb-1"><span class="text-muted">Rows:</span> ${summary.rows_count}</div>
      <div class="mb-1"><span class="text-muted">Columns:</span> ${summary.columns_count}</div>
      <div class="mb-1"><span class="text-muted">Missing:</span> ${summary.missing_values}</div>
      <div class="mt-2 text-muted">Pick a target and model to start training.</div>
    </div>
  `);
}

document.getElementById('trainDataset').addEventListener('change', onTrainDatasetChanged);
document.getElementById('trainTestSize').addEventListener('input', (e) => {
  document.getElementById('trainTestSizeLabel').textContent = Number(e.target.value).toFixed(2);
});

document.getElementById('trainBtn').addEventListener('click', async () => {
  window.clearGlobalAlert();
  const datasetId = document.getElementById('trainDataset').value;
  const target = document.getElementById('trainTarget').value;
  const modelName = document.getElementById('modelSelect').value;
  const testSize = Number(document.getElementById('trainTestSize').value);
  const scaleNumeric = document.getElementById('trainScaleNumeric').checked;
  const crossValidate = document.getElementById('trainCrossValidate')?.checked ?? true;
  const cvFolds = Number(document.getElementById('trainCvFolds')?.value ?? 5);
  if (!datasetId || !target || !modelName) {
    window.setGlobalAlert('warning', 'Select dataset, target, and model.');
    return;
  }

  setOutput(`<div class="text-muted">Training… please wait.</div>`);
  const res = await window.apiPostJson('/api/train', {
    dataset_id: Number(datasetId),
    target_column: target,
    model_name: modelName,
    test_size: testSize,
    scale_numeric: scaleNumeric,
    cross_validate: crossValidate,
    cv_folds: cvFolds,
    params: currentParams[modelName] || {}
  });
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Training failed.');
    setOutput(`
      <div class="text-danger small">${window.escapeHtml(res.message || 'Training failed.')}</div>
      <div class="small text-muted mt-2">Tip: choose a numeric target column (bike demand/count/rides) with minimal missing values.</div>
    `);
    return;
  }
  window.setGlobalAlert('success', res.message || 'Trained.');
  const m = res.data.model;
  const meta = res.data.meta || {};
  const cv = meta.training?.cross_validation || {};
  setOutput(`
    <div class="row g-2">
      <div class="col-12">
        <div class="p-3 rounded-4 border border-secondary-subtle bg-transparent">
          <div class="h6 mb-1">${window.escapeHtml(m.model_name)} trained</div>
          <div class="small text-muted">Model ID: ${m.id} • Dataset: #${m.dataset_id}</div>
          <div class="row mt-2 g-2 small">
            <div class="col-6"><span class="text-muted">R²:</span> ${Number(m.r2_score).toFixed(3)}</div>
            <div class="col-6"><span class="text-muted">Adj R²:</span> ${Number(m.adjusted_r2).toFixed(3)}</div>
            <div class="col-6"><span class="text-muted">MAE:</span> ${Number(m.mae).toFixed(3)}</div>
            <div class="col-6"><span class="text-muted">RMSE:</span> ${Number(m.rmse).toFixed(3)}</div>
          </div>
          ${cv.enabled ? `<div class="mt-2 small text-muted">CV(${cv.folds}) • R² ${Number(cv.r2_mean || 0).toFixed(3)} ± ${Number(cv.r2_std || 0).toFixed(3)}${cv.rmse_mean != null ? ` • RMSE ${Number(cv.rmse_mean).toFixed(3)}` : ''}</div>` : ``}
          <div class="mt-2 small text-muted">Top factors: ${(meta.top_factors || []).map(window.escapeHtml).join(', ') || '—'}</div>
          <div class="mt-3 d-flex gap-2">
            <a href="/performance" class="btn btn-sm btn-outline-primary"><i class="bi bi-graph-up-arrow"></i> View performance</a>
            <a href="/predictions" class="btn btn-sm btn-primary"><i class="bi bi-magic"></i> Predict</a>
          </div>
        </div>
      </div>
    </div>
  `);
});

loadTrainPage();
