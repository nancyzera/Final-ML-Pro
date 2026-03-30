let datasetsCache = [];
let selectedDataset = null;

function setSummaryText(html) {
  document.getElementById('preprocessSummary').innerHTML = html;
}

async function loadDatasets() {
  const res = await window.apiGet('/api/datasets');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load datasets.');
    return;
  }
  datasetsCache = res.data || [];
  const select = document.getElementById('datasetSelect');
  select.innerHTML = '';
  if (datasetsCache.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No datasets yet — upload one first';
    select.appendChild(opt);
    return;
  }
  const state = await window.apiGet('/api/state');
  const activeId = state?.data?.active_dataset_id;
  datasetsCache.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `#${d.id} • ${d.filename}`;
    select.appendChild(opt);
  });
  select.value = String(activeId || datasetsCache[0].id);
  await onDatasetChanged();
}

async function onDatasetChanged() {
  window.clearGlobalAlert();
  const datasetId = document.getElementById('datasetSelect').value;
  if (!datasetId) return;
  selectedDataset = datasetsCache.find(d => String(d.id) === String(datasetId));
  await window.apiPostJson('/api/state', { active_dataset_id: Number(datasetId) });

  const res = await window.apiGet(`/api/datasets/${datasetId}/summary`);
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load summary.');
    return;
  }
  const summary = res.data.summary;

  const targetSelect = document.getElementById('targetColumn');
  targetSelect.innerHTML = '';
  const numeric = summary.numeric_columns || [];
  numeric.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    targetSelect.appendChild(opt);
  });
  if (selectedDataset?.target_column && numeric.includes(selectedDataset.target_column)) {
    targetSelect.value = selectedDataset.target_column;
  }

  setSummaryText(`
    <div class="row g-2 small">
      <div class="col-6"><span class="text-muted">Rows:</span> ${summary.rows_count}</div>
      <div class="col-6"><span class="text-muted">Columns:</span> ${summary.columns_count}</div>
      <div class="col-6"><span class="text-muted">Missing:</span> ${summary.missing_values}</div>
      <div class="col-6"><span class="text-muted">Numeric:</span> ${(summary.numeric_columns || []).length}</div>
      <div class="col-12 mt-2"><span class="text-muted">Categorical:</span> ${(summary.categorical_columns || []).length}</div>
      <div class="col-12"><span class="text-muted">Columns:</span> ${(summary.columns || []).map(window.escapeHtml).join(', ')}</div>
    </div>
  `);
}

document.getElementById('datasetSelect').addEventListener('change', onDatasetChanged);
document.getElementById('testSize').addEventListener('input', (e) => {
  document.getElementById('testSizeLabel').textContent = Number(e.target.value).toFixed(2);
});

document.getElementById('runPreprocess').addEventListener('click', async () => {
  const datasetId = document.getElementById('datasetSelect').value;
  const target = document.getElementById('targetColumn').value;
  const testSize = Number(document.getElementById('testSize').value);
  const scaleNumeric = document.getElementById('scaleNumeric').checked;
  if (!datasetId || !target) {
    window.setGlobalAlert('warning', 'Select a dataset and target column.');
    return;
  }
  const res = await window.apiPostJson(`/api/preprocess/${datasetId}`, {
    target_column: target,
    test_size: testSize,
    scale_numeric: scaleNumeric
  });
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Preprocess failed.');
    setSummaryText(`<div class="text-danger small">${window.escapeHtml(res.message || 'Preprocess failed.')}</div>`);
    return;
  }
  window.setGlobalAlert('success', res.message || 'Saved preprocessing.');
  const p = res.data.preprocess;
  setSummaryText(`
    <div class="small">
      <div class="mb-1"><span class="text-muted">Target:</span> ${window.escapeHtml(p.target_column)}</div>
      <div class="mb-1"><span class="text-muted">Test size:</span> ${p.test_size}</div>
      <div class="mb-1"><span class="text-muted">Scale numeric:</span> ${p.scale_numeric}</div>
      <div class="mb-1"><span class="text-muted">Train shape:</span> ${p.train_shape.join(' × ')}</div>
      <div class="mb-1"><span class="text-muted">Test shape:</span> ${p.test_shape.join(' × ')}</div>
      <div class="mb-1"><span class="text-muted">Features:</span> ${(p.feature_columns || []).length}</div>
      <div class="mt-2 text-muted">Preprocessor saved to: <code>${window.escapeHtml(p.saved_preprocessor)}</code></div>
    </div>
  `);
});

loadDatasets();
