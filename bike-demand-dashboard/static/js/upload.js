let currentDatasetId = null;
let currentSummary = null;
let currentCleaning = null;

function populateTargetSelect(summary) {
  const select = document.getElementById('targetSelect');
  select.innerHTML = '';
  const cols = summary?.numeric_columns || [];
  const opt0 = document.createElement('option');
  opt0.value = '';
  opt0.textContent = 'Select target (numeric)';
  select.appendChild(opt0);
  cols.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    select.appendChild(opt);
  });
  select.disabled = cols.length === 0;
  document.getElementById('saveTargetBtn').disabled = cols.length === 0;
}

function renderSummary(summary, dataset) {
  currentSummary = summary;
  document.getElementById('sumRows').textContent = summary.rows_count ?? '—';
  document.getElementById('sumCols').textContent = summary.columns_count ?? '—';
  document.getElementById('sumMissing').textContent = summary.missing_values ?? '—';
  document.getElementById('sumNumeric').textContent = (summary.numeric_columns || []).length;
  document.getElementById('sumFilename').textContent = dataset?.filename ? `File: ${dataset.filename}` : '';
}

function renderCleaningReport(report) {
  currentCleaning = report;
  const el = document.getElementById('cleaningReport');
  if (!el) return;
  if (!report) {
    el.textContent = 'No cleaning report available.';
    return;
  }
  const droppedCols = (report.dropped_columns || []).slice(0, 12);
  const convCols = report.converted_columns || {};
  const convKeys = Object.keys(convCols);
  const filled = (report.filled_missing || []).slice(0, 10);
  const filledTotal = report.filled_missing_total ?? 0;
  const before = report.before || {};
  const after = report.after || {};
  el.innerHTML = `
    <div class="row g-2">
      <div class="col-6"><span class="text-muted">Rows:</span> ${before.rows} → ${after.rows}</div>
      <div class="col-6"><span class="text-muted">Columns:</span> ${before.cols} → ${after.cols}</div>
      <div class="col-6"><span class="text-muted">Missing:</span> ${report.missing_values_before} → ${report.missing_values_after}</div>
      <div class="col-6"><span class="text-muted">Auto-filled cells:</span> ${filledTotal}</div>
      <div class="col-6"><span class="text-muted">Duplicates removed:</span> ${(report.dropped_rows?.duplicates ?? 0)}</div>
      <div class="col-12 mt-2"><span class="text-muted">Dropped columns:</span> ${droppedCols.length ? droppedCols.map(window.escapeHtml).join(', ') : '—'}${(report.dropped_columns || []).length > droppedCols.length ? '…' : ''}</div>
      <div class="col-12"><span class="text-muted">Type conversions:</span> ${convKeys.length ? convKeys.slice(0, 10).map(k => `${window.escapeHtml(k)}→${window.escapeHtml(convCols[k])}`).join(', ') : '—'}${convKeys.length > 10 ? '…' : ''}</div>
      <div class="col-12"><span class="text-muted">Filled missing:</span> ${filled.length ? filled.map(x => `${window.escapeHtml(x.column)}(${window.escapeHtml(x.strategy)}:${x.filled})`).join(', ') : '—'}${(report.filled_missing || []).length > filled.length ? '…' : ''}</div>
    </div>
  `;
}

function renderPreview(preview) {
  const table = document.getElementById('previewTable');
  const columns = preview.columns || [];
  const rows = preview.rows || [];
  window.renderTable(table, columns, rows);
  document.getElementById('previewInfo').textContent = `${rows.length} rows shown`;
}

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  window.clearGlobalAlert();

  const fileInput = document.getElementById('datasetFile');
  if (!fileInput.files || fileInput.files.length === 0) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const json = await window.apiPostForm('/api/datasets/upload', formData);
    if (!json.success) {
      window.setGlobalAlert('danger', json.message || 'Upload failed.');
      return;
    }

    const { dataset, summary, preview, cleaning_report } = json.data;
    currentDatasetId = dataset.id;
    await window.apiPostJson('/api/state', { active_dataset_id: Number(dataset.id) });
    renderSummary(summary, dataset);
    renderPreview(preview);
    renderCleaningReport(cleaning_report);
    populateTargetSelect(summary);
    window.setGlobalAlert('success', json.message || 'Uploaded.');
    await loadDatasetHistory();
  } catch (err) {
    window.setGlobalAlert('danger', String(err));
  }
});

document.getElementById('saveTargetBtn').addEventListener('click', async () => {
  const target = document.getElementById('targetSelect').value;
  if (!currentDatasetId || !target) {
    window.setGlobalAlert('warning', 'Upload a dataset and choose a target column.');
    return;
  }
  const res = await window.apiPostJson(`/api/datasets/${currentDatasetId}/target`, { target_column: target });
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to save target.');
    return;
  }
  window.setGlobalAlert('success', `Target column saved: ${target}`);
});

async function loadDatasetHistory() {
  const el = document.getElementById('datasetHistory');
  if (!el) return;
  const [dsRes, stateRes] = await Promise.all([window.apiGet('/api/datasets'), window.apiGet('/api/state')]);
  if (!dsRes.success) { el.textContent = dsRes.message || 'Failed to load datasets.'; return; }
  const datasets = dsRes.data || [];
  const activeId = stateRes?.data?.active_dataset_id;
  if (datasets.length === 0) { el.textContent = 'No datasets yet.'; return; }

  const current = datasets.find(d => String(d.id) === String(activeId)) || datasets[0];
  const previous = datasets.filter(d => String(d.id) !== String(current.id)).slice(0, 8);

  el.innerHTML = `
    <div class="p-3 rounded-4 border mb-2">
      <div class="small text-muted">Current</div>
      <div class="fw-semibold">${window.escapeHtml(current.filename)}</div>
      <div class="small text-muted">#${current.id}${current.target_column ? ` • target: ${window.escapeHtml(current.target_column)}` : ''}</div>
    </div>
    <div class="p-3 rounded-4 border">
      <div class="small text-muted mb-2">Previous</div>
      ${previous.length ? previous.map(d => `
        <div class="d-flex justify-content-between align-items-center mb-1">
          <div class="small">${window.escapeHtml(d.filename)}</div>
          <div class="d-flex gap-2">
            <button class="btn btn-sm btn-outline-primary" data-set-active="${d.id}">Set current</button>
            <button class="btn btn-sm btn-outline-danger" data-delete="${d.id}">Delete</button>
          </div>
        </div>
      `).join('') : `<div class="small text-muted">—</div>`}
    </div>
  `;

  el.querySelectorAll('button[data-set-active]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-set-active');
      await window.apiPostJson('/api/state', { active_dataset_id: Number(id) });
      window.setGlobalAlert('success', 'Current dataset updated.');
      await loadDatasetHistory();
    });
  });

  el.querySelectorAll('button[data-delete]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-delete');
      const ok = confirm(`Delete dataset #${id}? This will also delete its trained models and prediction history.`);
      if (!ok) return;
      const res = await window.apiDelete(`/api/datasets/${id}`);
      if (!res.success) {
        window.setGlobalAlert('danger', res.message || 'Delete failed.');
        return;
      }
      window.setGlobalAlert('success', res.message || 'Dataset deleted.');
      await loadDatasetHistory();
    });
  });
}

loadDatasetHistory();
