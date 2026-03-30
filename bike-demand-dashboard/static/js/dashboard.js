(async () => {
  window.clearGlobalAlert();
  const res = await window.apiGet('/api/dashboard/stats');
  if (!res.success) {
    window.setGlobalAlert('danger', res.message || 'Failed to load dashboard stats.');
    return;
  }
  const s = res.data;
  const el = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => (v == null || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);

  el('kpiDatasets').textContent = String(s.total_uploaded_datasets ?? 0);
  el('kpiModels').textContent = String(s.total_trained_models ?? 0);
  el('kpiPredictions').textContent = String(s.total_predictions_made ?? 0);

  el('kpiBestR2').textContent = fmt(s.best_r2_score, 3);
  el('kpiRmse').textContent = fmt(s.current_rmse, 3);
  if (el('kpiBestR2Text')) el('kpiBestR2Text').textContent = fmt(s.best_r2_score, 3);
  if (el('kpiRmseText')) el('kpiRmseText').textContent = fmt(s.current_rmse, 3);

  el('latestDataset').textContent = s.latest_dataset?.filename ? s.latest_dataset.filename : '—';

  const hasBest = !!s.best_model;
  const bestBlock = el('bestModelBlock');
  const bestEmpty = el('bestModelEmpty');
  if (bestBlock && bestEmpty) {
    bestBlock.classList.toggle('d-none', !hasBest);
    bestEmpty.classList.toggle('d-none', hasBest);
  }
  if (hasBest) {
    const bestText = `${s.best_model.model_name} • R² ${fmt(s.best_model.r2_score, 3)} • RMSE ${fmt(s.best_model.rmse, 3)}`;
    el('bestModelText').textContent = bestText;
  } else {
    el('bestModelText').textContent = 'No model trained yet.';
  }
  el('latestTrainingText').textContent = s.latest_training_date ? `Latest training: ${new Date(s.latest_training_date).toLocaleString()}` : '';

  const list = document.getElementById('recentActivity');
  list.innerHTML = '';
  (s.recent_activity || []).forEach(item => {
    const icon = item.type === 'dataset' ? 'cloud-arrow-up' : item.type === 'model' ? 'cpu' : 'magic';
    const el = document.createElement('div');
    el.className = 'list-group-item bg-transparent';
    el.innerHTML = `
      <div class="d-flex justify-content-between align-items-start gap-3">
        <div class="d-flex align-items-start gap-2">
          <i class="bi bi-${icon}"></i>
          <div>
            <div>${window.escapeHtml(item.title)}</div>
            <div class="small text-muted">${new Date(item.at).toLocaleString()}</div>
          </div>
        </div>
      </div>
    `;
    list.appendChild(el);
  });

  // Current vs previous dataset + workflow
  const state = await window.apiGet('/api/state');
  const activeDatasetId = state?.data?.active_dataset_id;
  const activeModelId = state?.data?.active_model_id;
  const datasetsRes = await window.apiGet('/api/datasets');
  const datasets = datasetsRes.success ? (datasetsRes.data || []) : [];

  const panel = document.getElementById('datasetPanel');
  if (panel) {
    panel.innerHTML = '';
    const current = datasets.find(d => String(d.id) === String(activeDatasetId)) || (datasets[0] || null);
    const previous = datasets.filter(d => !current || String(d.id) !== String(current.id)).slice(0, 5);

    if (current) {
      const cur = document.createElement('div');
      cur.className = 'p-3 rounded-4 border';
      cur.innerHTML = `
        <div class="small text-muted">Current</div>
        <div class="fw-semibold">${window.escapeHtml(current.filename)}</div>
        <div class="small text-muted">Dataset #${current.id}${current.target_column ? ` • target: ${window.escapeHtml(current.target_column)}` : ''}</div>
      `;
      panel.appendChild(cur);
    } else {
      panel.innerHTML = `<div class="small text-muted">No datasets uploaded yet.</div>`;
    }

    if (previous.length) {
      const prevWrap = document.createElement('div');
      prevWrap.className = 'p-3 rounded-4 border';
      prevWrap.innerHTML = `<div class="small text-muted mb-2">Previous</div>` + previous.map(d => `
        <div class="d-flex justify-content-between align-items-center mb-1">
          <div class="small">${window.escapeHtml(d.filename)}</div>
          <div class="d-flex gap-2">
            <button class="btn btn-sm btn-outline-primary" data-set-active="${d.id}">Set current</button>
            <button class="btn btn-sm btn-outline-danger" data-delete="${d.id}">Delete</button>
          </div>
        </div>
      `).join('');
      panel.appendChild(prevWrap);
      prevWrap.querySelectorAll('button[data-set-active]').forEach(btn => {
        btn.addEventListener('click', async () => {
          const id = btn.getAttribute('data-set-active');
          await window.apiPostJson('/api/state', { active_dataset_id: Number(id) });
          window.location.reload();
        });
      });

      prevWrap.querySelectorAll('button[data-delete]').forEach(btn => {
        btn.addEventListener('click', async () => {
          const id = btn.getAttribute('data-delete');
          const ok = confirm(`Delete dataset #${id}? This also deletes its models and prediction history.`);
          if (!ok) return;
          const res = await window.apiDelete(`/api/datasets/${id}`);
          if (!res.success) {
            window.setGlobalAlert('danger', res.message || 'Delete failed.');
            return;
          }
          window.location.reload();
        });
      });
    }
  }

  const wf = document.getElementById('workflowSteps');
  if (wf) {
    wf.innerHTML = '';
    const current = datasets.find(d => String(d.id) === String(activeDatasetId)) || (datasets[0] || null);
    if (!current) {
      wf.innerHTML = `
        <div class="step is-current">
          <div class="step-top">
            <div class="step-name">Upload</div>
            <span class="step-badge"><i class="bi bi-circle"></i> Start</span>
          </div>
          <div class="step-meta">Upload a CSV/Excel dataset to begin.</div>
          <div class="mt-2"><a class="btn btn-sm btn-primary" href="/upload"><i class="bi bi-cloud-arrow-up"></i> Upload</a></div>
        </div>
      `;
      return;
    }
    const statusRes = current ? await window.apiGet(`/api/datasets/${current.id}/status`) : { success: false };
    const st = statusRes.success ? (statusRes.data || {}) : {};

    // Prediction completion is scoped to the active model when available
    const predsRes = await window.apiGet('/api/predictions');
    const preds = predsRes.success ? (predsRes.data || []) : [];
    const predsForActiveModel = activeModelId ? preds.filter(p => String(p.model_id) === String(activeModelId)) : preds;

    const steps = [
      { key: 'upload', title: 'Upload', desc: 'Dataset & target selection', done: true, link: '/upload', icon: 'cloud-arrow-up' },
      { key: 'preprocess', title: 'Preprocess', desc: 'Impute • encode • split', done: !!st.has_preprocessor, link: '/preprocess', icon: 'sliders' },
      { key: 'train', title: 'Train', desc: 'Fit regression models', done: (st.models_count || 0) > 0, link: '/train', icon: 'cpu' },
      { key: 'predict', title: 'Predict', desc: 'Run what-if predictions', done: (predsForActiveModel.length || 0) > 0, link: '/predictions', icon: 'magic' },
    ];

    const currentKey = steps.find(x => !x.done)?.key || 'predict';
    steps.forEach(step => {
      const div = document.createElement('div');
      const cls = ['step'];
      if (step.done) cls.push('is-done');
      if (step.key === currentKey) cls.push('is-current');
      div.className = cls.join(' ');
      div.innerHTML = `
        <div class="step-top">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-${step.icon}"></i>
            <div class="step-name">${window.escapeHtml(step.title)}</div>
          </div>
          <span class="step-badge">${step.done ? '<i class="bi bi-check2"></i> Done' : (step.key === currentKey ? '<i class="bi bi-record-circle"></i> Current' : '<i class="bi bi-circle"></i> Pending')}</span>
        </div>
        <div class="step-meta">${window.escapeHtml(step.desc)}</div>
        <div class="mt-2"><a class="btn btn-sm ${step.key === currentKey ? 'btn-primary' : 'btn-outline-primary'}" href="${step.link}">Open</a></div>
      `;
      wf.appendChild(div);
    });
  }

  // Health indicators
  const r2 = (s.best_r2_score == null) ? null : Number(s.best_r2_score);
  const rmse = (s.current_rmse == null) ? null : Number(s.current_rmse);
  const r2Pct = (r2 == null || Number.isNaN(r2)) ? 0 : Math.max(0, Math.min(100, r2 * 100));
  const rmseHealth = (rmse == null || Number.isNaN(rmse)) ? 0 : Math.max(0, Math.min(100, 100 / (1 + rmse)));

  const barR2 = document.getElementById('barR2');
  const barRmse = document.getElementById('barRmse');
  if (barR2) barR2.style.width = `${r2Pct.toFixed(1)}%`;
  if (barRmse) barRmse.style.width = `${rmseHealth.toFixed(1)}%`;

  const status = document.getElementById('healthStatus');
  const hint = document.getElementById('healthHint');
  if (status) {
    const sText = status.querySelector('span');
    status.className = 'status';
    let label = 'Needs training';
    let cls = 'is-bad';
    if (r2 != null && !Number.isNaN(r2)) {
      if (r2 >= 0.80) { label = 'Healthy'; cls = 'is-healthy'; }
      else if (r2 >= 0.60) { label = 'Monitoring'; cls = 'is-monitor'; }
      else { label = 'Needs retraining'; cls = 'is-bad'; }
    }
    status.classList.add(cls);
    if (sText) sText.textContent = label;
  }
  if (hint) {
    if (r2 == null) hint.textContent = 'Train a model to unlock readiness monitoring.';
    else if (r2 >= 0.80) hint.textContent = 'Strong explanatory power. Continue monitoring drift with new data.';
    else if (r2 >= 0.60) hint.textContent = 'Usable baseline. Consider feature engineering and model tuning.';
    else hint.textContent = 'Low signal captured. Review target suitability and preprocessing choices.';
  }

  // Prediction sparkline (counts per day)
  try {
    const canvas = document.getElementById('predSpark');
    if (canvas && window.Chart) {
      const css = getComputedStyle(document.documentElement);
      const primary = (css.getPropertyValue('--primary') || '').trim() || '#4f46e5';
      const primaryRgb = (css.getPropertyValue('--primary-rgb') || '').trim() || '79,70,229';

      const predsRes = await window.apiGet('/api/predictions');
      const preds = predsRes.success ? (predsRes.data || []) : [];
      const now = new Date();
      const days = 14;
      const labels = [];
      const counts = [];
      const byDay = new Map();
      preds.forEach(p => {
        if (!p.predicted_at) return;
        const d = new Date(p.predicted_at);
        if (Number.isNaN(d.getTime())) return;
        const key = d.toISOString().slice(0, 10);
        byDay.set(key, (byDay.get(key) || 0) + 1);
      });
      for (let i = days - 1; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(now.getDate() - i);
        const key = d.toISOString().slice(0, 10);
        labels.push(key.slice(5));
        counts.push(byDay.get(key) || 0);
      }

      const existing = window.__predSpark;
      window.chartDestroy(existing);
      window.__predSpark = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            data: counts,
            borderColor: primary,
            backgroundColor: `rgba(${primaryRgb},0.14)`,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: true } },
          scales: { x: { display: false }, y: { display: false } },
          elements: { line: { capBezierPoints: true } }
        }
      });
    }
  } catch (e) {
    // non-blocking
  }
})();
