function apiUrl(url) {
  const base = (window.APP_CONFIG && window.APP_CONFIG.apiBase) ? String(window.APP_CONFIG.apiBase) : '';
  if (!url) return base || '';
  // If already absolute, do not prefix
  if (String(url).startsWith('http://') || String(url).startsWith('https://')) return url;
  if (!base) return url;

  // If user configured apiBase with a page path (e.g. ".../upload"), avoid generating ".../upload/api/..."
  // Use origin/root for API paths.
  try {
    const baseResolved = new URL(base, window.location.href);
    const urlStr = String(url);
    if (urlStr.startsWith('/api/')) {
      return baseResolved.origin + urlStr;
    }
    // Non-API resources: join with apiBase as provided
    return baseResolved.href.replace(/\/+$/, '') + '/' + urlStr.replace(/^\/+/, '');
  } catch (e) {
    // Fallback: naive join
    return base.replace(/\/+$/, '') + '/' + String(url).replace(/^\/+/, '');
  }
}

window.apiGet = async function (url) {
  try {
    const requestUrl = apiUrl(url);
    const res = await fetch(requestUrl, { headers: { 'Accept': 'application/json' } });
    return await parseJsonResponse(res, requestUrl);
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

window.apiPostJson = async function (url, payload) {
  try {
    const requestUrl = apiUrl(url);
    const res = await fetch(requestUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload || {})
    });
    return await parseJsonResponse(res, requestUrl);
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

window.apiDelete = async function (url) {
  try {
    const requestUrl = apiUrl(url);
    const res = await fetch(requestUrl, { method: 'DELETE', headers: { 'Accept': 'application/json' } });
    return await parseJsonResponse(res, requestUrl);
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

window.apiPostForm = async function (url, formData) {
  try {
    const requestUrl = apiUrl(url);
    const res = await fetch(requestUrl, { method: 'POST', body: formData });
    return await parseJsonResponse(res, requestUrl);
  } catch (e) {
    return { success: false, message: String(e) };
  }
}

async function parseJsonResponse(res, requestUrl = '') {
  const text = await res.text();
  // Try JSON even if content-type is wrong (some proxies/middleware)
  try {
    return JSON.parse(text);
  } catch (e) {
    const t = String(text || '').trim().slice(0, 300);
    const looksHtml = t.startsWith('<!doctype') || t.startsWith('<html') || t.includes('<body');
    // Helpful console hint without spamming the UI
    try { console.warn('Non-JSON response snippet:', t.slice(0, 180)); } catch (_e) {}
    return {
      success: false,
      message: looksHtml
        ? `API expected JSON but received HTML (HTTP ${res.status}). URL: ${requestUrl || '(unknown)'}. This usually means the request hit a UI/download route instead of an /api JSON route, or a reverse proxy returned an HTML page.`
        : `Server returned non-JSON response (HTTP ${res.status}). URL: ${requestUrl || '(unknown)'}.`,
      detail: String(e),
      status: res.status,
      request_url: requestUrl
    };
  }
}

window.formatObjectSummary = function (obj, maxKeys = 6) {
  try {
    if (!obj || typeof obj !== 'object') return '';
    const keys = Object.keys(obj);
    const chosen = keys.slice(0, maxKeys);
    const parts = chosen.map(k => {
      const v = obj[k];
      const vs = (v == null) ? '' : String(v);
      return `${k}=${vs}`;
    });
    if (keys.length > maxKeys) parts.push('…');
    return parts.join('; ');
  } catch (e) {
    return '';
  }
}

window.setGlobalAlert = function (type, message) {
  const el = document.getElementById('globalAlert');
  if (!el) return;
  el.className = `alert alert-${type}`;
  el.textContent = message;
  el.classList.remove('d-none');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.clearGlobalAlert = function () {
  const el = document.getElementById('globalAlert');
  if (!el) return;
  el.classList.add('d-none');
}

window.renderTable = function (tableEl, columns, rows) {
  if (!tableEl) return;
  const thead = `<thead><tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r => `<tr>${columns.map(c => `<td>${escapeHtml(String(r?.[c] ?? ''))}</td>`).join('')}</tr>`).join('')}</tbody>`;
  tableEl.innerHTML = thead + tbody;
}

window.escapeHtml = function (text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

window.chartDestroy = function (chart) {
  if (chart && typeof chart.destroy === 'function') chart.destroy();
}

// Theme (light/dark) + Chart.js defaults
window.theme = {
  getSetting() {
    const saved = localStorage.getItem('bd_theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return 'system';
  },
  get() {
    const saved = localStorage.getItem('bd_theme');
    if (saved === 'light' || saved === 'dark') return saved;
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return prefersDark ? 'dark' : 'light';
  },
  set(name) {
    if (name === 'system') {
      try { localStorage.removeItem('bd_theme'); } catch (e) {}
      const resolved = window.theme.get();
      document.documentElement.setAttribute('data-theme', resolved);
      document.documentElement.setAttribute('data-bs-theme', resolved);
      applyChartDefaultsForTheme(resolved);
      const icon = document.getElementById('themeToggleIcon');
      if (icon) icon.className = resolved === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
      return;
    }
    const t = (name === 'dark') ? 'dark' : 'light';
    try { localStorage.setItem('bd_theme', t); } catch (e) {}
    document.documentElement.setAttribute('data-theme', t);
    document.documentElement.setAttribute('data-bs-theme', t);
    applyChartDefaultsForTheme(t);
    const icon = document.getElementById('themeToggleIcon');
    if (icon) icon.className = t === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
  }
};

function applyChartDefaultsForTheme(name) {
  if (!window.Chart) return;
  const isDark = name === 'dark';
  const text = isDark ? 'rgba(233,238,252,0.92)' : 'rgba(15,23,42,0.92)';
  const grid = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(2,6,23,0.08)';
  Chart.defaults.color = text;
  Chart.defaults.borderColor = grid;
  Chart.defaults.font.family = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial';
  Chart.defaults.plugins.legend.labels.color = text;
  Chart.defaults.scales.linear.grid.color = grid;
  Chart.defaults.scales.category.grid.color = grid;
  Chart.defaults.scales.time && (Chart.defaults.scales.time.grid.color = grid);
}

document.addEventListener('DOMContentLoaded', () => {
  window.theme.set(window.theme.get());
  // Theme toggle button removed from header; settings page controls theme mode.

  // Cmd/Ctrl+K focuses global search (command center feel)
  document.addEventListener('keydown', (e) => {
    const isMac = navigator.platform.toLowerCase().includes('mac');
    const metaOrCtrl = isMac ? e.metaKey : e.ctrlKey;
    if (metaOrCtrl && (e.key === 'k' || e.key === 'K')) {
      const el = document.getElementById('globalSearch');
      if (el) {
        e.preventDefault();
        el.focus();
      }
    }
  });
});
