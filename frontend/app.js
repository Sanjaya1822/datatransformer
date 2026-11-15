/**
 * DataTransformer 2.0 Frontend Application Logic
 * Integrates all 12 platform modules with modern reactive rendering.
 */

const API_BASE = (window.location.origin && window.location.origin !== 'null' && !window.location.origin.startsWith('file:'))
  ? window.location.origin 
  : 'http://localhost:8000';

let currentRole = 'recruiter';
let stagedFiles = [];
let allCandidatesCache = [];
let confidenceChart = null;
let missingFieldsChart = null;
let graphInstance = null;

// Role Credentials & Session Token Cache
const roleCredentials = {
  recruiter: { username: 'recruiter', password: 'recruiter123' },
};
const roleTokens = {};

async function getAuthToken(role = currentRole) {
  if (roleTokens[role]) return roleTokens[role];
  const creds = roleCredentials[role];
  if (!creds) return null;
  try {
    const res = await fetch(`${API_BASE}/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(creds),
    });
    if (res.ok) {
      const data = await res.json();
      roleTokens[role] = data.access_token;
      return data.access_token;
    }
  } catch (err) {
    console.warn(`Could not negotiate token for role '${role}':`, err);
  }
  return null;
}

// ── Auth Header & Fetch Helper ─────────────────────────────────────────────
function getAuthHeaders(extraHeaders = {}) {
  const token = roleTokens[currentRole];
  const headers = {
    'Accept': 'application/json',
    ...extraHeaders,
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function apiFetch(url, options = {}) {
  let token = await getAuthToken(currentRole);
  const headers = new Headers(options.headers || {});
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  let res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    roleTokens[currentRole] = null;
    token = await getAuthToken(currentRole);
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
      res = await fetch(url, { ...options, headers });
    }
  }
  return res;
}

async function parseApiResponse(res) {
  let data = null;
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      data = await res.json();
    } catch (e) {
      const text = await res.text().catch(() => '');
      throw new Error(`Invalid JSON received from server: ${text.slice(0, 100)}`);
    }
  } else {
    const text = await res.text().catch(() => '');
    if (!res.ok) {
      throw new Error(`Server error (${res.status}): ${text.slice(0, 150) || res.statusText}`);
    }
    throw new Error(`Unexpected non-JSON response from server (HTTP ${res.status}): ${text.slice(0, 100)}`);
  }

  if (!res.ok) {
    const errorMsg = (data && (data.detail || data.message || data.error)) || `Request failed with status ${res.status}`;
    throw new Error(errorMsg);
  }
  return data;
}


// Initialize on load
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  await getAuthToken(currentRole);
  loadDashboardMetrics();
  loadCandidates();
  loadReviewQueue();
});

// ── Role Switcher ──────────────────────────────────────────────────────────
async function setRole(role) {
  currentRole = 'recruiter';
  await getAuthToken('recruiter');
  loadCandidates();
  loadDashboardMetrics();
  loadReviewQueue();
}

// ── Tab Navigation ─────────────────────────────────────────────────────────
function switchTab(tabId) {
  const tabs = [
    'dashboard', 'upload', 'candidates', 'review-queue', 
    'compare', 'conflicts', 'jobs'
  ];

  tabs.forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    const nav = document.getElementById(`nav-${t}`);
    if (el) el.classList.add('hidden');
    if (nav) nav.classList.remove('active');
  });

  const targetEl = document.getElementById(`tab-${tabId}`);
  const targetNav = document.getElementById(`nav-${tabId}`);
  if (targetEl) targetEl.classList.remove('hidden');
  if (targetNav) targetNav.classList.add('active');

  // Trigger tab-specific loaders
  if (tabId === 'dashboard') loadDashboardMetrics();
  if (tabId === 'upload') loadUploadedFiles();
  if (tabId === 'candidates') loadCandidates();
  if (tabId === 'review-queue') loadReviewQueue();
  if (tabId === 'compare') populateCompareSelectors();
  if (tabId === 'conflicts') loadConflictDecisions();
  if (tabId === 'jobs') loadJobDescriptions();

  if (window.lucide) lucide.createIcons();
}

// ── 1. Dashboard Metrics & Charts ──────────────────────────────────────────
async function loadDashboardMetrics() {
  try {
    const res = await apiFetch(`${API_BASE}/v1/analytics/dashboard`);
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('metric-total-candidates').innerText = data.summary.total_active_profiles;
    document.getElementById('metric-duplicates-merged').innerText = data.summary.total_duplicates_merged;
    document.getElementById('metric-uncertain-queue').innerText = data.matching_metrics.uncertain_in_queue;
    document.getElementById('metric-avg-confidence').innerText = `${Math.round(data.summary.average_confidence * 100)}%`;

    const badge = document.getElementById('queue-badge-count');
    if (badge) badge.innerText = data.matching_metrics.uncertain_in_queue;

    // Confidence Doughnut Chart
    renderConfidenceChart(data.confidence_distribution);

    // Missingness Bar Chart
    renderMissingFieldsChart(data.field_missing_rates_percent);

    // Run History Table
    renderRunHistoryTable(data.run_history);
  } catch (err) {
    console.error('Failed to load dashboard metrics:', err);
  }
}

function renderConfidenceChart(dist) {
  const ctx = document.getElementById('confidence-chart');
  if (!ctx) return;

  if (confidenceChart) confidenceChart.destroy();
  confidenceChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['High (>= 85%)', 'Medium (60-84%)', 'Low (< 60%)'],
      datasets: [{
        data: [dist.high_85_100 || 0, dist.medium_60_85 || 0, dist.low_0_60 || 0],
        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
        borderWidth: 2,
        borderColor: '#ffffff',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });
}

function renderMissingFieldsChart(rates) {
  const ctx = document.getElementById('missing-fields-chart');
  if (!ctx) return;

  if (missingFieldsChart) missingFieldsChart.destroy();
  missingFieldsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(rates || {}).map(k => k.toUpperCase()),
      datasets: [{
        label: '% Missing or Incomplete',
        data: Object.values(rates || {}),
        backgroundColor: '#60a5fa',
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100 }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderRunHistoryTable(runs) {
  const tbody = document.getElementById('pipeline-runs-tbody');
  if (!tbody) return;

  if (!runs || runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="px-5 py-6 text-center text-slate-400">No pipeline executions recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = runs.map(r => `
    <tr class="hover:bg-slate-50 transition">
      <td class="px-5 py-3.5 font-mono text-xs text-brand-700 font-semibold">${r.run_id}</td>
      <td class="px-5 py-3.5">
        <span class="px-2 py-0.5 rounded-full text-xs font-bold ${r.status === 'completed' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'}">
          ${r.status}
        </span>
      </td>
      <td class="px-5 py-3.5">${r.files}</td>
      <td class="px-5 py-3.5 font-semibold text-slate-900">${r.candidates}</td>
      <td class="px-5 py-3.5 text-emerald-600 font-medium">${r.duplicates}</td>
      <td class="px-5 py-3.5 text-slate-500">${r.duration_s}s</td>
      <td class="px-5 py-3.5 text-slate-400 text-xs">${r.date || '--'}</td>
    </tr>
  `).join('');
}

// ── 2. Ingestion & Transformation ──────────────────────────────────────────
function handleFileSelection(evt) {
  const files = Array.from(evt.target.files);
  files.forEach(f => {
    if (!stagedFiles.some(existing => existing.name === f.name)) {
      stagedFiles.push(f);
    }
  });
  renderStagedFiles();
}

function renderStagedFiles() {
  const countEl = document.getElementById('staged-count');
  const listEl = document.getElementById('staged-files-list');
  if (countEl) countEl.innerText = stagedFiles.length;

  if (stagedFiles.length === 0) {
    listEl.innerHTML = '<div class="text-slate-400 italic">No files currently staged.</div>';
    return;
  }

  listEl.innerHTML = stagedFiles.map((f, i) => `
    <div class="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-200">
      <div class="flex items-center space-x-2">
        <i data-lucide="file" class="w-3.5 h-3.5 text-brand-600"></i>
        <span class="font-medium text-slate-800">${f.name}</span>
        <span class="text-slate-400 text-[11px]">(${(f.size / 1024).toFixed(1)} KB)</span>
      </div>
      <button onclick="removeStagedFile(${i})" class="text-rose-600 hover:text-rose-800 font-bold">&times;</button>
    </div>
  `).join('');

  if (window.lucide) lucide.createIcons();
}

function removeStagedFile(index) {
  stagedFiles.splice(index, 1);
  renderStagedFiles();
}

function clearStagedFiles() {
  stagedFiles = [];
  renderStagedFiles();
}

async function executePipelineRun() {
  const btn = document.getElementById('run-pipeline-btn');
  const originalText = btn.innerHTML;

  if (stagedFiles.length === 0) {
    alert('Please stage at least one resume, CSV, or ATS JSON file.');
    return;
  }

  try {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">⏳</span> Processing Ingestion & Deduplication...`;

    const configStr = document.getElementById('pipeline-config-json').value;
    const schemaStr = document.getElementById('output-schema-json').value;
    const ghUrl = document.getElementById('github-url-input').value;
    const liUrl = document.getElementById('linkedin-url-input').value;

    const formData = new FormData();
    formData.append('config_json', configStr);
    formData.append('output_schema_json', schemaStr);
    if (ghUrl) formData.append('github_url', ghUrl);
    if (liUrl) formData.append('linkedin_url', liUrl);

    stagedFiles.forEach(f => {
      formData.append('files', f);
    });

    const res = await apiFetch(`${API_BASE}/v1/merge`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Pipeline execution failed.');
    }

    const data = await res.json();
    alert(`Pipeline Completed!\n- Candidates Generated: ${data.profiles.length}\n- Duplicates Merged: ${data.summary?.duplicates_merged || 0}`);
    
    stagedFiles = [];
    renderStagedFiles();
    loadDashboardMetrics();
    loadCandidates();
    loadUploadedFiles();
    switchTab('candidates');

  } catch (err) {
    alert(`Execution Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalText;
  }
}

// ── 2b. Uploaded Files Management ────────────────────────────────────────────

let selectedFileIds = new Set();

async function loadUploadedFiles() {
  const tbody = document.getElementById('uploaded-files-tbody');
  const badge = document.getElementById('uploaded-files-badge');
  if (!tbody) return;

  try {
    tbody.innerHTML = '<tr><td colspan="8" class="px-5 py-8 text-center text-slate-400 italic">Loading uploaded files...</td></tr>';
    const res = await apiFetch(`${API_BASE}/v1/files/`);
    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}`;
      try {
        const errData = await res.json();
        if (errData && errData.detail) errorMsg = errData.detail;
      } catch (_) {}
      tbody.innerHTML = `<tr><td colspan="8" class="px-5 py-8 text-center text-rose-500 font-medium">Failed to load files: ${errorMsg}</td></tr>`;
      return;
    }
    const data = await res.json();
    const items = data.items || [];

    if (badge) badge.innerText = items.length;

    // Reset selection
    selectedFileIds = new Set();
    updateBatchDeleteButton();

    if (items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="px-5 py-8 text-center text-slate-400 italic">No uploaded files found. Use Ingest & Transform above to upload source files.</td></tr>`;
      return;
    }

    const typeColors = {
      resume: 'bg-blue-100 text-blue-700',
      csv:    'bg-green-100 text-green-700',
      github: 'bg-purple-100 text-purple-700',
      linkedin: 'bg-sky-100 text-sky-700',
      ats:    'bg-amber-100 text-amber-700',
      notes:  'bg-slate-100 text-slate-600',
    };

    tbody.innerHTML = items.map(f => {
      const typeClass = typeColors[f.file_type] || 'bg-slate-100 text-slate-600';
      const sizeKB = (f.file_size / 1024).toFixed(1);
      const uploadedAt = f.uploaded_at
        ? new Date(f.uploaded_at).toLocaleString()
        : '--';
      const statusClass = f.status === 'processed'
        ? 'text-emerald-600 font-semibold'
        : (f.status === 'error' ? 'text-rose-600 font-semibold' : 'text-amber-600 font-semibold');

      const safeFilename = (f.filename || '').replace(/'/g, "\'");

      return `
        <tr class="hover:bg-slate-50 transition" data-file-id="${f.id}">
          <td class="px-4 py-3">
            <input type="checkbox" class="file-row-checkbox rounded border-slate-300"
              value="${f.id}" onchange="onFileCheckChange('${f.id}', this.checked)" />
          </td>
          <td class="px-4 py-3 font-medium text-slate-800 flex items-center space-x-2">
            <i data-lucide="file-text" class="w-3.5 h-3.5 text-brand-500 flex-shrink-0"></i>
            <span class="truncate max-w-[200px]" title="${f.filename}">${f.filename}</span>
          </td>
          <td class="px-4 py-3">
            <span class="px-2 py-0.5 rounded-full text-[11px] font-semibold ${typeClass}">${f.file_type}</span>
          </td>
          <td class="px-4 py-3 text-slate-500">${sizeKB} KB</td>
          <td class="px-4 py-3 ${statusClass}">${f.status}</td>
          <td class="px-4 py-3 text-slate-600 text-center">${f.candidates_linked}</td>
          <td class="px-4 py-3 text-slate-400">${uploadedAt}</td>
          <td class="px-4 py-3">
            <div class="flex items-center space-x-2.5">
              <button onclick="reprocessSingleFile('${f.id}', '${safeFilename}')"
                class="text-brand-600 hover:text-brand-800 hover:underline text-xs font-semibold flex items-center space-x-1" title="Reprocess file">
                <i data-lucide="play" class="w-3 h-3"></i>
                <span>Reprocess</span>
              </button>
              <span class="text-slate-200">|</span>
              <button onclick="deleteSingleFile('${f.id}', '${safeFilename}')"
                class="text-rose-500 hover:text-rose-700 hover:underline text-xs font-medium flex items-center space-x-1" title="Delete file">
                <i data-lucide="trash-2" class="w-3 h-3"></i>
                <span>Delete</span>
              </button>
            </div>
          </td>
        </tr>
      `;
    }).join('');

    if (window.lucide) lucide.createIcons();

    const selectAll = document.getElementById('select-all-files');
    if (selectAll) selectAll.checked = false;

  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="px-5 py-8 text-center text-rose-500 font-medium">Error loading files: ${err.message}</td></tr>`;
  }
}

async function reprocessSingleFile(fileId, filename) {
  try {
    const res = await apiFetch(`${API_BASE}/v1/files/${fileId}/reprocess`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      alert(`Reprocessing failed: ${data.detail || 'Unknown error'}`);
      return;
    }
    alert(`File "${filename}" reprocessed successfully! Extracted ${data.candidates_extracted || 0} candidate(s).`);
    loadUploadedFiles();
    loadDashboardMetrics();
    loadCandidates();
  } catch (err) {
    alert(`Reprocessing error: ${err.message}`);
  }
}

function onFileCheckChange(fileId, checked) {
  if (checked) {
    selectedFileIds.add(fileId);
  } else {
    selectedFileIds.delete(fileId);
    const selectAll = document.getElementById('select-all-files');
    if (selectAll) selectAll.checked = false;
  }
  updateBatchDeleteButton();
}

function toggleSelectAllFiles(checked) {
  document.querySelectorAll('.file-row-checkbox').forEach(cb => {
    cb.checked = checked;
    const fid = cb.value;
    if (checked) {
      selectedFileIds.add(fid);
    } else {
      selectedFileIds.delete(fid);
    }
  });
  updateBatchDeleteButton();
}

function updateBatchDeleteButton() {
  const btn = document.getElementById('batch-delete-btn');
  const countEl = document.getElementById('selected-files-count');
  if (!btn) return;
  const n = selectedFileIds.size;
  btn.disabled = n === 0;
  if (countEl) countEl.innerText = n;
}

async function deleteSingleFile(fileId, filename) {
  if (!confirm(`Delete "${filename}"? This will also remove associated raw records and may update linked candidate profiles.`)) return;
  try {
    const res = await apiFetch(`${API_BASE}/v1/files/${fileId}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json();
      alert(`Delete failed: ${err.detail || 'Unknown error'}`);
      return;
    }
    await loadUploadedFiles();
    loadDashboardMetrics();
  } catch (err) {
    alert(`Error deleting file: ${err.message}`);
  }
}

async function batchDeleteSelectedFiles() {
  const ids = Array.from(selectedFileIds);
  if (ids.length === 0) return;
  if (!confirm(`Delete ${ids.length} selected file(s)? Associated raw records will be removed and linked profiles may be updated.`)) return;
  try {
    const res = await apiFetch(`${API_BASE}/v1/files/batch-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: ids }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Batch delete failed: ${err.detail || 'Unknown error'}`);
      return;
    }
    const result = await res.json();
    alert(`Deleted ${result.deleted_count ?? ids.length} file(s) successfully.`);
    await loadUploadedFiles();
    loadDashboardMetrics();
  } catch (err) {
    alert(`Error during batch delete: ${err.message}`);
  }
}

// ── 3. Candidate Explorer ──────────────────────────────────────────────────
async function loadCandidates(searchQuery = '') {
  const tbody = document.getElementById('candidates-table-tbody');
  if (!tbody) return;

  try {
    const url = new URL(`${API_BASE}/v1/candidates/`);
    if (searchQuery) url.searchParams.set('search', searchQuery);

    const res = await apiFetch(url.toString());
    if (!res.ok) {
      tbody.innerHTML = `<tr><td colspan="6" class="px-5 py-8 text-center text-rose-500">Failed to load candidate profiles (HTTP ${res.status}).</td></tr>`;
      return;
    }
    const data = await res.json();
    allCandidatesCache = data.items || [];

    if (allCandidatesCache.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="px-5 py-8 text-center text-slate-400">No candidates available. Ingest files via the 'Ingest & Transform' tab.</td></tr>`;
      return;
    }

    tbody.innerHTML = allCandidatesCache.map(c => {
      const conf = Math.round(c.overall_confidence * 100);
      const confClass = conf >= 85 ? 'conf-high' : (conf >= 60 ? 'conf-med' : 'conf-low');
      const sources = (c.sources_merged || []).map(s => `
        <span class="px-1.5 py-0.5 rounded text-[10px] bg-slate-100 text-slate-600 border border-slate-200 font-mono">${s}</span>
      `).join(' ');

      return `
        <tr class="hover:bg-slate-50 transition">
          <td class="px-5 py-3.5 font-medium text-slate-900">
            <div>${c.full_name || 'Anonymous Candidate'}</div>
            <div class="text-xs text-slate-400">${c.headline || 'No headline specified'}</div>
          </td>
          <td class="px-5 py-3.5 text-xs text-slate-600 font-mono">
            ${c.primary_email || '<span class="text-slate-400">None</span>'}
          </td>
          <td class="px-5 py-3.5 text-xs text-slate-600">
            <span class="font-semibold text-brand-700">${c.skills_count}</span> skills verified
          </td>
          <td class="px-5 py-3.5">
            <div class="flex flex-wrap gap-1">${sources || '--'}</div>
          </td>
          <td class="px-5 py-3.5">
            <span class="px-2 py-0.5 rounded-full text-xs font-bold ${confClass}">
              ${conf}%
            </span>
          </td>
          <td class="px-5 py-3.5 text-right space-x-2">
            <button onclick="viewCandidateDetail('${c.candidate_id}')" class="px-2.5 py-1 bg-brand-50 text-brand-700 hover:bg-brand-100 rounded text-xs font-semibold transition">
              View Profile
            </button>
            <button onclick="openCandidateHistory('${c.candidate_id}')" class="px-2 py-1 text-slate-500 hover:text-slate-800 text-xs">
              History
            </button>
          </td>
        </tr>
      `;
    }).join('');

  } catch (err) {
    console.error('Failed to load candidates:', err);
    tbody.innerHTML = `<tr><td colspan="6" class="px-5 py-8 text-center text-rose-500">Error connecting to server: ${err.message}</td></tr>`;
  }
}

let searchDebounceTimer = null;
function debounceCandidateSearch() {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    const query = document.getElementById('candidate-search-input').value;
    loadCandidates(query);
  }, 300);
}

// ── Candidate Detail Modal ─────────────────────────────────────────────────
async function viewCandidateDetail(candidateId) {
  try {
    const res = await apiFetch(`${API_BASE}/v1/candidates/${candidateId}`);
    if (!res.ok) return;
    const data = await res.json();
    const c = data.canonical;

    document.getElementById('modal-candidate-name').innerText = c.full_name?.value || c.full_name || 'Anonymous Candidate';
    document.getElementById('modal-candidate-id').innerText = `ID: ${candidateId} | Version: v${data.current_version} | Overall Confidence: ${Math.round(data.overall_confidence * 100)}%`;

    const skillsHtml = (c.skills || []).map(s => {
      const sName = s.name || s;
      return `<span class="px-2.5 py-1 rounded-md text-xs font-medium bg-brand-50 text-brand-700 border border-brand-200">${sName}</span>`;
    }).join(' ') || '<span class="text-slate-400">No skills recorded</span>';

    const expHtml = (c.experience || []).map(exp => `
      <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
        <div class="flex items-center justify-between">
          <span class="font-bold text-slate-900">${exp.title || 'Role'}</span>
          <span class="text-xs text-slate-500">${exp.start || exp.start_date || ''} - ${exp.end || exp.end_date || (exp.is_current ? 'Present' : '')}</span>
        </div>
        <div class="text-xs text-brand-600 font-semibold">${exp.company || 'Company'}</div>
        <div class="text-xs text-slate-600">${exp.description || exp.summary || ''}</div>
      </div>
    `).join('') || '<div class="text-slate-400">No employment records</div>';

    const eduHtml = (c.education || []).map(edu => `
      <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
        <div class="font-bold text-slate-900">${edu.institution || 'University'}</div>
        <div class="text-xs text-slate-600">${edu.degree || 'Degree'} ${edu.field || edu.field_of_study || ''} (${edu.end_year || ''})</div>
      </div>
    `).join('') || '<div class="text-slate-400">No education records</div>';

    document.getElementById('modal-candidate-body').innerHTML = `
      <div class="grid grid-cols-2 gap-4 pb-4 border-b border-slate-200 text-xs">
        <div>
          <span class="font-semibold text-slate-500">Emails:</span>
          <div>${(c.emails || []).map(e => e.value || e).join(', ') || '--'}</div>
        </div>
        <div>
          <span class="font-semibold text-slate-500">Phones:</span>
          <div>${(c.phones || []).map(p => p.value || p).join(', ') || '--'}</div>
        </div>
        <div>
          <span class="font-semibold text-slate-500">Location:</span>
          <div>${c.location?.formatted || c.location?.city || '--'}</div>
        </div>
        <div>
          <span class="font-semibold text-slate-500">Merged Sources:</span>
          <div>${(data.sources_merged || []).join(', ') || '--'}</div>
        </div>
      </div>

      <div>
        <h4 class="text-xs font-bold uppercase text-slate-500 tracking-wider mb-2">Verified Skills</h4>
        <div class="flex flex-wrap gap-1.5">${skillsHtml}</div>
      </div>

      <div>
        <h4 class="text-xs font-bold uppercase text-slate-500 tracking-wider mb-2">Work Experience</h4>
        <div class="space-y-2">${expHtml}</div>
      </div>

      <div>
        <h4 class="text-xs font-bold uppercase text-slate-500 tracking-wider mb-2">Education History</h4>
        <div class="space-y-2">${eduHtml}</div>
      </div>

      <div class="pt-4 border-t border-slate-200 flex justify-end space-x-3">
        <button onclick="promptFieldOverride('${candidateId}')" class="px-3 py-1.5 bg-amber-50 text-amber-700 border border-amber-200 rounded-lg text-xs font-bold hover:bg-amber-100">
          Manual Field Override
        </button>
      </div>
    `;

    document.getElementById('candidate-modal').classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error('Failed to load candidate modal:', err);
  }
}

async function openCandidateHistory(candidateId) {
  try {
    const res = await apiFetch(`${API_BASE}/v1/candidates/${candidateId}/versions`);
    if (!res.ok) {
      alert('Could not retrieve candidate history.');
      return;
    }
    const versions = await res.json();
    if (!versions || versions.length === 0) {
      alert(`Candidate has no recorded version history updates yet.`);
      return;
    }
    const msg = versions.map(v => `• v${v.version} (${v.changed_at ? new Date(v.changed_at).toLocaleString() : 'N/A'}): ${v.change_reason || 'Updated'}`).join('\n');
    alert(`Candidate History for ${candidateId}:\n\n${msg}`);
  } catch (e) {
    alert(`Error: ${e.message}`);
  }
}

function closeCandidateModal() {
  document.getElementById('candidate-modal').classList.add('hidden');
}

// ── Manual Field Override Prompt ───────────────────────────────────────────
async function promptFieldOverride(candidateId) {
  const fieldName = prompt('Enter field name to override (e.g. full_name, headline):');
  if (!fieldName) return;
  const newValue = prompt(`Enter new value for '${fieldName}':`);
  if (!newValue) return;
  const reason = prompt('Enter reason for manual override:');

  try {
    const res = await apiFetch(`${API_BASE}/v1/candidates/${candidateId}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        field_name: fieldName,
        new_value: newValue,
        reason: reason || 'Manual reviewer correction'
      })
    });
    if (!res.ok) throw new Error('Override failed.');
    alert('Field successfully updated! Snapshot version incremented.');
    viewCandidateDetail(candidateId);
  } catch (err) {
    alert(err.message);
  }
}

// ── 4. Review Queue (Active Learning) ──────────────────────────────────────
async function loadReviewQueue() {
  const container = document.getElementById('review-queue-container');
  if (!container) return;

  try {
    const res = await apiFetch(`${API_BASE}/v1/matching/queue`);
    if (!res.ok) return;
    const data = await res.json();

    const items = data.items || [];
    if (items.length === 0) {
      container.innerHTML = `
        <div class="bg-white p-8 rounded-xl border border-slate-200 text-center text-slate-500">
          <i data-lucide="check-circle" class="w-8 h-8 text-emerald-500 mx-auto mb-2"></i>
          <h4 class="font-bold text-slate-800">Review Queue is Clear</h4>
          <p class="text-xs text-slate-400 mt-1">No uncertain matches currently pending triage.</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
      return;
    }

    container.innerHTML = items.map(pair => `
      <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-50 text-amber-700 border border-amber-200">
              Uncertain Match (${Math.round(pair.score * 100)}%)
            </span>
            <span class="text-xs text-slate-400 font-mono">Pair: ${pair.pair_id.substring(0, 8)}</span>
          </div>
          <div class="space-x-2">
            <button onclick="resolvePairReview('${pair.pair_id}', 'approve_merge')" class="px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-bold hover:bg-emerald-700 transition">
              Approve Merge
            </button>
            <button onclick="resolvePairReview('${pair.pair_id}', 'reject_match')" class="px-3 py-1.5 bg-rose-600 text-white rounded-lg text-xs font-bold hover:bg-rose-700 transition">
              Reject Match
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4 text-xs bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div>
            <span class="font-bold text-slate-800">${pair.candidate_a.name || 'Candidate A'}</span>
            <div class="text-slate-400 font-mono mt-0.5">ID: ${pair.candidate_a.candidate_id}</div>
          </div>
          <div>
            <span class="font-bold text-slate-800">${pair.candidate_b.name || 'Candidate B'}</span>
            <div class="text-slate-400 font-mono mt-0.5">ID: ${pair.candidate_b.candidate_id}</div>
          </div>
        </div>

        <div class="text-xs text-slate-600 space-y-1">
          <div class="font-semibold text-slate-700">Contributing Signals:</div>
          <ul class="list-disc pl-5 space-y-0.5 text-slate-500">
            ${(pair.signals || []).map(s => `<li>${s.explanation} (weight: ${s.contribution})</li>`).join('')}
          </ul>
        </div>
      </div>
    `).join('');

    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error('Failed to load review queue:', err);
  }
}

async function resolvePairReview(pairId, decision) {
  try {
    const res = await apiFetch(`${API_BASE}/v1/matching/pairs/${pairId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision })
    });
    if (!res.ok) throw new Error('Decision submission failed.');
    loadReviewQueue();
    loadDashboardMetrics();
  } catch (err) {
    alert(err.message);
  }
}

async function triggerRecalibration() {
  try {
    const res = await apiFetch(`${API_BASE}/v1/matching/recalibrate`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Recalibration failed.');
    alert(`Model Recalibration Complete!\n- Version: v${data.new_version}\n- Threshold: ${data.high_confidence_threshold}\n- Feedback Samples: ${data.feedback_count}`);
  } catch (err) {
    alert(err.message);
  }
}

// ── 5. Side-by-Side Candidate Compare ──────────────────────────────────────
function populateCompareSelectors() {
  const selA = document.getElementById('compare-select-a');
  const selB = document.getElementById('compare-select-b');
  if (!selA || !selB) return;

  const options = allCandidatesCache.map(c => 
    `<option value="${c.candidate_id}">${c.full_name || 'Anonymous'} (${c.candidate_id.substring(0, 8)})</option>`
  ).join('');

  selA.innerHTML = options;
  selB.innerHTML = options;
  if (selB.children.length > 1) selB.selectedIndex = 1;
}

async function runCandidateComparison() {
  const idA = document.getElementById('compare-select-a').value;
  const idB = document.getElementById('compare-select-b').value;
  const container = document.getElementById('comparison-results-container');
  const btn = document.querySelector('#tab-compare button');

  if (!idA || !idB || idA === idB) {
    alert('Please select two distinct candidates to compare.');
    return;
  }

  try {
    if (btn) btn.disabled = true;
    container.innerHTML = '<div class="text-center p-12 text-slate-400"><div class="animate-spin w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full mx-auto mb-2"></div>Executing attribute reconciliation comparison...</div>';

    const res = await apiFetch(`${API_BASE}/v1/matching/compare/${idA}/${idB}`);
    const data = await parseApiResponse(res);

    const statusBadgeStyles = {
      'Exact Match': 'bg-emerald-50 text-emerald-700 border-emerald-200',
      'Partial Match': 'bg-sky-50 text-sky-700 border-sky-200',
      'Conflict': 'bg-rose-50 text-rose-700 border-rose-200',
      'Missing in Candidate A': 'bg-amber-50 text-amber-700 border-amber-200',
      'Missing in Candidate B': 'bg-amber-50 text-amber-700 border-amber-200',
      'Insufficient Evidence': 'bg-slate-100 text-slate-600 border-slate-200',
    };

    const statusIcons = {
      'Exact Match': 'check-circle-2',
      'Partial Match': 'git-commit',
      'Conflict': 'alert-triangle',
      'Missing in Candidate A': 'minus-circle',
      'Missing in Candidate B': 'minus-circle',
      'Insufficient Evidence': 'help-circle',
    };

    const diffRows = Object.entries(data.field_diffs || {}).map(([fKey, d]) => {
      const badgeClass = statusBadgeStyles[d.status] || 'bg-slate-100 text-slate-600 border-slate-200';
      const iconName = statusIcons[d.status] || 'info';
      const rowBg = d.status === 'Conflict' ? 'bg-rose-50/20' : (d.status === 'Exact Match' ? 'bg-emerald-50/20' : '');

      return `
        <tr class="border-b border-slate-100 hover:bg-slate-50/80 transition ${rowBg}">
          <td class="py-3 px-4 font-semibold text-slate-800 whitespace-nowrap">${d.field_label || fKey}</td>
          <td class="py-3 px-4 text-xs text-slate-700 max-w-[200px] break-words">${d.value_a || '--'}</td>
          <td class="py-3 px-4 text-xs text-slate-700 max-w-[200px] break-words">${d.value_b || '--'}</td>
          <td class="py-3 px-4 text-center whitespace-nowrap">
            <span class="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${badgeClass}">
              <span>${d.status}</span>
            </span>
          </td>
          <td class="py-3 px-4 text-xs text-slate-500">${d.explanation || '--'}</td>
        </tr>
      `;
    }).join('');

    const breakdownRows = (data.score_breakdown || []).map(b => {
      const badgeClass = statusBadgeStyles[b.status] || 'bg-slate-100 text-slate-600 border-slate-200';
      return `
        <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
          <div class="flex items-center justify-between text-xs">
            <span class="font-bold text-slate-800">${b.attribute}</span>
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${badgeClass}">${b.status}</span>
          </div>
          <div class="flex items-center justify-between text-[11px] text-slate-500">
            <span>Weight: ${(b.weight * 100).toFixed(0)}%</span>
            <span>Score: ${(b.score * 100).toFixed(0)}% (contrib: +${(b.contribution * 100).toFixed(1)}%)</span>
          </div>
          <div class="text-[11px] text-slate-600">${b.explanation}</div>
        </div>
      `;
    }).join('');

    const identityBadgeClass = data.identity_verdict === 'Same Candidate' || data.is_same_candidate
      ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
      : (data.identity_verdict === 'Different Candidates' ? 'bg-slate-100 text-slate-700 border-slate-300' : 'bg-amber-50 text-amber-800 border-amber-300');

    container.innerHTML = `
      <div class="space-y-6">
        <!-- Summary Header Card -->
        <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-200">
            <div>
              <div class="flex items-center space-x-2">
                <h3 class="text-lg font-bold text-slate-900">Similarity Score: ${data.similarity_percentage}%</h3>
                <span class="px-2.5 py-0.5 rounded-full text-xs font-bold border ${identityBadgeClass}">
                  ${data.identity_verdict || 'Evaluation Complete'}
                </span>
              </div>
              <p class="text-xs text-slate-600 mt-1">${data.explanation}</p>
            </div>
            <div class="flex items-center space-x-2">
              <span class="px-3 py-1 rounded-full text-xs font-bold ${data.classification === 'high_confidence' ? 'conf-high' : 'conf-med'}">
                ${data.classification.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>

          <!-- Score Breakdown Grid -->
          <div>
            <h4 class="text-xs font-bold text-slate-600 uppercase tracking-wider mb-2">Attribute Similarity Breakdown</h4>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
              ${breakdownRows}
            </div>
          </div>
        </div>

        <!-- Field-by-Field Diff Table -->
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div class="px-5 py-3.5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
            <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider">Granular Attribute Alignment & Diff Matrix</h4>
            <span class="text-xs text-slate-400">Comparing meaningful profile fields</span>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-slate-50 uppercase text-slate-500 border-b border-slate-200">
                <tr>
                  <th class="py-2.5 px-4">Field Attribute</th>
                  <th class="py-2.5 px-4">Candidate A</th>
                  <th class="py-2.5 px-4">Candidate B</th>
                  <th class="py-2.5 px-4 text-center">Match Status</th>
                  <th class="py-2.5 px-4">Explanation / Difference</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-100">${diffRows}</tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

  } catch (err) {
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl space-y-2 text-center">
        <div class="font-bold text-sm">Comparison Error</div>
        <p class="text-xs text-rose-600">${err.message}</p>
      </div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ── 5b. Job-Specific Candidate Evaluation (AI Engine) ─────────────────────

function loadSampleJobRole(role) {
  const presets = {
    java: {
      title: 'Java Full Stack Developer',
      reqSkills: 'Java, PostgreSQL, HTML, CSS, JavaScript',
      prefSkills: 'Spring Boot, React, Docker',
      minExp: '1.0',
      edu: "Bachelor's Degree in CS, IT, or related field",
      desc: 'Develop robust web platforms, design relational database schemas, and create responsive user interfaces. Work with REST APIs and backend Java services.',
      additional: 'REST APIs, Git, Problem Solving, Teamwork',
    },
    frontend: {
      title: 'Frontend Developer',
      reqSkills: 'React, HTML, CSS, JavaScript, TypeScript',
      prefSkills: 'Next.js, TailwindCSS, Figma',
      minExp: '0.5',
      edu: "Bachelor's Degree in any technical field",
      desc: 'Build responsive, accessible web UIs using modern JavaScript frameworks. Collaborate with backend teams on API integration and design systems.',
      additional: 'UI/UX awareness, Component-driven development',
    },
    ml: {
      title: 'Machine Learning Engineer',
      reqSkills: 'Python, Pandas, NumPy, Machine Learning, Scikit-Learn',
      prefSkills: 'TensorFlow, PyTorch, SQL, Deep Learning',
      minExp: '1.0',
      edu: "Bachelor's or Master's Degree in CS, Data Science, or related field",
      desc: 'Build, train, and deploy ML models. Perform EDA, feature engineering, and model evaluation. Work with large datasets and present insights.',
      additional: 'Statistical analysis, Jupyter, Matplotlib, Seaborn',
    },
  };
  const p = presets[role];
  if (!p) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
  set('eval-job-title', p.title);
  set('eval-req-skills', p.reqSkills);
  set('eval-pref-skills', p.prefSkills);
  set('eval-min-exp', p.minExp);
  set('eval-req-edu', p.edu);
  set('eval-job-desc', p.desc);
  set('eval-additional-req', p.additional);
}

async function runCandidateJobEvaluation() {
  const idA = document.getElementById('compare-select-a')?.value;
  const idB = document.getElementById('compare-select-b')?.value;
  const container = document.getElementById('comparison-results-container');
  const btn = document.getElementById('btn-eval-candidates');

  if (!idA || !idB || idA === idB) {
    alert('Please select two distinct candidates to evaluate.');
    return;
  }

  const jobTitle = document.getElementById('eval-job-title')?.value?.trim();
  const reqSkills = document.getElementById('eval-req-skills')?.value?.trim();
  if (!jobTitle || !reqSkills) {
    alert('Please enter at least a Job Title and Required Skills.');
    return;
  }

  const payload = {
    candidate_id_a: idA,
    candidate_id_b: idB,
    job_title: jobTitle,
    job_description: document.getElementById('eval-job-desc')?.value?.trim() || '',
    required_skills: reqSkills.split(',').map(s => s.trim()).filter(Boolean),
    preferred_skills: (document.getElementById('eval-pref-skills')?.value || '').split(',').map(s => s.trim()).filter(Boolean),
    min_experience_years: parseFloat(document.getElementById('eval-min-exp')?.value || '0') || 0,
    required_education: document.getElementById('eval-req-edu')?.value?.trim() || '',
    additional_requirements: document.getElementById('eval-additional-req')?.value?.trim() || '',
  };

  try {
    if (btn) { btn.disabled = true; btn.classList.add('opacity-60', 'cursor-not-allowed'); }
    container.innerHTML = `
      <div class="bg-white p-12 rounded-xl border border-slate-200 text-center space-y-3">
        <div class="animate-spin w-8 h-8 border-2 border-brand-600 border-t-transparent rounded-full mx-auto"></div>
        <p class="text-sm text-slate-500 font-medium">Running AI-powered multi-dimensional evaluation…</p>
        <p class="text-xs text-slate-400">Analyzing skills, experience, education, projects, and certifications</p>
      </div>`;

    const res = await apiFetch(`${API_BASE}/v1/matching/evaluate-candidates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await parseApiResponse(res);
    renderJobEvaluationResults(data);
  } catch (err) {
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl space-y-2 text-center">
        <div class="font-bold text-sm">Evaluation Error</div>
        <p class="text-xs text-rose-600">${err.message}</p>
        <button onclick="runCandidateJobEvaluation()" class="mt-2 px-3 py-1.5 bg-rose-600 text-white text-xs font-semibold rounded-lg hover:bg-rose-700 transition">Retry</button>
      </div>`;
  } finally {
    if (btn) { btn.disabled = false; btn.classList.remove('opacity-60', 'cursor-not-allowed'); }
  }
}

function renderJobEvaluationResults(data) {
  const container = document.getElementById('comparison-results-container');
  if (!container) return;

  const overall = data.overall_recommendation || {};
  const ca = data.candidate_a || {};
  const cb = data.candidate_b || {};
  const aiRec = data.ai_recommendation || {};
  const details = data.detailed_comparison || [];
  const jobDet = data.job_details || {};

  const scoreA = overall.candidate_a_score ?? ca.overall_score ?? 0;
  const scoreB = overall.candidate_b_score ?? cb.overall_score ?? 0;
  const nameA = ca.name || 'Candidate A';
  const nameB = cb.name || 'Candidate B';

  const matchCategoryColor = (cat) => ({
    'Excellent Match': 'bg-emerald-50 text-emerald-800 border-emerald-300',
    'Strong Match': 'bg-sky-50 text-sky-800 border-sky-300',
    'Moderate Match': 'bg-amber-50 text-amber-800 border-amber-300',
    'Low Match': 'bg-rose-50 text-rose-800 border-rose-300',
    'Insufficient Evidence': 'bg-slate-100 text-slate-600 border-slate-300',
  })[cat] || 'bg-slate-100 text-slate-600 border-slate-300';

  const confidenceColor = (c) => ({
    'High': 'bg-emerald-100 text-emerald-800',
    'Moderate': 'bg-amber-100 text-amber-800',
    'Low': 'bg-rose-100 text-rose-800',
  })[c] || 'bg-slate-100 text-slate-700';

  const scoreBar = (score) => {
    const pct = Math.min(100, Math.max(0, score));
    const color = pct >= 75 ? 'bg-emerald-500' : pct >= 55 ? 'bg-sky-500' : pct >= 40 ? 'bg-amber-500' : 'bg-rose-400';
    return `<div class="w-full h-2 bg-slate-200 rounded-full mt-1 overflow-hidden"><div class="${color} h-full rounded-full transition-all" style="width:${pct}%"></div></div>`;
  };

  const tagList = (items, colorClass) =>
    items && items.length
      ? items.map(s => `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold border ${colorClass}">${s}</span>`).join('')
      : '<span class="text-slate-400 italic text-[11px]">None</span>';

  const dimensionRows = details.map(d => `
    <tr class="border-b border-slate-100 hover:bg-slate-50/60 transition">
      <td class="py-3 px-4">
        <div class="font-semibold text-slate-800 text-xs">${d.criterion}</div>
        <div class="text-[11px] text-slate-400">${d.weight_percentage} weight</div>
      </td>
      <td class="py-3 px-4">
        <div class="text-sm font-bold ${d.candidate_a.score >= 70 ? 'text-emerald-700' : d.candidate_a.score >= 45 ? 'text-amber-700' : 'text-rose-600'}">${d.candidate_a.score}%</div>
        <div class="text-[11px] text-slate-500 mt-0.5 leading-relaxed">${d.candidate_a.explanation}</div>
      </td>
      <td class="py-3 px-4">
        <div class="text-sm font-bold ${d.candidate_b.score >= 70 ? 'text-emerald-700' : d.candidate_b.score >= 45 ? 'text-amber-700' : 'text-rose-600'}">${d.candidate_b.score}%</div>
        <div class="text-[11px] text-slate-500 mt-0.5 leading-relaxed">${d.candidate_b.explanation}</div>
      </td>
      <td class="py-3 px-4 text-center text-xs font-bold ${d.candidate_a.score > d.candidate_b.score ? 'text-emerald-600' : d.candidate_b.score > d.candidate_a.score ? 'text-sky-600' : 'text-slate-400'}">
        ${d.candidate_a.score > d.candidate_b.score ? nameA : (d.candidate_b.score > d.candidate_a.score ? nameB : 'Tie')}
      </td>
    </tr>
  `).join('');

  const isWinnerA = scoreA > scoreB;
  const isWinnerB = scoreB > scoreA;
  const isTie = !isWinnerA && !isWinnerB;

  container.innerHTML = `
    <div class="space-y-5">

      <!-- Overall Recommendation Card -->
      <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="px-5 py-3.5 border-b border-slate-200 bg-gradient-to-r from-brand-50 to-sky-50 flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <i data-lucide="award" class="w-4 h-4 text-brand-600"></i>
            <h3 class="text-sm font-bold text-slate-900">AI Evaluation Recommendation — ${jobDet.title || payload?.job_title || 'Job Role'}</h3>
          </div>
          <span class="px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${confidenceColor(overall.recommendation_confidence)} border-current">
            ${overall.recommendation_confidence || ''} Confidence
          </span>
        </div>

        <div class="p-5 grid grid-cols-1 md:grid-cols-3 gap-4">
          <!-- Candidate A Score -->
          <div class="p-4 rounded-xl border-2 ${isWinnerA ? 'border-emerald-400 bg-emerald-50/50' : 'border-slate-200 bg-slate-50/50'} space-y-1 text-center">
            ${isWinnerA ? '<div class="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-1">✓ Recommended</div>' : ''}
            <div class="text-2xl font-extrabold ${isWinnerA ? 'text-emerald-700' : 'text-slate-700'}">${scoreA}%</div>
            <div class="font-bold text-slate-900 text-sm">${nameA}</div>
            <span class="inline-block px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${matchCategoryColor(overall.candidate_a_category)}">${overall.candidate_a_category || ca.match_category || ''}</span>
            ${scoreBar(scoreA)}
          </div>

          <!-- Verdict -->
          <div class="p-4 rounded-xl border border-slate-200 bg-white flex flex-col items-center justify-center text-center space-y-2">
            <i data-lucide="${isTie ? 'scale' : 'git-compare'}" class="w-6 h-6 text-brand-600"></i>
            <div class="text-xs font-bold text-slate-500 uppercase tracking-wider">Recommended</div>
            <div class="text-sm font-extrabold text-brand-800">${overall.recommended_candidate || 'See below'}</div>
            ${overall.score_margin !== undefined && overall.score_margin !== null
              ? `<div class="text-[11px] text-slate-500">Margin: <strong>${overall.score_margin}%</strong></div>`
              : ''}
          </div>

          <!-- Candidate B Score -->
          <div class="p-4 rounded-xl border-2 ${isWinnerB ? 'border-emerald-400 bg-emerald-50/50' : 'border-slate-200 bg-slate-50/50'} space-y-1 text-center">
            ${isWinnerB ? '<div class="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-1">✓ Recommended</div>' : ''}
            <div class="text-2xl font-extrabold ${isWinnerB ? 'text-emerald-700' : 'text-slate-700'}">${scoreB}%</div>
            <div class="font-bold text-slate-900 text-sm">${nameB}</div>
            <span class="inline-block px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${matchCategoryColor(overall.candidate_b_category)}">${overall.candidate_b_category || cb.match_category || ''}</span>
            ${scoreBar(scoreB)}
          </div>
        </div>

        <!-- Summary -->
        ${overall.summary ? `<div class="px-5 pb-4"><div class="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-700 leading-relaxed"><i data-lucide="info" class="w-3.5 h-3.5 text-brand-600 inline mr-1"></i>${overall.summary}</div></div>` : ''}
      </div>

      <!-- Dimension Score Table -->
      <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="px-5 py-3.5 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider">Multi-Dimensional Evaluation Breakdown</h4>
          <span class="text-xs text-slate-400">Weighted scoring across 6 dimensions</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase">
              <tr>
                <th class="py-2.5 px-4">Dimension / Weight</th>
                <th class="py-2.5 px-4">${nameA}</th>
                <th class="py-2.5 px-4">${nameB}</th>
                <th class="py-2.5 px-4 text-center">Edge</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100">${dimensionRows}</tbody>
          </table>
        </div>
      </div>

      <!-- Strengths & Gaps Side-by-Side -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
        <!-- Candidate A -->
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
          <h4 class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center space-x-1.5">
            <span class="w-5 h-5 rounded-full ${isWinnerA ? 'bg-emerald-500' : 'bg-brand-200'} text-white flex items-center justify-center text-[10px] font-black">A</span>
            <span>${nameA}</span>
          </h4>
          <div>
            <div class="text-[11px] font-bold text-emerald-700 uppercase tracking-wider mb-1">Matching Skills</div>
            <div class="flex flex-wrap gap-1">${tagList(ca.matching_skills, 'bg-emerald-100 text-emerald-800 border-emerald-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-rose-700 uppercase tracking-wider mb-1">Missing Required Skills</div>
            <div class="flex flex-wrap gap-1">${tagList(ca.missing_required_skills, 'bg-rose-100 text-rose-800 border-rose-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-sky-700 uppercase tracking-wider mb-1">Strengths</div>
            <div class="flex flex-wrap gap-1">${tagList(ca.strengths, 'bg-sky-100 text-sky-800 border-sky-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-amber-700 uppercase tracking-wider mb-1">Gaps</div>
            <div class="flex flex-wrap gap-1">${tagList(ca.gaps, 'bg-amber-100 text-amber-800 border-amber-300')}</div>
          </div>
          ${ca.experience_summary ? `<div class="p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600"><span class="font-bold">Experience:</span> ${ca.experience_summary}</div>` : ''}
          ${ca.education_summary ? `<div class="p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600"><span class="font-bold">Education:</span> ${ca.education_summary}</div>` : ''}
        </div>

        <!-- Candidate B -->
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
          <h4 class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center space-x-1.5">
            <span class="w-5 h-5 rounded-full ${isWinnerB ? 'bg-emerald-500' : 'bg-brand-200'} text-white flex items-center justify-center text-[10px] font-black">B</span>
            <span>${nameB}</span>
          </h4>
          <div>
            <div class="text-[11px] font-bold text-emerald-700 uppercase tracking-wider mb-1">Matching Skills</div>
            <div class="flex flex-wrap gap-1">${tagList(cb.matching_skills, 'bg-emerald-100 text-emerald-800 border-emerald-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-rose-700 uppercase tracking-wider mb-1">Missing Required Skills</div>
            <div class="flex flex-wrap gap-1">${tagList(cb.missing_required_skills, 'bg-rose-100 text-rose-800 border-rose-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-sky-700 uppercase tracking-wider mb-1">Strengths</div>
            <div class="flex flex-wrap gap-1">${tagList(cb.strengths, 'bg-sky-100 text-sky-800 border-sky-300')}</div>
          </div>
          <div>
            <div class="text-[11px] font-bold text-amber-700 uppercase tracking-wider mb-1">Gaps</div>
            <div class="flex flex-wrap gap-1">${tagList(cb.gaps, 'bg-amber-100 text-amber-800 border-amber-300')}</div>
          </div>
          ${cb.experience_summary ? `<div class="p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600"><span class="font-bold">Experience:</span> ${cb.experience_summary}</div>` : ''}
          ${cb.education_summary ? `<div class="p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600"><span class="font-bold">Education:</span> ${cb.education_summary}</div>` : ''}
        </div>
      </div>

      <!-- AI Recommendation Q&A Panel -->
      <div class="bg-white rounded-xl border border-brand-200 shadow-sm overflow-hidden">
        <div class="px-5 py-3.5 border-b border-brand-200 bg-gradient-to-r from-brand-50 to-sky-50 flex items-center space-x-2">
          <i data-lucide="sparkles" class="w-4 h-4 text-brand-600"></i>
          <h4 class="text-xs font-bold text-brand-900 uppercase tracking-wider">AI Hiring Intelligence — Answers to Key Questions</h4>
        </div>
        <div class="p-5 space-y-3 text-xs">
          ${[
            ['Which candidate is more suitable?', aiRec.which_candidate_more_suitable],
            ['Why are they more suitable?', aiRec.why_more_suitable],
            ['Requirements satisfied by recommended candidate', Array.isArray(aiRec.requirements_satisfied_by_winner) ? aiRec.requirements_satisfied_by_winner.join(', ') || 'None identified' : (aiRec.requirements_satisfied_by_winner || 'N/A')],
            ['Important requirements still missing', Array.isArray(aiRec.important_requirements_missing) ? aiRec.important_requirements_missing.join(', ') || 'None' : (aiRec.important_requirements_missing || 'None')],
            ['Where does the other candidate excel?', aiRec.where_other_candidate_excels],
            ['Are both equally suitable?', aiRec.are_both_equally_suitable ? 'Yes — both candidates are within tie threshold.' : 'No — a clear recommendation exists.'],
            ['Is either candidate unsuitable?', aiRec.is_either_unsuitable],
          ].map(([q, a]) => `
            <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg">
              <div class="font-bold text-slate-800 mb-1 flex items-center space-x-1.5">
                <i data-lucide="help-circle" class="w-3 h-3 text-brand-500 shrink-0"></i>
                <span>${q}</span>
              </div>
              <div class="text-slate-600 leading-relaxed pl-4">${a || 'N/A'}</div>
            </div>
          `).join('')}
          ${aiRec.synthesis ? `
            <div class="p-3 bg-brand-50 border border-brand-200 rounded-lg">
              <div class="font-bold text-brand-900 mb-1 flex items-center space-x-1.5">
                <i data-lucide="file-text" class="w-3 h-3 text-brand-600 shrink-0"></i>
                <span>Executive Summary</span>
              </div>
              <div class="text-brand-800 leading-relaxed pl-4">${aiRec.synthesis}</div>
            </div>` : ''}
        </div>
      </div>

    </div>
  `;

  if (window.lucide) lucide.createIcons();
}

// ── 6. Conflict Resolution Workspace ───────────────────────────────────────
async function loadConflictDecisions() {
  const container = document.getElementById('conflicts-container');
  if (!container) return;

  try {
    const res = await apiFetch(`${API_BASE}/v1/conflict/decisions`);
    if (!res.ok) return;
    const data = await res.json();

    const items = data.items || [];
    if (items.length === 0) {
      container.innerHTML = `
        <div class="bg-white p-8 rounded-xl border border-slate-200 text-center text-slate-400">
          No field-level conflicts logged in recent batch runs.
        </div>
      `;
      return;
    }

    container.innerHTML = items.map(d => `
      <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
        <div class="flex items-center justify-between">
          <div class="font-bold text-slate-800 text-sm">Field Conflict: <span class="font-mono text-brand-700">${d.field_name}</span></div>
          <span class="px-2.5 py-0.5 rounded text-xs font-semibold bg-slate-100 text-slate-700">
            Strategy: ${d.resolution_method}
          </span>
        </div>
        <div class="grid grid-cols-2 gap-4 text-xs">
          <div class="p-3 bg-emerald-50 rounded-lg border border-emerald-200">
            <div class="font-bold text-emerald-800 mb-1">Selected Winning Value</div>
            <div>${JSON.stringify(d.selected_value)}</div>
          </div>
          <div class="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div class="font-bold text-slate-600 mb-1">Rejected Alternative(s)</div>
            <div>${JSON.stringify(d.rejected_values)}</div>
          </div>
        </div>
        <div class="text-xs text-slate-500 italic">Resolution explanation: ${d.explanation || 'Automatically resolved via pipeline policy'}</div>
      </div>
    `).join('');
  } catch (err) {
    console.error('Failed to load conflicts:', err);
  }
}

// ── 7. Document Evidence Viewer ────────────────────────────────────────────
async function inspectDocumentEvidence() {
  const input = document.getElementById('evidence-file-input');
  const output = document.getElementById('evidence-viewer-output');
  if (!input.files[0]) {
    alert('Please select a resume file (PDF, DOCX, TXT) to inspect.');
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);

  try {
    output.classList.remove('hidden');
    output.innerHTML = '<div class="text-center p-8 text-slate-400">Running Extractor-Validator-Refiner on document...</div>';

    const res = await apiFetch(`${API_BASE}/v1/extraction/upload-and-extract`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Evidence inspection failed.');
    const data = await res.json();

    const spansHtml = Object.entries(data.evidence_spans || {}).map(([key, span]) => `
      <div class="p-3 bg-white border border-slate-200 rounded-lg shadow-sm space-y-1">
        <div class="flex items-center justify-between text-xs">
          <span class="font-bold text-brand-700 font-mono">${key}</span>
          <span class="text-slate-400">Chars: [${span.start} - ${span.end}] | Page ${span.page || 1}</span>
        </div>
        <div class="text-xs font-mono bg-slate-50 p-2 rounded border border-slate-100 text-slate-700">"...${span.text}..."</div>
      </div>
    `).join('') || '<div class="text-slate-400">No evidence spans found</div>';

    output.innerHTML = `
      <div class="space-y-4">
        <div class="bg-brand-50 border border-brand-200 rounded-xl p-4 text-xs text-brand-900 flex items-center justify-between">
          <div>
            <span class="font-bold">Extraction Self-Correction Summary:</span>
            <span> ${data.correction_attempts} bounded correction attempts performed. Status: ${data.requires_human_review ? 'Pending Review' : 'Verified Consistent'}.</span>
          </div>
        </div>

        <div class="space-y-2">
          <h4 class="text-sm font-bold text-slate-800">Anchored Evidence Snippets</h4>
          <div class="space-y-2">${spansHtml}</div>
        </div>
      </div>
    `;
  } catch (err) {
    output.innerHTML = `<div class="text-rose-600 p-8 text-center">${err.message}</div>`;
  }
}

// ── 8. Job Description Matching & Candidate Comparison ───────────────────────

let currentJDMatchData = null;
let currentJDId = null;

function toggleJDInputMode(mode) {
  const textContainer = document.getElementById('jd-input-text-container');
  const fileContainer = document.getElementById('jd-input-file-container');
  const textBtn = document.getElementById('jd-mode-text-btn');
  const fileBtn = document.getElementById('jd-mode-file-btn');

  if (mode === 'file') {
    if (textContainer) textContainer.classList.add('hidden');
    if (fileContainer) fileContainer.classList.remove('hidden');
    if (fileBtn) fileBtn.className = 'px-2.5 py-1 rounded-md bg-brand-50 text-brand-700 font-semibold border border-brand-200';
    if (textBtn) textBtn.className = 'px-2.5 py-1 rounded-md text-slate-600 hover:text-slate-900';
  } else {
    if (fileContainer) fileContainer.classList.add('hidden');
    if (textContainer) textContainer.classList.remove('hidden');
    if (textBtn) textBtn.className = 'px-2.5 py-1 rounded-md bg-brand-50 text-brand-700 font-semibold border border-brand-200';
    if (fileBtn) fileBtn.className = 'px-2.5 py-1 rounded-md text-slate-600 hover:text-slate-900';
  }
}

async function loadJobDescriptions() {
  // Can load saved JDs if needed
}

async function createAndMatchJD() {
  const title = document.getElementById('jd-title-input').value;
  const content = document.getElementById('jd-content-input').value;
  const container = document.getElementById('jd-results-container');
  const btn = document.querySelector('#jd-input-text-container button');

  if (!title.trim() || !content.trim()) {
    alert('Please enter a Job Title and Description.');
    return;
  }

  try {
    if (btn) {
      btn.disabled = true;
      btn.classList.add('opacity-60', 'cursor-not-allowed');
    }
    container.innerHTML = '<div class="text-center p-12 text-slate-400"><div class="animate-spin w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full mx-auto mb-2"></div>Parsing job requirements and matching candidate profiles...</div>';

    // 1. Create JD
    const jdRes = await apiFetch(`${API_BASE}/v1/jobs/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content })
    });
    const jd = await parseApiResponse(jdRes);

    // 2. Match candidates
    const matchRes = await apiFetch(`${API_BASE}/v1/jobs/${jd.id}/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ min_score: 0.0 })
    });
    const matchData = await parseApiResponse(matchRes);
    renderJDResults(jd, matchData);

  } catch (err) {
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl space-y-2">
        <div class="font-bold flex items-center space-x-2">
          <span>⚠️ Job Description Matching Error</span>
        </div>
        <p class="text-xs text-rose-600">${err.message}</p>
        <button onclick="createAndMatchJD()" class="mt-2 px-3 py-1.5 bg-rose-600 text-white text-xs font-semibold rounded-lg hover:bg-rose-700 transition">
          Retry Matching
        </button>
      </div>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('opacity-60', 'cursor-not-allowed');
    }
  }
}

async function uploadAndMatchJD() {
  const fileInput = document.getElementById('jd-file-input');
  const titleInput = document.getElementById('jd-file-title-input');
  const container = document.getElementById('jd-results-container');
  const btn = document.querySelector('#jd-input-file-container button');

  if (!fileInput.files || !fileInput.files[0]) {
    alert('Please select a Job Description file (.pdf, .docx, .txt, .md).');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  if (titleInput && titleInput.value.trim()) {
    formData.append('title', titleInput.value.trim());
  }

  try {
    if (btn) {
      btn.disabled = true;
      btn.classList.add('opacity-60', 'cursor-not-allowed');
    }
    container.innerHTML = '<div class="text-center p-12 text-slate-400"><div class="animate-spin w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full mx-auto mb-2"></div>Uploading, extracting text, and parsing requirements...</div>';

    const uploadRes = await apiFetch(`${API_BASE}/v1/jobs/upload`, {
      method: 'POST',
      body: formData,
    });
    const jd = await parseApiResponse(uploadRes);

    // Match candidates
    const matchRes = await apiFetch(`${API_BASE}/v1/jobs/${jd.id}/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ min_score: 0.0 })
    });
    const matchData = await parseApiResponse(matchRes);
    renderJDResults(jd, matchData);

  } catch (err) {
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 text-rose-700 p-6 rounded-xl space-y-2">
        <div class="font-bold flex items-center space-x-2">
          <span>⚠️ Job Description Matching Error</span>
        </div>
        <p class="text-xs text-rose-600">${err.message}</p>
        <button onclick="uploadAndMatchJD()" class="mt-2 px-3 py-1.5 bg-rose-600 text-white text-xs font-semibold rounded-lg hover:bg-rose-700 transition">
          Retry Upload
        </button>
      </div>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('opacity-60', 'cursor-not-allowed');
    }
  }
}

function renderJDResults(jd, matchData) {
  currentJDId = jd.id;
  currentJDMatchData = matchData;
  const container = document.getElementById('jd-results-container');
  if (!container) return;

  const rankings = matchData.rankings || [];

  if (rankings.length === 0) {
    container.innerHTML = `
      <div class="bg-white rounded-xl border border-slate-200 p-8 text-center text-slate-400">
        No candidate profiles found in the system to evaluate. Ingest source resumes first.
      </div>
    `;
    return;
  }

  const reqCount = (jd.extracted_requirements || []).length;

  // Comparison Options for Head-to-Head
  const candidateOptions = rankings.map(r => 
    `<option value="${r.candidate_id}">${r.name} (${r.overall_score}% - ${r.fit_level})</option>`
  ).join('');

  container.innerHTML = `
    <div class="space-y-6">
      <!-- Executive Summary Card -->
      <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-100">
          <div>
            <div class="flex items-center space-x-2">
              <span class="text-xs font-semibold px-2 py-0.5 rounded bg-brand-50 text-brand-700 border border-brand-200">Active Evaluation</span>
              <h3 class="text-base font-bold text-slate-900">${matchData.job_title}</h3>
            </div>
            <p class="text-xs text-slate-500 mt-1">${matchData.total_evaluated} candidate(s) evaluated against ${reqCount} extracted requirements</p>
          </div>
          <div class="flex items-center space-x-2 text-xs">
            <span class="px-2.5 py-1 font-semibold rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200">Top Match: ${rankings[0].name} (${rankings[0].overall_score}%)</span>
          </div>
        </div>

        <!-- Head-to-Head Comparison Bar -->
        <div class="mt-4 pt-3 border-t border-slate-100 bg-slate-50 -mx-5 -mb-5 p-4 rounded-b-xl">
          <div class="text-xs font-bold text-slate-700 mb-2 flex items-center space-x-1.5">
            <i data-lucide="git-compare" class="w-3.5 h-3.5 text-brand-600"></i>
            <span>Head-to-Head Candidate Comparison Matrix</span>
          </div>
          <div class="flex flex-col sm:flex-row items-center gap-2">
            <select id="jd-compare-sel-a" class="w-full sm:w-1/2 text-xs px-3 py-1.5 border border-slate-300 rounded-lg bg-white focus:outline-none focus:ring-1 focus:ring-brand-500">
              ${candidateOptions}
            </select>
            <span class="text-xs font-bold text-slate-400">VS</span>
            <select id="jd-compare-sel-b" class="w-full sm:w-1/2 text-xs px-3 py-1.5 border border-slate-300 rounded-lg bg-white focus:outline-none focus:ring-1 focus:ring-brand-500">
              ${candidateOptions}
            </select>
            <button onclick="executeJDHeadToHeadComparison()" class="w-full sm:w-auto px-4 py-1.5 bg-brand-600 text-white rounded-lg text-xs font-semibold hover:bg-brand-700 transition shadow whitespace-nowrap">
              Compare Head-to-Head
            </button>
          </div>
          <div id="head-to-head-result" class="mt-3 hidden"></div>
        </div>
      </div>

      <!-- Detailed Comparison & Ranking Table -->
      <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="px-5 py-3.5 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
          <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider">Candidate Match Rankings & Comparative Analysis</h4>
          <span class="text-xs text-slate-400">Click candidate row to toggle requirement breakdown</span>
        </div>
        <div class="divide-y divide-slate-100">
          ${rankings.map((r, idx) => {
            const fitClasses = {
              'Strong Fit': 'bg-emerald-50 text-emerald-700 border-emerald-200',
              'Moderate Fit': 'bg-sky-50 text-sky-700 border-sky-200',
              'Potential Fit': 'bg-amber-50 text-amber-700 border-amber-200',
              'Low Alignment': 'bg-slate-100 text-slate-600 border-slate-200',
            }[r.fit_level] || 'bg-slate-100 text-slate-600 border-slate-200';

            const scoreBarColor = r.overall_score >= 80 ? 'bg-emerald-500' : (r.overall_score >= 60 ? 'bg-sky-500' : (r.overall_score >= 40 ? 'bg-amber-500' : 'bg-slate-400'));

            return `
              <div class="p-4 hover:bg-slate-50/70 transition space-y-3">
                <div class="flex items-start justify-between gap-4">
                  <div class="flex items-start space-x-3">
                    <span class="w-7 h-7 rounded-full ${idx === 0 ? 'bg-amber-400 text-amber-950 font-black' : (idx === 1 ? 'bg-slate-300 text-slate-800 font-bold' : 'bg-slate-100 text-slate-600 font-bold')} flex items-center justify-center text-xs shrink-0 shadow-sm">
                      #${idx + 1}
                    </span>
                    <div>
                      <div class="flex items-center space-x-2">
                        <span class="font-bold text-slate-900 text-sm">${r.name}</span>
                        <span class="text-xs text-slate-400 font-mono">(${r.candidate_id})</span>
                        <span class="text-xs px-2 py-0.5 rounded-full font-semibold border ${fitClasses}">${r.fit_level}</span>
                      </div>
                      <div class="text-xs text-slate-500 mt-0.5 flex items-center space-x-3">
                        <span><strong>Experience:</strong> ${r.experience_years.toFixed(1)} yrs recorded</span>
                        <span>•</span>
                        <span><strong>Matched:</strong> ${(r.matched_requirements || []).length} / ${((r.matched_requirements || []).length + (r.missing_requirements || []).length)} requirements</span>
                      </div>
                    </div>
                  </div>

                  <div class="text-right shrink-0">
                    <div class="text-xl font-extrabold text-slate-900">${r.overall_score}%</div>
                    <div class="w-24 h-1.5 bg-slate-200 rounded-full mt-1 overflow-hidden">
                      <div class="${scoreBarColor} h-full rounded-full" style="width: ${r.overall_score}%"></div>
                    </div>
                  </div>
                </div>

                <!-- Comparative explanation: Why this candidate ranks here -->
                <div class="text-xs bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-slate-700 flex items-start space-x-2">
                  <i data-lucide="info" class="w-3.5 h-3.5 text-brand-600 shrink-0 mt-0.5"></i>
                  <div>
                    <span class="font-semibold text-slate-800">Ranking Rationale:</span>
                    <span class="text-slate-600"> ${r.why_better_match || r.summary}</span>
                  </div>
                </div>

                <!-- Matched Skills & Missing Required Skills -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  <div class="p-2.5 rounded-lg bg-emerald-50/60 border border-emerald-200">
                    <span class="font-bold text-emerald-900 text-[11px] block mb-1">Matched Skills:</span>
                    <div class="flex flex-wrap gap-1">
                      ${(r.matched_skills && r.matched_skills.length > 0) ? r.matched_skills.map(s => `<span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-semibold border border-emerald-300">${s}</span>`).join('') : '<span class="text-slate-400 italic text-[11px]">No skills matched</span>'}
                    </div>
                  </div>

                  <div class="p-2.5 rounded-lg bg-rose-50/60 border border-rose-200">
                    <span class="font-bold text-rose-900 text-[11px] block mb-1">Missing Required Skills:</span>
                    <div class="flex flex-wrap gap-1">
                      ${(r.missing_required_skills && r.missing_required_skills.length > 0) ? r.missing_required_skills.map(s => `<span class="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 text-[10px] font-semibold border border-rose-300">${s}</span>`).join('') : '<span class="text-emerald-700 italic text-[11px]">All required skills satisfied ✓</span>'}
                    </div>
                  </div>
                </div>

                <!-- Relevant Experience & Education Match -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span class="font-bold text-slate-700 text-[11px] block mb-0.5">Relevant Experience:</span>
                    <span class="text-slate-600">${r.relevant_experience || 'No recorded experience'}</span>
                  </div>
                  <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span class="font-bold text-slate-700 text-[11px] block mb-0.5">Education Match:</span>
                    <span class="text-slate-600">${r.education_match || 'No educational credentials on file'}</span>
                  </div>
                </div>

                <!-- Strengths & Gaps -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  <div class="p-2 rounded bg-slate-50 border border-slate-200">
                    <span class="font-semibold text-slate-700 text-[11px] block mb-1">Key Strengths:</span>
                    <div class="flex flex-wrap gap-1">
                      ${(r.strengths && r.strengths.length > 0) ? r.strengths.map(s => `<span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 text-[10px] font-medium">${s}</span>`).join('') : '<span class="text-slate-400 italic text-[11px]">None identified</span>'}
                    </div>
                  </div>

                  <div class="p-2 rounded bg-slate-50 border border-slate-200">
                    <span class="font-semibold text-slate-700 text-[11px] block mb-1">Identified Gaps:</span>
                    <div class="flex flex-wrap gap-1">
                      ${(r.gaps && r.gaps.length > 0) ? r.gaps.map(g => `<span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 text-[10px] font-medium">${g}</span>`).join('') : '<span class="text-emerald-600 italic text-[11px]">None</span>'}
                    </div>
                  </div>
                </div>

                <!-- Toggle Evidence Button -->
                <div>
                  <button onclick="toggleCandidateEvidenceRow('ev-row-${idx}')" class="text-xs font-semibold text-brand-600 hover:text-brand-800 flex items-center space-x-1">
                    <span>Toggle Requirement Evidence Breakdown</span>
                    <i data-lucide="chevron-down" class="w-3 h-3"></i>
                  </button>
                </div>

                <!-- Collapsible Evidence Sub-table -->
                <div id="ev-row-${idx}" class="hidden border border-slate-200 rounded-lg overflow-hidden mt-2">
                  <table class="w-full text-xs text-left">
                    <thead class="bg-slate-100 text-slate-600 border-b border-slate-200">
                      <tr>
                        <th class="px-3 py-2">Requirement</th>
                        <th class="px-3 py-2">Category</th>
                        <th class="px-3 py-2">Status</th>
                        <th class="px-3 py-2">Verified Candidate Evidence</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 bg-white">
                      ${[...(r.matched_requirements || []), ...(r.missing_requirements || [])].map(item => `
                        <tr>
                          <td class="px-3 py-2 font-medium text-slate-800">${item.requirement}</td>
                          <td class="px-3 py-2 text-slate-500">${item.category}</td>
                          <td class="px-3 py-2">
                            <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${item.status === 'verified' ? 'bg-emerald-100 text-emerald-800' : (item.status === 'partial' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-800')}">
                              ${item.status.toUpperCase()}
                            </span>
                          </td>
                          <td class="px-3 py-2 text-slate-600">${item.evidence || '--'}</td>
                        </tr>
                      `).join('')}
                    </tbody>
                  </table>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    </div>
  `;

  // Set second dropdown to second candidate if available
  const selB = document.getElementById('jd-compare-sel-b');
  if (selB && selB.children.length > 1) {
    selB.selectedIndex = 1;
  }

  if (window.lucide) lucide.createIcons();
}

function toggleCandidateEvidenceRow(rowId) {
  const el = document.getElementById(rowId);
  if (el) {
    el.classList.toggle('hidden');
  }
}

async function executeJDHeadToHeadComparison() {
  const selA = document.getElementById('jd-compare-sel-a');
  const selB = document.getElementById('jd-compare-sel-b');
  const resultDiv = document.getElementById('head-to-head-result');
  if (!selA || !selB || !resultDiv || !currentJDId) return;

  const idA = selA.value;
  const idB = selB.value;

  if (idA === idB) {
    alert('Please select two different candidates to compare.');
    return;
  }

  resultDiv.classList.remove('hidden');
  resultDiv.innerHTML = '<div class="text-center p-4 text-xs text-slate-500">Generating head-to-head comparison matrix...</div>';

  try {
    const res = await apiFetch(`${API_BASE}/v1/jobs/${currentJDId}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_id_a: idA, candidate_id_b: idB }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Comparison failed.');
    }
    const data = await res.json();

    const ca = data.candidate_a;
    const cb = data.candidate_b;

    resultDiv.innerHTML = `
      <div class="bg-white p-4 rounded-xl border border-brand-200 shadow-sm space-y-3">
        <!-- Winner Banner -->
        <div class="p-3 bg-gradient-to-r from-brand-50 to-sky-50 border border-brand-200 rounded-lg text-xs">
          <div class="font-bold text-brand-900 flex items-center space-x-1.5">
            <i data-lucide="award" class="w-4 h-4 text-brand-600"></i>
            <span>Head-to-Head Comparative Assessment:</span>
          </div>
          <p class="text-slate-700 mt-1 leading-relaxed">${data.comparison_explanation}</p>
        </div>

        <!-- Scores comparison -->
        <div class="grid grid-cols-2 gap-3 text-xs">
          <div class="p-2.5 rounded-lg border ${data.stronger_candidate_id === ca.id ? 'border-emerald-300 bg-emerald-50/40' : 'border-slate-200 bg-slate-50'}">
            <div class="font-bold text-slate-900">${ca.name}</div>
            <div class="text-slate-500 text-[11px]">Score: <strong>${ca.score}%</strong> (${ca.fit_level}) | Exp: <strong>${ca.experience_years.toFixed(1)} yrs</strong></div>
          </div>
          <div class="p-2.5 rounded-lg border ${data.stronger_candidate_id === cb.id ? 'border-emerald-300 bg-emerald-50/40' : 'border-slate-200 bg-slate-50'}">
            <div class="font-bold text-slate-900">${cb.name}</div>
            <div class="text-slate-500 text-[11px]">Score: <strong>${cb.score}%</strong> (${cb.fit_level}) | Exp: <strong>${cb.experience_years.toFixed(1)} yrs</strong></div>
          </div>
        </div>

        <!-- Matrix table -->
        <div class="overflow-x-auto border border-slate-200 rounded-lg">
          <table class="w-full text-xs text-left">
            <thead class="bg-slate-100 text-slate-600 border-b border-slate-200">
              <tr>
                <th class="px-3 py-2">Requirement</th>
                <th class="px-3 py-2">${ca.name} Evidence</th>
                <th class="px-3 py-2">${cb.name} Evidence</th>
                <th class="px-3 py-2 text-center">Relative Advantage</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 bg-white">
              ${(data.matrix || []).map(m => {
                const advText = m.advantage === ca.id ? ca.name : (m.advantage === cb.id ? cb.name : 'Tie / Neutral');
                const advClass = m.advantage === ca.id || m.advantage === cb.id ? 'text-emerald-700 font-bold' : 'text-slate-400';
                return `
                  <tr>
                    <td class="px-3 py-2 font-medium text-slate-800">${m.requirement}</td>
                    <td class="px-3 py-2 text-slate-600">${m.candidate_a.evidence}</td>
                    <td class="px-3 py-2 text-slate-600">${m.candidate_b.evidence}</td>
                    <td class="px-3 py-2 text-center text-[11px] ${advClass}">${advText}</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

  } catch (err) {
    resultDiv.innerHTML = `<div class="p-3 bg-rose-50 border border-rose-200 text-rose-700 rounded-lg text-xs">${err.message}</div>`;
  }
}

// ── Export Triggers ────────────────────────────────────────────────────────
function exportData(format) {
  const maskParam = currentRole === 'recruiter' ? 'mask_pii=true' : 'mask_pii=false';
  window.open(`${API_BASE}/v1/export/${format}?${maskParam}`, '_blank');
}
