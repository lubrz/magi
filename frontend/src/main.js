/**
 * TRIAD Frontend Core
 */

import { TriadWS } from './lib/websocket.js';
import { setPulseSpeed } from './lib/animations.js';

// --- State ---
let isDeliberating = false;
let currentDeliberationId = null;

// --- Elements ---
const terminalInput   = document.getElementById('terminal-input');
const terminalSubmit  = document.getElementById('terminal-submit');
const logBody         = document.getElementById('log-body');
const logRound        = document.getElementById('log-round');
const consensusBar    = document.getElementById('consensus-bar');
const consensusLabel  = document.getElementById('consensus-label');
const consensusDetail = document.getElementById('consensus-detail');
const uploadPanel     = document.getElementById('upload-panel');
const uploadToggle    = document.getElementById('upload-toggle');
const uploadClose     = document.getElementById('upload-close');

const agentPanels = {
  axiom: {
    status:   document.getElementById('axiom-status'),
    body:     document.getElementById('axiom-response'),
    confFill: document.getElementById('axiom-confidence'),
    confText: document.getElementById('axiom-conf-text'),
    sources:  document.getElementById('axiom-sources'),
    vote:     document.getElementById('vote-axiom'),
  },
  prism: {
    status:   document.getElementById('prism-status'),
    body:     document.getElementById('prism-response'),
    confFill: document.getElementById('prism-confidence'),
    confText: document.getElementById('prism-conf-text'),
    sources:  document.getElementById('prism-sources'),
    vote:     document.getElementById('vote-prism'),
  },
  forge: {
    status:   document.getElementById('forge-status'),
    body:     document.getElementById('forge-response'),
    confFill: document.getElementById('forge-confidence'),
    confText: document.getElementById('forge-conf-text'),
    sources:  document.getElementById('forge-sources'),
    vote:     document.getElementById('vote-forge'),
  },
};

// ---------------------------------------------------------------------------
// Backend URL helpers
// ---------------------------------------------------------------------------

/**
 * Build the base HTTP URL for the backend API.
 *
 * In dev (Vite), requests to /api/* are proxied to http://localhost:8080/*.
 * In production (nginx), requests to /api/* are proxied the same way via
 * nginx.conf — so we always use the /api prefix.
 */
function apiUrl(path) {
  // path should start with '/', e.g. '/vote'
  return `/api${path}`;
}

/**
 * Build the WebSocket URL.
 *
 * Dev: Vite proxies /ws → ws://localhost:8080/ws
 * Prod: nginx proxies /ws → ws://backend:8080/ws
 * Both handled via /ws relative path.
 */
function wsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws`;
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

const triad = new TriadWS(wsUrl(), handleMessage, handleWSError);
triad.connect();

// ---------------------------------------------------------------------------
// Terminal
// ---------------------------------------------------------------------------

terminalSubmit.addEventListener('click', submitQuery);
terminalInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') submitQuery();
});

function submitQuery() {
  const query = terminalInput.value.trim();
  if (!query || isDeliberating) return;

  resetUI();
  isDeliberating = true;
  terminalInput.disabled  = true;
  terminalSubmit.disabled = true;
  setPulseSpeed('0.5s');

  addLogEntry(`QUERY INITIATED: "${query}"`, 'log-system');
  triad.ask(query);
  terminalInput.value = '';
}

// ---------------------------------------------------------------------------
// Vote (deadlock resolution)
// ---------------------------------------------------------------------------

Object.keys(agentPanels).forEach((agent) => {
  agentPanels[agent].vote.addEventListener('click', () => submitVote(agent));
});

async function submitVote(agent) {
  if (!currentDeliberationId) return;

  addLogEntry(`USER OVERRIDE: SELECTING ${agent.toUpperCase()} POSITION...`, 'log-system');

  try {
    // BUG FIX: original code used `/api/vote` in dev but the production nginx
    // had no /api proxy so this would 404.  Now consistently uses apiUrl()
    // which resolves correctly in both environments.
    const resp = await fetch(apiUrl('/vote'), {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        deliberation_id: currentDeliberationId,
        selected_agent:  agent,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || resp.statusText);
    }

    const result = await resp.json();
    handleFinalAnswer(result);
  } catch (err) {
    addLogEntry(`VOTE FAILED: ${err.message}`, 'log-error');
  }
}

// ---------------------------------------------------------------------------
// Upload panel
// ---------------------------------------------------------------------------

uploadToggle.addEventListener('click', () => {
  const visible = uploadPanel.style.display !== 'none';
  uploadPanel.style.display = visible ? 'none' : 'block';
});

uploadClose.addEventListener('click', () => {
  uploadPanel.style.display = 'none';
});

// Wire up file inputs and drag-and-drop for each agent zone
['axiom', 'prism', 'forge'].forEach((agent) => {
  const input  = document.getElementById(`upload-${agent}`);
  const label  = input.closest('label');
  const status = document.getElementById(`upload-status-${agent}`);

  // Click-to-upload
  input.addEventListener('change', async () => {
    if (input.files && input.files[0]) {
      await handleUpload(agent, input.files[0], status);
      input.value = ''; // reset so the same file can be re-uploaded if needed
    }
  });

  // Drag-and-drop
  label.addEventListener('dragover', (e) => {
    e.preventDefault();
    label.classList.add('drag-over');
  });

  label.addEventListener('dragleave', () => {
    label.classList.remove('drag-over');
  });

  label.addEventListener('drop', async (e) => {
    e.preventDefault();
    label.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      await handleUpload(agent, file, status);
    }
  });
});

/**
 * Upload a single file to the given agent's knowledge base.
 */
async function handleUpload(agent, file, statusEl) {
  const allowedExts = ['.txt', '.md', '.pdf'];
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();

  if (!allowedExts.includes(ext)) {
    setUploadStatus(statusEl, `✖ UNSUPPORTED FORMAT: ${ext}`, 'error');
    return;
  }

  setUploadStatus(statusEl, `▸ UPLOADING ${file.name}…`, 'uploading');
  addLogEntry(`INGESTING '${file.name}' → ${agent.toUpperCase()} KNOWLEDGE GRAPH`, 'log-system');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch(apiUrl(`/upload/${agent}`), {
      method: 'POST',
      body:   formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error || resp.statusText);
    }

    const msg = `✔ ${data.concepts_created} CONCEPT(S) INGESTED`;
    setUploadStatus(statusEl, msg, 'success');
    addLogEntry(
      `INGESTION COMPLETE: ${data.concepts_created} concept(s) added to ${agent.toUpperCase()} — ${data.concept_names.slice(0, 3).join(', ')}${data.concept_names.length > 3 ? '…' : ''}`,
      'log-consensus',
    );
  } catch (err) {
    setUploadStatus(statusEl, `✖ ERROR: ${err.message}`, 'error');
    addLogEntry(`INGESTION FAILED: ${err.message}`, 'log-error');
  }
}

function setUploadStatus(el, text, type) {
  el.textContent = text;
  el.className   = `upload-status ${type}`;
}

// ---------------------------------------------------------------------------
// WebSocket message handler
// ---------------------------------------------------------------------------

function handleMessage(event) {
  const { type, data } = event;

  switch (type) {
    case 'system_status':
      addLogEntry(data.message, 'log-system');
      break;

    case 'round_start':
      logRound.innerText = `ROUND ${data.round}/${data.max_rounds}`;
      addLogEntry(`ROUND ${data.round} DELIBERATION COMMENCED`, 'log-system');
      break;

    case 'agent_thinking':
      updateAgentStatus(data.agent, 'THINKING...');
      break;

    case 'agent_response':
      handleAgentResponse(data);
      break;

    case 'agent_critique':
      handleAgentCritique(data);
      break;

    case 'consensus_check':
      addLogEntry('EVALUATING CROSS-NODE AGREEMENT...', 'log-system');
      consensusBar.style.display = 'block';
      consensusLabel.innerText   = 'EVALUATING';
      consensusDetail.innerText  = '';
      break;

    case 'consensus_reached':
      handleConsensusReached(data);
      break;

    case 'deadlock':
      handleDeadlock(data);
      break;

    case 'final_answer':
      handleFinalAnswer(data);
      break;

    case 'error':
      addLogEntry(`CRITICAL ERROR: ${data.error}`, 'log-error');
      finishDeliberation();
      break;
  }
}

function handleWSError() {
  addLogEntry('CONNECTION FAILURE: UNABLE TO REACH CORE', 'log-error');
  document.getElementById('system-status').innerText = 'SYSTEM OFFLINE';
  document.getElementById('system-status').className = 'system-status status-offline';
}

// ---------------------------------------------------------------------------
// UI Updaters
// ---------------------------------------------------------------------------

function resetUI() {
  consensusBar.style.display = 'none';
  Object.keys(agentPanels).forEach((a) => {
    agentPanels[a].status.innerText   = 'STANDBY';
    agentPanels[a].body.innerHTML     = '<p class="placeholder-text">Awaiting query…</p>';
    agentPanels[a].confFill.style.width = '0%';
    agentPanels[a].confText.innerText = '—';
    agentPanels[a].sources.innerHTML  = '';
    agentPanels[a].vote.style.display = 'none';
  });
}

function updateAgentStatus(agent, status) {
  if (agentPanels[agent]) {
    agentPanels[agent].status.innerText = status;
  }
}

function handleAgentResponse(data) {
  const panel = agentPanels[data.agent];
  if (!panel) return;

  panel.status.innerText        = 'RESPONSE RECEIVED';
  panel.confFill.style.width    = `${data.confidence * 100}%`;
  panel.confText.innerText      = Math.round(data.confidence * 100);

  panel.body.innerHTML = `
    <p><strong>POSITION:</strong> ${escapeHtml(data.position)}</p>
    <br/>
    <p><strong>REASONING:</strong> ${escapeHtml(data.reasoning)}</p>
  `;

  if (data.sources && data.sources.length > 0) {
    panel.sources.innerHTML =
      'SOURCES: ' +
      data.sources
        .map((s) => `<span class="source-tag">${escapeHtml(s)}</span>`)
        .join(' ');
  }

  addLogEntry(
    `NODE ${data.agent.toUpperCase()} TRANSMITTED POSITION (CONF: ${Math.round(data.confidence * 100)}%)`,
    'log-agent',
  );
}

function handleAgentCritique(data) {
  addLogEntry(
    `NODE ${data.critic.toUpperCase()} CRITIQUED ${data.target.toUpperCase()} (AGREE: ${Math.round(data.agreement * 100)}%)`,
    'log-agent',
  );
}

function handleConsensusReached(data) {
  consensusLabel.innerText  = data.status.toUpperCase();
  consensusDetail.innerText = `AGREEMENT: ${data.agreeing.join(', ').toUpperCase()}`;
  addLogEntry(`CONSENSUS REACHED: ${data.status.toUpperCase()}`, 'log-consensus');
}

function handleDeadlock(data) {
  consensusLabel.innerText  = 'DEADLOCK';
  consensusLabel.style.color = 'var(--accent-prism)';
  consensusDetail.innerText  = 'MAX ROUNDS EXCEEDED. NODES DIVERGENT.';
  addLogEntry('DELIBERATION DEADLOCK: RESOLUTION REQUIRED', 'log-error');

  Object.keys(agentPanels).forEach((a) => {
    agentPanels[a].vote.style.display = 'block';
  });
}

function handleFinalAnswer(data) {
  currentDeliberationId = data.id;
  addLogEntry(`FINAL OUTPUT COMPILED (${data.duration_seconds}s)`, 'log-consensus');

  if (data.status === 'consensus' || data.status === 'user_decided') {
    const preview = (data.final_answer || '').substring(0, 120);
    addLogEntry(`STANCE: ${preview}${preview.length === 120 ? '…' : ''}`, 'log-system');
  }

  finishDeliberation();
}

function finishDeliberation() {
  isDeliberating          = false;
  terminalInput.disabled  = false;
  terminalSubmit.disabled = false;
  setPulseSpeed('2s');
  terminalInput.focus();
}

function addLogEntry(text, className) {
  const entry = document.createElement('p');
  entry.className = `log-entry ${className}`;
  const time = new Date().toLocaleTimeString([], {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  entry.innerText = `[${time}] ${text}`;
  logBody.appendChild(entry);
  logBody.scrollTop = logBody.scrollHeight;
}

/** Basic XSS guard for agent-supplied strings rendered as innerHTML. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
