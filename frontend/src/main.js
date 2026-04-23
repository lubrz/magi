/**
 * TRIAD Frontend Core
 */

import { TriadWS } from './lib/websocket.js';
import { typeWriter, setPulseSpeed } from './lib/animations.js';

// --- State ---
let isDeliberating = false;
let currentDeliberationId = null;

// --- Elements ---
const terminalInput = document.getElementById('terminal-input');
const terminalSubmit = document.getElementById('terminal-submit');
const logBody = document.getElementById('log-body');
const logRound = document.getElementById('log-round');
const consensusBar = document.getElementById('consensus-bar');
const consensusLabel = document.getElementById('consensus-label');
const consensusDetail = document.getElementById('consensus-detail');

const agentPanels = {
  axiom: {
    status: document.getElementById('axiom-status'),
    body: document.getElementById('axiom-response'),
    confFill: document.getElementById('axiom-confidence'),
    confText: document.getElementById('axiom-conf-text'),
    sources: document.getElementById('axiom-sources'),
    vote: document.getElementById('vote-axiom')
  },
  prism: {
    status: document.getElementById('prism-status'),
    body: document.getElementById('prism-response'),
    confFill: document.getElementById('prism-confidence'),
    confText: document.getElementById('prism-conf-text'),
    sources: document.getElementById('prism-sources'),
    vote: document.getElementById('vote-prism')
  },
  forge: {
    status: document.getElementById('forge-status'),
    body: document.getElementById('forge-response'),
    confFill: document.getElementById('forge-confidence'),
    confText: document.getElementById('forge-conf-text'),
    sources: document.getElementById('forge-sources'),
    vote: document.getElementById('vote-forge')
  }
};

// --- Initialization ---

// Determine WS URL (prod vs dev)
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = import.meta.env.DEV 
  ? `${protocol}//${window.location.hostname}:8080/ws`
  : `${protocol}//${window.location.host}/ws`;

const triad = new TriadWS(wsUrl, handleMessage, handleWSError);
triad.connect();

// --- Event Handlers ---

terminalSubmit.addEventListener('click', submitQuery);
terminalInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') submitQuery();
});

// Handle User Votes (Deadlock)
Object.keys(agentPanels).forEach(agent => {
  agentPanels[agent].vote.addEventListener('click', () => submitVote(agent));
});

// --- Actions ---

function submitQuery() {
  const query = terminalInput.value.trim();
  if (!query || isDeliberating) return;

  resetUI();
  isDeliberating = true;
  terminalInput.disabled = true;
  terminalSubmit.disabled = true;
  setPulseSpeed('0.5s');

  addLogEntry(`QUERY INITIATED: "${query}"`, 'log-system');
  triad.ask(query);
  terminalInput.value = '';
}

async function submitVote(agent) {
  if (!currentDeliberationId) return;
  
  addLogEntry(`USER OVERRIDE: SELECTING ${agent.toUpperCase()} POSITION...`, 'log-system');
  
  try {
    const resp = await fetch('/api/vote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        deliberation_id: currentDeliberationId,
        selected_agent: agent
      })
    });
    const result = await resp.json();
    handleFinalAnswer(result);
  } catch (err) {
    addLogEntry(`VOTE FAILED: ${err.message}`, 'log-error');
  }
}

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
      addLogEntry(`EVALUATING CROSS-NODE AGREEMENT...`, 'log-system');
      consensusBar.style.display = 'block';
      consensusLabel.innerText = 'EVALUATING';
      consensusDetail.innerText = '';
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

function handleWSError(err) {
  addLogEntry(`CONNECTION FAILURE: UNABLE TO REACH CORE`, 'log-error');
  document.getElementById('system-status').innerText = 'SYSTEM OFFLINE';
  document.getElementById('system-status').className = 'system-status status-offline';
}

// --- UI Updaters ---

function resetUI() {
  consensusBar.style.display = 'none';
  Object.keys(agentPanels).forEach(a => {
    agentPanels[a].status.innerText = 'STANDBY';
    agentPanels[a].body.innerHTML = '<p class="placeholder-text">Awaiting query…</p>';
    agentPanels[a].confFill.style.width = '0%';
    agentPanels[a].confText.innerText = '—';
    agentPanels[a].sources.innerHTML = '';
    agentPanels[a].vote.style.display = 'none';
  });
}

function updateAgentStatus(agent, status) {
  if (agentPanels[agent]) {
    agentPanels[agent].status.innerText = status;
  }
}

async function handleAgentResponse(data) {
  const panel = agentPanels[data.agent];
  if (!panel) return;

  panel.status.innerText = 'RESPONSE RECEIVED';
  panel.confFill.style.width = `${data.confidence * 100}%`;
  panel.confText.innerText = Math.round(data.confidence * 100);

  // Typewriter effect for the position
  panel.body.innerHTML = `<p><strong>POSITION:</strong> ${data.position}</p><br/><p><strong>REASONING:</strong> ${data.reasoning}</p>`;
  
  // Update sources
  if (data.sources && data.sources.length > 0) {
    panel.sources.innerHTML = 'SOURCES: ' + data.sources.map(s => `<span class="source-tag">${s}</span>`).join(' ');
  }

  addLogEntry(`NODE ${data.agent.toUpperCase()} TRANSMITTED POSITION (CONF: ${Math.round(data.confidence * 100)}%)`, 'log-agent');
}

function handleAgentCritique(data) {
  addLogEntry(`NODE ${data.critic.toUpperCase()} CRITIQUED ${data.target.toUpperCase()} (AGREE: ${Math.round(data.agreement * 100)}%)`, 'log-agent');
}

function handleConsensusReached(data) {
  consensusLabel.innerText = data.status.toUpperCase();
  consensusDetail.innerText = `AGREEMENT: ${data.agreeing.join(', ').toUpperCase()}`;
  addLogEntry(`CONSENSUS REACHED: ${data.status.toUpperCase()}`, 'log-consensus');
}

function handleDeadlock(data) {
  consensusLabel.innerText = 'DEADLOCK';
  consensusLabel.style.color = 'var(--accent-prism)';
  consensusDetail.innerText = 'MAX ROUNDS EXCEEDED. NODES DIVERGENT.';
  addLogEntry(`DELIBERATION DEADLOCK: RESOLUTION REQUIRED`, 'log-error');
  
  // Show vote buttons
  Object.keys(agentPanels).forEach(a => {
    agentPanels[a].vote.style.display = 'block';
  });
}

function handleFinalAnswer(data) {
  currentDeliberationId = data.id;
  addLogEntry(`FINAL OUTPUT COMPILED (${data.duration_seconds}s)`, 'log-consensus');
  
  if (data.status === 'consensus' || data.status === 'user_decided') {
    addLogEntry(`STANCE: ${data.final_answer.substring(0, 100)}...`, 'log-system');
  }

  finishDeliberation();
}

function finishDeliberation() {
  isDeliberating = false;
  terminalInput.disabled = false;
  terminalSubmit.disabled = false;
  setPulseSpeed('2s');
  terminalInput.focus();
}

function addLogEntry(text, className) {
  const entry = document.createElement('p');
  entry.className = `log-entry ${className}`;
  const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  entry.innerText = `[${time}] ${text}`;
  logBody.appendChild(entry);
  logBody.scrollTop = logBody.scrollHeight;
}
