// ── State ────────────────────────────────────────────────────────────
let currentFlow = null; // 'protect' | 'fto'

// ── Flow selection ───────────────────────────────────────────────────
function selectFlow(flow) {
  currentFlow = flow;

  const toolSection = document.getElementById('tool');
  const titleEl     = document.getElementById('tool-title');
  const descEl      = document.getElementById('tool-desc');

  if (flow === 'protect') {
    titleEl.textContent = 'Find Infringers';
    descEl.textContent  = 'Enter your granted patent number or upload the PDF. We will identify companies likely infringing it and score each one for enforcement probability.';
  } else {
    titleEl.textContent = 'Prior Art & Design-Around';
    descEl.textContent  = 'Enter a patent similar to your idea. We will surface invalidity challenges (prior art) and non-infringement arguments you can use to differentiate.';
  }

  toolSection.classList.remove('hidden');
  setTimeout(() => toolSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
}

function resetFlow() {
  currentFlow = null;
  document.getElementById('tool').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('status').classList.add('hidden');
  document.getElementById('analyze-form').reset();
  document.getElementById('results').innerHTML = '';
  document.getElementById('hero').scrollIntoView({ behavior: 'smooth' });
}

// ── Example links ────────────────────────────────────────────────────
// .example-link  → fills input and runs analysis (primary demo action).
// .view-patent   → opens the patent on Google Patents in a new tab.
// Kept as separate affordances so the presenter controls what the audience sees.
document.querySelectorAll('.example-link').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.getElementById('patent-number').value = a.dataset.patent;
    document.getElementById('analyze-form').dispatchEvent(new Event('submit'));
  });
});

document.querySelectorAll('.view-patent').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const key = a.dataset.patent.replace(/-/g, '').toUpperCase();
    window.open(`https://patents.google.com/patent/${key}/en`, '_blank', 'noopener,noreferrer');
  });
});

// ── Nav: also trigger flow select if Analyze link clicked ────────────
document.getElementById('nav-analyze').addEventListener('click', e => {
  if (!currentFlow) {
    e.preventDefault();
    selectFlow('protect'); // default to infringement flow
  }
});

// ── Form submit ──────────────────────────────────────────────────────
document.getElementById('analyze-form').addEventListener('submit', async e => {
  e.preventDefault();

  const statusEl  = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  const btn       = document.getElementById('submit-btn');

  resultsEl.innerHTML = '';
  document.getElementById('results-section').classList.add('hidden');
  showStatus('Analyzing patent — this may take 15–30 seconds…', false);
  btn.disabled = true;

  const fd = new FormData(e.target);
  const pdf = fd.get('patent_pdf');
  if (!pdf || pdf.size === 0) fd.delete('patent_pdf');

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showStatus(data.error || 'Server error — check the patent number and try again.', true);
      return;
    }

    hideStatus();
    renderResults(data);
  } catch (err) {
    showStatus('Network error: ' + err.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ── Status helpers ───────────────────────────────────────────────────
function showStatus(msg, isError) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className   = isError ? 'error-msg' : '';
  el.classList.remove('hidden');
}
function hideStatus() { document.getElementById('status').classList.add('hidden'); }

// ── Render results ───────────────────────────────────────────────────
function renderResults(data) {
  const section   = document.getElementById('results-section');
  const headerEl  = document.getElementById('results-header');
  const resultsEl = document.getElementById('results');

  if (!data.candidates || data.candidates.length === 0) {
    resultsEl.innerHTML = '<p style="color:#64748b;font-size:.9rem">No candidates found. Try a different patent number.</p>';
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth' });
    return;
  }

  const flow = currentFlow || 'protect';
  const label = flow === 'fto'
    ? `Prior art & design-around analysis for <strong>${esc(data.patent_id)}</strong> — ${data.num_claims_parsed} independent claim(s) parsed`
    : `Infringement analysis for <strong>${esc(data.patent_id)}</strong> — ${data.num_claims_parsed} independent claim(s) parsed`;

  headerEl.innerHTML = label;

  data.candidates.forEach(c => {
    resultsEl.appendChild(buildCard(c, flow));
  });

  section.classList.remove('hidden');
  section.scrollIntoView({ behavior: 'smooth' });
}

function buildCard(c, flow) {
  const card = document.createElement('div');
  card.className = 'candidate-card';

  const prob       = c.enforcement_probability;
  const badgeClass = prob >= 0.6 ? 'enf-high' : prob >= 0.35 ? 'enf-mid' : 'enf-low';
  const badgeLabel = prob >= 0.6 ? 'High enforcement risk'
                   : prob >= 0.35 ? 'Medium enforcement risk'
                   : 'Low enforcement risk';

  // In FTO mode we emphasise invalidity + design-around over the claim chart
  const primarySection = flow === 'fto'
    ? `<div class="card-section-title">Prior Art Challenges (Invalidity Arguments)</div>
       ${buildArgList(c.invalidity_args)}
       <div class="card-section-title">Design-Around Options (Non-Infringement Arguments)</div>
       ${buildArgList(c.non_infringement_args)}
       <div class="card-section-title">Claim Chart (faithfulness: ${(c.claim_chart.overall_confidence * 100).toFixed(0)}%)</div>
       ${buildChartTable(c.claim_chart.mappings)}`
    : `<div class="card-section-title">Claim Chart (confidence: ${(c.claim_chart.overall_confidence * 100).toFixed(0)}%)</div>
       ${buildChartTable(c.claim_chart.mappings)}
       <div class="card-section-title">Non-Infringement Arguments</div>
       ${buildArgList(c.non_infringement_args)}
       <div class="card-section-title">Invalidity Risks</div>
       ${buildArgList(c.invalidity_args)}`;

  card.innerHTML = `
    <div class="card-top">
      <div>
        <span class="rank-badge">#${c.rank}</span>
        <span class="company-name">${esc(c.company_name)}</span>
      </div>
      <span class="enf-badge ${badgeClass}">${badgeLabel} — ${(prob * 100).toFixed(0)}%</span>
    </div>
    <div class="card-body">${primarySection}</div>
  `;
  return card;
}

function buildChartTable(mappings) {
  if (!mappings || mappings.length === 0)
    return '<p style="font-size:.83rem;color:#94a3b8">No claim mappings generated.</p>';

  const rows = mappings.map(m => {
    const cls = m.faithfulness_label === 'supports'           ? 'verdict-supports'
              : m.faithfulness_label === 'partially_supports' ? 'verdict-partial'
              : 'verdict-no';
    const label = m.faithfulness_label.replace(/_/g, ' ');
    return `<tr>
      <td>${esc(m.limitation)}</td>
      <td>${esc(m.evidence || '—')}</td>
      <td class="${cls}">${label}</td>
    </tr>`;
  }).join('');

  return `<table>
    <thead><tr><th>Claim Limitation</th><th>Product Evidence</th><th>Verdict</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function buildArgList(args) {
  if (!args || args.length === 0)
    return '<p style="font-size:.83rem;color:#94a3b8">None generated.</p>';

  const items = args.map(a =>
    `<li>${esc(a.summary)}<span class="arg-basis">${esc(a.legal_basis)}</span></li>`
  ).join('');
  return `<ul class="arg-list">${items}</ul>`;
}

function esc(str) {
  return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
