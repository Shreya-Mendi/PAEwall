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

  if (data.patent_claim) {
    resultsEl.appendChild(buildClaimPanel(data.patent_claim));
  }

  data.candidates.forEach(c => {
    resultsEl.appendChild(buildCard(c, flow));
  });

  attachClaimHighlighting();

  section.classList.remove('hidden');
  section.scrollIntoView({ behavior: 'smooth' });
}

// ── Claim panel + highlighter interaction ─────────────────────────────
function buildClaimPanel(claim) {
  const panel = document.createElement('div');
  panel.className = 'claim-panel';
  panel.id = 'claim-panel';

  const parts = (claim.parts || []).map(p => `
    <li class="claim-part" data-claim-ref="${esc(p.ref)}">
      <span class="claim-ref-badge">${esc(p.ref)}</span>
      <span class="claim-part-text" data-claim-ref="${esc(p.ref)}">${esc(p.text)}</span>
    </li>`).join('');

  panel.innerHTML = `
    <div class="claim-panel-head">
      <span class="claim-panel-eyebrow">The Patent Claim</span>
      <span class="claim-panel-note">Hover a row below to see which part of the claim it matches.</span>
    </div>
    <div class="claim-panel-body">
      <p class="claim-preamble">Claim ${claim.claim_number}. ${esc(claim.preamble || '')}</p>
      <ol class="claim-parts">${parts}</ol>
    </div>`;

  return panel;
}

function attachClaimHighlighting() {
  const panel = document.getElementById('claim-panel');
  if (!panel) return;

  const parts = panel.querySelectorAll('.claim-part-text, .claim-part');
  const rows  = document.querySelectorAll('#results tr[data-claim-ref]');

  const setActive = (ref) => {
    parts.forEach(el => {
      el.classList.toggle('highlighted', el.dataset.claimRef === ref);
    });
    rows.forEach(r => {
      r.classList.toggle('row-active', r.dataset.claimRef === ref);
    });
  };
  const clearActive = () => {
    parts.forEach(el => el.classList.remove('highlighted'));
    rows.forEach(r => r.classList.remove('row-active'));
  };

  rows.forEach(row => {
    const ref = row.dataset.claimRef;
    row.addEventListener('mouseenter', () => setActive(ref));
    row.addEventListener('mouseleave', clearActive);
    row.addEventListener('click', () => {
      setActive(ref);
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  });

  panel.querySelectorAll('.claim-part').forEach(p => {
    const ref = p.dataset.claimRef;
    p.addEventListener('mouseenter', () => setActive(ref));
    p.addEventListener('mouseleave', clearActive);
  });
}

function buildCard(c, flow) {
  const card = document.createElement('div');

  const prob      = c.enforcement_probability;
  const riskClass = prob >= 0.6 ? 'risk-high' : prob >= 0.35 ? 'risk-mid' : 'risk-low';
  const riskLabel = prob >= 0.6 ? 'High risk' : prob >= 0.35 ? 'Medium risk' : 'Low risk';
  card.className  = `candidate-card ${riskClass}`;

  const retrievalPct = ((c.retrieval_score || 0) * 100).toFixed(0);
  const confidencePct = ((c.claim_chart.overall_confidence || 0) * 100).toFixed(0);

  // Claim chart section — always present
  const claimChartSection = `
    <div class="card-section-title">Claim Chart · Faithfulness</div>
    <div class="confidence-bar-wrap">
      <span class="confidence-bar-label">Overall Confidence</span>
      <div class="confidence-bar">
        <div class="confidence-bar-fill" style="width:${confidencePct}%"></div>
      </div>
      <span class="confidence-bar-val">${confidencePct}%</span>
    </div>
    <div class="chart-wrap">${buildChartTable(c.claim_chart.mappings)}</div>`;

  const nonInfringementSection = `
    <div class="card-section-title">Non-Infringement Arguments</div>
    ${buildArgList(c.non_infringement_args)}`;

  const invaliditySection = `
    <div class="card-section-title card-section-invalidity">Invalidity Risks</div>
    ${buildArgList(c.invalidity_args)}`;

  // FTO mode leads with invalidity + design-around; Protect mode leads with claim chart
  const body = flow === 'fto'
    ? `${invaliditySection}${nonInfringementSection}${claimChartSection}`
    : `${claimChartSection}${nonInfringementSection}${invaliditySection}`;

  card.innerHTML = `
    <div class="card-top">
      <div class="card-top-left">
        <div class="card-top-row">
          <span class="rank-badge">Rank ${c.rank}</span>
          <span class="company-name">${esc(c.company_name)}</span>
        </div>
        <div class="retrieval-score-wrap">
          <span class="label">Retrieval Score</span>
          <div class="retrieval-score-bar">
            <div class="retrieval-score-fill" style="width:${retrievalPct}%"></div>
          </div>
          <span class="retrieval-score-val">${(c.retrieval_score || 0).toFixed(2)}</span>
        </div>
      </div>
      <div class="enf-gauge">
        <span class="enf-label">Enforcement Probability</span>
        <span class="enf-value">${(prob * 100).toFixed(0)}%</span>
        <span class="enf-pill">${riskLabel}</span>
      </div>
    </div>
    <div class="card-body">${body}</div>
  `;
  return card;
}

function buildChartTable(mappings) {
  if (!mappings || mappings.length === 0)
    return '<p style="font-size:.83rem;color:#94a3b8">No claim mappings generated.</p>';

  const rows = mappings.map(m => {
    const cls = m.faithfulness_label === 'supports'           ? 'verdict-supports'
              : m.faithfulness_label === 'partial'            ? 'verdict-partial'
              : m.faithfulness_label === 'partially_supports' ? 'verdict-partial'
              : 'verdict-no';
    const label = (m.faithfulness_label || '').replace(/_/g, ' ');
    const scorePct = (typeof m.faithfulness_score === 'number')
      ? ` <span class="verdict-score">${(m.faithfulness_score * 100).toFixed(0)}%</span>`
      : '';
    const refBadge = m.claim_ref
      ? `<span class="claim-ref-badge">${esc(m.claim_ref)}</span>`
      : '';
    const rowRefAttr = m.claim_ref ? ` data-claim-ref="${esc(m.claim_ref)}"` : '';
    const pct = typeof m.faithfulness_score === 'number' ? Math.round(m.faithfulness_score * 100) : null;
    const barCls = cls === 'verdict-supports' ? 'supports' : cls === 'verdict-partial' ? 'partial' : 'no';
    const miniBar = pct !== null
      ? `<div class="row-score-bar"><div class="row-score-fill ${barCls}" style="width:${pct}%"></div></div>
         <span class="row-score-pct">${pct}%</span>`
      : '';
    return `<tr${rowRefAttr}>
      <td>${refBadge}</td>
      <td>${esc(m.limitation)}</td>
      <td>${esc(m.evidence || '—')}</td>
      <td><div class="verdict-cell"><span class="verdict-pill ${cls}">${label}</span>${miniBar}</div></td>
    </tr>`;
  }).join('');

  return `<table>
    <thead><tr><th>Matches</th><th>Claim Limitation</th><th>Product Evidence</th><th>Verdict</th></tr></thead>
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
