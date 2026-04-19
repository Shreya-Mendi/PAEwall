const form = document.getElementById('analyze-form');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const submitBtn = document.getElementById('submit-btn');

// Example patent quick-fill links
document.querySelectorAll('.example-link').forEach(a => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('patent-number').value = a.dataset.patent;
    form.dispatchEvent(new Event('submit'));
  });
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultsEl.innerHTML = '';
  resultsEl.classList.add('hidden');
  showStatus('Analyzing patent…', false);
  submitBtn.disabled = true;

  const fd = new FormData(form);
  // Remove empty file field — browser includes it even when no file chosen,
  // which causes FastAPI to return 422 Unprocessable Content.
  const pdfFile = fd.get('patent_pdf');
  if (!pdfFile || pdfFile.size === 0) fd.delete('patent_pdf');

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showStatus(data.error || 'Server error.', true);
      return;
    }

    hideStatus();
    renderResults(data);
  } catch (err) {
    showStatus('Network error: ' + err.message, true);
  } finally {
    submitBtn.disabled = false;
  }
});

function showStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'error-msg' : '';
  statusEl.classList.remove('hidden');
}
function hideStatus() { statusEl.classList.add('hidden'); }

function renderResults(data) {
  if (!data.candidates || data.candidates.length === 0) {
    resultsEl.innerHTML = '<p>No candidates found.</p>';
    resultsEl.classList.remove('hidden');
    return;
  }

  const h = document.createElement('h2');
  h.textContent = `Results for patent ${data.patent_id} — ${data.num_claims_parsed} independent claim(s) parsed`;
  h.style.cssText = 'font-size:1rem;margin-bottom:16px;color:#334155';
  resultsEl.appendChild(h);

  data.candidates.forEach(c => {
    resultsEl.appendChild(buildCard(c));
  });
  resultsEl.classList.remove('hidden');
}

function buildCard(c) {
  const card = document.createElement('div');
  card.className = 'candidate-card';

  const prob = c.enforcement_probability;
  const badgeClass = prob >= 0.6 ? 'enf-high' : prob >= 0.35 ? 'enf-mid' : 'enf-low';
  const badgeLabel = prob >= 0.6 ? 'High risk' : prob >= 0.35 ? 'Medium risk' : 'Low risk';

  card.innerHTML = `
    <div class="candidate-header">
      <span class="company-name">#${c.rank} ${esc(c.company_name)}</span>
      <span class="enf-badge ${badgeClass}">${badgeLabel} — ${(prob * 100).toFixed(0)}% enforcement probability</span>
    </div>

    <div class="section-title">Claim Chart (confidence: ${(c.claim_chart.overall_confidence * 100).toFixed(0)}%)</div>
    ${buildChartTable(c.claim_chart.mappings)}

    <div class="section-title">Non-Infringement Arguments</div>
    ${buildArgList(c.non_infringement_args)}

    <div class="section-title">Invalidity Arguments</div>
    ${buildArgList(c.invalidity_args)}
  `;
  return card;
}

function buildChartTable(mappings) {
  if (!mappings || mappings.length === 0) return '<p style="font-size:.85rem;color:#999">No mappings generated.</p>';
  const rows = mappings.map(m => {
    const cls = m.faithfulness_label === 'supports' ? 'label-supports'
              : m.faithfulness_label === 'partially_supports' ? 'label-partial'
              : 'label-does-not';
    return `<tr>
      <td>${esc(m.limitation)}</td>
      <td>${esc(m.evidence || '—')}</td>
      <td class="${cls}">${m.faithfulness_label.replace(/_/g, ' ')}</td>
    </tr>`;
  }).join('');
  return `<table><thead><tr><th>Limitation</th><th>Evidence</th><th>Verdict</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function buildArgList(args) {
  if (!args || args.length === 0) return '<p style="font-size:.85rem;color:#999">None generated.</p>';
  const items = args.map(a =>
    `<li>${esc(a.summary)} <span class="arg-basis">[${esc(a.legal_basis)}]</span></li>`
  ).join('');
  return `<ul class="arg-list">${items}</ul>`;
}

function esc(str) {
  return String(str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
