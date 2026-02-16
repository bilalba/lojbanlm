// Sample comparison panels with tabbed display
// Requires: data.js loaded

function initSamples() {
  // Set up tab switching
  document.querySelectorAll('.sample-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.sample-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      renderSamples(tab.dataset.tab);
    });
  });

  // Render initial tab
  renderSamples('babi');
}

function renderSamples(tabName) {
  const container = document.getElementById('sample-content');
  if (!container) return;

  const samples = DATA.samples[tabName];
  if (!samples) return;

  const enSamples = samples.english;
  const ljSamples = samples.lojban;
  const maxPairs = Math.max(enSamples.length, ljSamples.length);

  let html = '';

  for (let i = 0; i < maxPairs; i++) {
    const en = enSamples[i];
    const lj = ljSamples[i];

    if (!en && !lj) continue;

    const groupLabel = en ? en.label : lj.label;

    html += `<div class="sample-group">`;
    html += `<div class="sample-group-title">${escapeHtml(groupLabel)}</div>`;
    html += `<div class="sample-pair">`;

    // English panel
    html += `<div class="sample-panel">`;
    html += `<div class="sample-lang">English</div>`;
    if (en) {
      html += `<div class="sample-prompt">${escapeHtml(en.prompt)}</div>`;
      html += `<div class="sample-generated">${escapeHtml(en.generated)}</div>`;
      if (en.note) html += `<div class="sample-note">${escapeHtml(en.note)}</div>`;
    } else {
      html += `<div class="sample-note">No matching English sample</div>`;
    }
    html += `</div>`;

    // Lojban panel
    html += `<div class="sample-panel">`;
    html += `<div class="sample-lang">Lojban</div>`;
    if (lj) {
      html += `<div class="sample-prompt">${escapeHtml(lj.prompt)}</div>`;
      html += `<div class="sample-generated">${escapeHtml(lj.generated)}</div>`;
      if (lj.note) html += `<div class="sample-note">${escapeHtml(lj.note)}</div>`;
    } else {
      html += `<div class="sample-note">No matching Lojban sample</div>`;
    }
    html += `</div>`;

    html += `</div>`; // .sample-pair
    html += `</div>`; // .sample-group
  }

  container.innerHTML = html;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
