// App initialization, navigation, scroll animations, timeline
// Requires: data.js, charts.js, samples.js, playground.js

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initTimeline();
  initArchTable();
  initScrollAnimations();
  initCharts();
  initSamples();
  initPlayground();
});

// ── Smooth-scrolling navigation ──────────────────────────────────────
function initNavigation() {
  const nav = document.getElementById('nav');
  const links = document.querySelectorAll('.nav-links a');

  // Highlight active section on scroll
  const sections = document.querySelectorAll('section[id]');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        links.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === '#' + id);
        });
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });

  sections.forEach(s => observer.observe(s));
}

// ── Timeline rendering ───────────────────────────────────────────────
function initTimeline() {
  const container = document.getElementById('timeline');
  if (!container) return;

  let html = '';
  DATA.timeline.forEach(item => {
    html += `
      <div class="timeline-item status-${item.status} fade-in">
        <div class="timeline-dot">${item.version.replace('V', '')}</div>
        <div class="timeline-header">
          <span class="timeline-version">${item.version}</span>
          <span class="timeline-title">${item.title}</span>
          <span class="timeline-params">${item.params}</span>
        </div>
        <div class="timeline-body">
          <p><strong>Changed:</strong> ${item.change}</p>
          <p><strong>Found:</strong> ${item.finding}</p>
          <p><strong>Lesson:</strong> ${item.lesson}</p>
          <span class="timeline-badge badge-${item.status}">${item.status}</span>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ── Architecture table ───────────────────────────────────────────────
function initArchTable() {
  const container = document.getElementById('arch-table');
  if (!container) return;

  let html = '';

  // V3/V3.1 architectures
  html += '<div class="arch-section-label">V3 / V3.1 (Character-level)</div>';
  html += buildArchTable(DATA.architectures.v3, ['Size', 'd_model', 'Layers', 'Heads', 'Params', 'Context', 'Dropout'],
    r => [r.size, r.d_model, r.n_layer, r.n_head, r.params, r.ctx, r.dropout]);

  // V4 architectures
  html += '<div class="arch-section-label">V4 / V5 (BPE, vocab=1024)</div>';
  html += buildArchTable(DATA.architectures.v4, ['Size', 'd_model', 'Layers', 'Heads', 'Params', 'Context', 'Dropout', 'Vocab'],
    r => [r.size, r.d_model, r.n_layer, r.n_head, r.params, r.ctx, r.dropout, r.vocab]);

  container.innerHTML = html;
}

function buildArchTable(rows, headers, rowFn) {
  let html = '<table class="arch-table"><thead><tr>';
  headers.forEach(h => { html += `<th>${h}</th>`; });
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    html += '<tr>';
    rowFn(r).forEach(v => { html += `<td>${v}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

// ── Scroll-triggered fade-in animations ──────────────────────────────
function initScrollAnimations() {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

  // Re-observe dynamically added elements after timeline renders
  setTimeout(() => {
    document.querySelectorAll('.fade-in:not(.visible)').forEach(el => observer.observe(el));
  }, 100);
}
