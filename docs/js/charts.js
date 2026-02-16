// Chart.js visualizations for the experiment results
// Requires: Chart.js v4, data.js loaded

const CHART_COLORS = {
  english: '#4A9EFF',
  englishBg: 'rgba(74, 158, 255, 0.7)',
  lojban: '#FFB347',
  lojbanBg: 'rgba(255, 179, 71, 0.7)',
  chance: '#ef4444',
  grid: 'rgba(255,255,255,0.06)',
  text: '#a0a0b8',
  textLight: '#6c6c84'
};

const FONT = { family: "'JetBrains Mono', monospace", size: 11 };

// Common chart defaults
Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.font.family = FONT.family;
Chart.defaults.font.size = FONT.size;

function initCharts() {
  createGrammarChart();
  createBPCChart();
  createDynamicsChart();
  createBabiChart();
}

// ── 4a. Grammar at Every Scale ────────────────────────────────────────
function createGrammarChart() {
  const ctx = document.getElementById('chart-grammar');
  if (!ctx) return;

  const d = DATA.grammarByScale;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: d.map(r => r.label),
      datasets: [
        {
          label: 'English',
          data: d.map(r => r.english),
          backgroundColor: CHART_COLORS.englishBg,
          borderColor: CHART_COLORS.english,
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Lojban',
          data: d.map(r => r.lojban),
          backgroundColor: CHART_COLORS.lojbanBg,
          borderColor: CHART_COLORS.lojban,
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } }
      },
      scales: {
        y: {
          min: 70,
          max: 101,
          grid: { color: CHART_COLORS.grid },
          ticks: { callback: v => v + '%' },
          title: { display: true, text: 'Grammar %', color: CHART_COLORS.textLight }
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 9 }, maxRotation: 45 }
        }
      }
    }
  });
}

// ── 4b. Prediction Quality (BPC) ─────────────────────────────────────
function createBPCChart() {
  const ctx = document.getElementById('chart-bpc');
  if (!ctx) return;

  const d = DATA.bpcComparison;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: d.map(r => r.label),
      datasets: [
        {
          label: 'English',
          data: d.map(r => r.english),
          backgroundColor: CHART_COLORS.englishBg,
          borderColor: CHART_COLORS.english,
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Lojban',
          data: d.map(r => r.lojban),
          backgroundColor: CHART_COLORS.lojbanBg,
          borderColor: CHART_COLORS.lojban,
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } }
      },
      scales: {
        y: {
          min: 1.5,
          grid: { color: CHART_COLORS.grid },
          title: { display: true, text: 'Test BPC (lower = better)', color: CHART_COLORS.textLight }
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 9 }, maxRotation: 45 }
        }
      }
    }
  });
}

// ── 4c. Training Dynamics ────────────────────────────────────────────
function createDynamicsChart() {
  const ctx = document.getElementById('chart-dynamics');
  if (!ctx) return;

  const en = DATA.trainingLog.english;
  const lj = DATA.trainingLog.lojban;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: en.map(p => p.step),
      datasets: [
        {
          label: 'English val BPC',
          data: en.map(p => p.val_bpc),
          borderColor: CHART_COLORS.english,
          backgroundColor: 'rgba(74, 158, 255, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: false
        },
        {
          label: 'Lojban val BPC',
          data: lj.map(p => p.val_bpc),
          borderColor: CHART_COLORS.lojban,
          backgroundColor: 'rgba(255, 179, 71, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.3,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } },
        annotation: null
      },
      scales: {
        y: {
          min: 3,
          grid: { color: CHART_COLORS.grid },
          title: { display: true, text: 'Validation BPC', color: CHART_COLORS.textLight }
        },
        x: {
          grid: { color: CHART_COLORS.grid },
          title: { display: true, text: 'Training Step', color: CHART_COLORS.textLight },
          ticks: { maxTicksLimit: 10 }
        }
      }
    }
  });
}

// ── 4d. bAbI Confound Story ──────────────────────────────────────────
function createBabiChart() {
  const ctx = document.getElementById('chart-babi');
  if (!ctx) return;

  const d = DATA.babiConfound;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: d.map(r => r.label),
      datasets: [
        {
          label: 'English bAbI (seen)',
          data: d.map(r => r.english),
          backgroundColor: CHART_COLORS.englishBg,
          borderColor: CHART_COLORS.english,
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Lojban bAbI (seen)',
          data: d.map(r => r.lojban),
          backgroundColor: CHART_COLORS.lojbanBg,
          borderColor: CHART_COLORS.lojban,
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } },
        annotation: {
          annotations: {
            chanceLine: {
              type: 'line',
              xMin: 20,
              xMax: 20,
              borderColor: CHART_COLORS.chance,
              borderWidth: 2,
              borderDash: [6, 4],
              label: {
                display: true,
                content: 'chance baseline',
                position: 'end',
                color: CHART_COLORS.chance,
                font: { size: 10, family: FONT.family }
              }
            }
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 55,
          grid: { color: CHART_COLORS.grid },
          ticks: { callback: v => v + '%' },
          title: { display: true, text: 'bAbI Accuracy (seen vocab)', color: CHART_COLORS.textLight }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}
