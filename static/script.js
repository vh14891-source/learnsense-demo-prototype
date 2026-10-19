// show live value next to each slider
document.querySelectorAll('input[type=range]').forEach(function (el) {
  el.addEventListener('input', function () {
    document.getElementById('v-' + el.name).textContent = el.value;
  });
});

// send form data to the Python back end and show the profile
document.getElementById('form').addEventListener('submit', async function (e) {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target).entries());
  const out = document.getElementById('output');
  out.innerHTML = '<h3>Learning profile</h3><p class="muted">Analyzing...</p>';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const r = await res.json();
    if (!res.ok) throw new Error(r.error || 'Request failed');

    const bars = Object.entries(r.vark_probs).map(function ([k, v]) {
      return '<div class="bar">' + k + ' - ' + v + '%' +
             '<div class="bar-track"><div class="bar-fill" style="width:' + v + '%"></div></div></div>';
    }).join('');

    const depthBox = r.depth === 'Surface'
      ? '<div class="warn">Engagement looks surface-level here. Add active recall (self-quizzing) and spaced revision.</div>'
      : '<div class="good">Engagement looks deep here. Keep cross-checking with mixed-topic practice.</div>';

    out.innerHTML =
      '<h3>Learning profile</h3>' +
      '<div class="metrics">' +
        '<div class="metric"><small>Learning preference (VARK)</small><strong>' + r.vark + '</strong><em>' + r.vark_conf + '% confidence</em></div>' +
        '<div class="metric"><small>Study engagement</small><strong>' + r.depth + '</strong><em>' + r.depth_conf + '% confidence</em></div>' +
        '<div class="metric"><small>Study frequency</small><strong>' + r.days + ' days/wk</strong></div>' +
        '<div class="metric"><small>Quiz performance</small><strong>' + r.quiz + '%</strong></div>' +
      '</div>' + bars +
      '<div class="tip"><b>Suggested focus:</b> ' + r.tip + '</div>' + depthBox +
      '<p class="muted" style="margin-top:10px;font-size:.85rem">Indicative patterns from behaviour data - not a diagnosis of the student.</p>';
  } catch (err) {
    out.innerHTML = '<h3>Learning profile</h3><p style="color:#b00020">Error: ' + err.message + '</p>';
  }
});

// load model evaluation tables and charts
fetch('/api/results').then(function (r) { return r.json(); }).then(function (data) {
  const titles = {
    vark_label: 'Learning preference (VARK)',
    engagement_depth: 'Engagement depth (Deep / Surface)'
  };
  let html = '';
  for (const target in data) {
    let rows = '';
    for (const model in data[target]) {
      const m = data[target][model];
      rows += '<tr><td>' + model + '</td><td>' + m.accuracy + '</td><td>' + m.precision_macro +
              '</td><td>' + m.recall_macro + '</td><td>' + m.f1_macro + '</td></tr>';
    }
    html += '<div class="eval"><h3>' + titles[target] + '</h3>' +
      '<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>' + rows + '</table>' +
      '<div class="imgs"><img src="/models/cm_' + target + '.png" alt="confusion matrix">' +
      '<img src="/models/fi_' + target + '.png" alt="feature importance"></div></div>';
  }
  document.getElementById('tables').innerHTML = html;
});