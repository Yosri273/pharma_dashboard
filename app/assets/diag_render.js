document.addEventListener('DOMContentLoaded', function () {
  function prettyJSON(text) {
    try {
      var obj = JSON.parse(text);
      return JSON.stringify(obj, null, 2);
    } catch (e) {
      return text;
    }
  }

  function downloadText(filename, text) {
    var a = document.createElement('a');
    var blob = new Blob([text], {type: 'application/json'});
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // Copy handlers (already in diag_copy.js) - keep compatibility
  document.addEventListener('click', function(e){
    if(e.target && e.target.id === 'comp-diag-download-metrics'){
      var pre = document.getElementById('diag-metrics-validator');
      if(pre) downloadText('metrics_validation.json', pre.innerText || pre.textContent || '');
    }
    if(e.target && e.target.id === 'comp-diag-download-sessions'){
      var pre = document.getElementById('diag-sessions-report');
      if(pre) downloadText('sessions_report.json', pre.innerText || pre.textContent || '');
    }
    if(e.target && e.target.id === 'comp-diag-refresh'){
      // Trigger the Dash refresh by clicking the existing button (it will call the callback)
      // No-op here since Dash handles the Input event; this is a fallback.
    }
  });

  // Observe modal content and pretty-print JSON when it arrives
  var metricsPre = document.getElementById('diag-metrics-validator');
  var sessionsPre = document.getElementById('diag-sessions-report');
  if (metricsPre) {
    var obs = new MutationObserver(function(m){
      metricsPre.textContent = prettyJSON(metricsPre.textContent || metricsPre.innerText || '');
    });
    obs.observe(metricsPre, {childList: true, subtree: true, characterData: true});
  }
  if (sessionsPre) {
    var obs2 = new MutationObserver(function(m){
      sessionsPre.textContent = prettyJSON(sessionsPre.textContent || sessionsPre.innerText || '');
    });
    obs2.observe(sessionsPre, {childList: true, subtree: true, characterData: true});
  }
});
