document.addEventListener('DOMContentLoaded', function () {
  function copyText(text) {
    if (!navigator.clipboard) {
      // fallback
      var ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta);
      return;
    }
    navigator.clipboard.writeText(text).catch(function(e){ console.warn('copy failed', e); });
  }

  document.addEventListener('click', function(e){
    if(e.target && e.target.id === 'comp-diag-copy-metrics'){
      var pre = document.getElementById('diag-metrics-validator');
      if(pre) copyText(pre.innerText || pre.textContent || '');
    }
    if(e.target && e.target.id === 'comp-diag-copy-sessions'){
      var pre = document.getElementById('diag-sessions-report');
      if(pre) copyText(pre.innerText || pre.textContent || '');
    }
  });
});
