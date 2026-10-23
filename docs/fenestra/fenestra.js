// Fenestra: förbättring, aldrig krav. Sidan läses fullt ut utan detta skript.
(function () {
  var root = document.documentElement;
  var NYCKEL = 'fenestra.textskala';

  function sattSkala(v) {
    v = Math.min(1.6, Math.max(0.8, v));
    root.style.setProperty('--textskala', String(v));
    try { localStorage.setItem(NYCKEL, String(v)); } catch (e) {}
  }
  var sparad = 1;
  try { sparad = parseFloat(localStorage.getItem(NYCKEL)) || 1; } catch (e) {}
  sattSkala(sparad);

  var verktyg = document.querySelector('.verktyg');
  if (verktyg) {
    verktyg.hidden = false;
    verktyg.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-text]');
      if (!b) return;
      var nu = parseFloat(getComputedStyle(root).getPropertyValue('--textskala')) || 1;
      sattSkala(b.dataset.text === '+' ? nu + 0.1 : nu - 0.1);
    });
  }

  var form = document.querySelector('.filter');
  var lista = document.querySelector('.lista');
  if (form && lista) {
    form.hidden = false;
    form.addEventListener('submit', function (e) { e.preventDefault(); });
    var falt = form.querySelector('input');
    falt.addEventListener('input', function () {
      var q = falt.value.trim().toLocaleLowerCase('sv');
      lista.querySelectorAll('li').forEach(function (li) {
        var t = (li.textContent || '').toLocaleLowerCase('sv');
        li.hidden = q !== '' && t.indexOf(q) === -1;
      });
    });
  }
})();
