// Fenestra: markdown-delmängd. Renderar ENDAST h1–h4, listor och stycken.
// Allt annat (tabeller, kodstaket, bilder, h5+) lämnas som escapad text.
(function (root) {
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function render(src) {
    var lines = String(src == null ? '' : src).replace(/\r\n?/g, '\n').split('\n');
    var out = [];
    var para = [];
    var list = null;

    function flushP() {
      if (para.length) { out.push('<p>' + esc(para.join(' ')) + '</p>'); para = []; }
    }
    function flushL() {
      if (list) {
        out.push('<' + list.type + '>' + list.items.map(function (i) { return '<li>' + esc(i) + '</li>'; }).join('') + '</' + list.type + '>');
        list = null;
      }
    }

    for (var n = 0; n < lines.length; n++) {
      var line = lines[n].replace(/\s+$/, '');
      var m;
      if (!line.trim()) { flushP(); flushL(); continue; }
      if ((m = /^(#{1,4})\s+(.+)$/.exec(line))) {
        flushP(); flushL();
        out.push('<h' + m[1].length + '>' + esc(m[2]) + '</h' + m[1].length + '>');
        continue;
      }
      if ((m = /^\s*[-*+]\s+(.+)$/.exec(line))) {
        flushP();
        if (!list || list.type !== 'ul') { flushL(); list = { type: 'ul', items: [] }; }
        list.items.push(m[1]);
        continue;
      }
      if ((m = /^\s*\d+[.)]\s+(.+)$/.exec(line))) {
        flushP();
        if (!list || list.type !== 'ol') { flushL(); list = { type: 'ol', items: [] }; }
        list.items.push(m[1]);
        continue;
      }
      flushL();
      para.push(line.trim());
    }
    flushP(); flushL();
    return out.join('\n');
  }

  var api = { render: render, esc: esc };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FenestraMd = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
