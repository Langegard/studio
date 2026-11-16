// Fenestra: HTML-mallar. Ren funktion av data → sträng, så att bakänden kan lyfta dem rakt av.
// Läsning kräver inget JS; fenestra.js är enbart förbättring (textstorlek, filter).
var md = require('./md.js');
var esc = md.esc;

var CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'";

function sida(titel, kropp, opts) {
  opts = opts || {};
  return '<!doctype html>\n<html lang="sv">\n<head>\n' +
    '<meta charset="utf-8">\n' +
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' +
    '<meta http-equiv="Content-Security-Policy" content="' + CSP + '">\n' +
    '<meta name="color-scheme" content="dark">\n' +
    '<title>' + esc(titel) + ' · Fenestra</title>\n' +
    '<link rel="stylesheet" href="/fenestra.css">\n' +
    '</head>\n<body class="' + (opts.klass || '') + '">\n' +
    '<header class="topp">\n' +
    '  <a class="hem mal" href="/" aria-label="Till mapplistan">Fenestra</a>\n' +
    '  <div class="verktyg" hidden>\n' +
    '    <button type="button" class="mal" data-text="-" aria-label="Mindre text">A−</button>\n' +
    '    <button type="button" class="mal" data-text="+" aria-label="Större text">A+</button>\n' +
    '  </div>\n' +
    '</header>\n' +
    '<main id="innehall">\n' + kropp + '\n</main>\n' +
    '<script src="/fenestra.js" defer></script>\n' +
    '</body>\n</html>\n';
}

function listning(poster, vy) {
  var rubrik = { '/': 'Utkast', '/k': 'Klara', '/f': 'Filer' }[vy] || 'Utkast';
  var rader = poster.length
    ? poster.map(function (p) {
        var href = p.ljud ? '/ljud/' + encodeURIComponent(p.namn) : '/d/' + encodeURIComponent(p.namn);
        return '<li><a class="post mal" href="' + href + '">' +
          '<span class="namn">' + esc(p.namn) + '</span>' +
          (p.ljud ? '<span class="tagg" aria-label="Ljudfil">ljud</span>' : '') +
          '</a></li>';
      }).join('\n')
    : '<li class="tomt">Inga poster.</li>';
  var filter = '<form class="filter" hidden>' +
    '<label for="sok" class="dampad">Filtrera</label>' +
    '<input id="sok" name="sok" type="search" autocomplete="off" inputmode="search">' +
    '</form>';
  return sida(rubrik, '<h1>' + esc(rubrik) + '</h1>\n' + filter + '\n<ul class="lista">\n' + rader + '\n</ul>', { klass: 'vy-lista' });
}

function dokument(namn, markdown, ljudNamn) {
  var html = md.render(markdown);
  var ljud = ljudNamn
    ? '<section class="ljud"><h2 class="dampad">Ljud</h2>' +
      '<audio controls preload="metadata" src="/ljud/' + encodeURIComponent(ljudNamn) + '"></audio></section>'
    : '';
  return sida(namn, '<article class="dok">\n<p class="dampad brodsmula">' + esc(namn) + '</p>\n' + html + '\n</article>\n' + ljud, { klass: 'vy-dok' });
}

module.exports = { sida: sida, listning: listning, dokument: dokument, CSP: CSP };
