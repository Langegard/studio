// Fenestra: fixturserver. Emulerar bakändens FORM, aldrig dess innehåll.
// GET /        mapplistning (döljer dotfiler och ljud)
// GET /k, /f   två vyer till (samma listningsform, andra filter)
// GET /d/:n    renderad markdown-delmängd
// GET /ljud/:n ljud med Range-stöd
// Allt material här är syntetiskt och imiterar ingen journalanteckning.
var http = require('http');
var fs = require('fs');
var path = require('path');
var mall = require('../mall.js');

var ROT = __dirname;
var STATIK = path.join(ROT, '..');
var PORT = parseInt(process.env.PORT || '4646', 10);
var LJUD = /\.(wav|mp3|m4a|ogg|flac)$/i;
var rangeRakn = 0;

function skapaWav(fil) {
  var hz = 8000, sek = 3, n = hz * sek;
  var buf = Buffer.alloc(44 + n);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(hz, 24); buf.writeUInt32LE(hz, 28); buf.writeUInt16LE(1, 32); buf.writeUInt16LE(8, 34);
  buf.write('data', 36); buf.writeUInt32LE(n, 40);
  for (var i = 0; i < n; i++) buf[44 + i] = 128 + Math.round(60 * Math.sin(2 * Math.PI * 440 * i / hz));
  fs.writeFileSync(fil, buf);
}
var WAV = path.join(ROT, 'prov-ljud.wav');
if (!fs.existsSync(WAV)) skapaWav(WAV);

function poster(filter) {
  return fs.readdirSync(ROT).filter(function (f) { return f !== 'server.js'; })
    .filter(filter).sort()
    .map(function (f) { return { namn: f, ljud: LJUD.test(f) }; });
}
var VYER = {
  '/': function (f) { return f[0] !== '.' && !LJUD.test(f) && /\.md$/i.test(f); },
  '/k': function (f) { return /^klar-/.test(f) && /\.md$/i.test(f); },
  '/f': function (f) { return f[0] !== '.'; }
};

function svara(res, kod, typ, kropp, extra) {
  var h = Object.assign({ 'Content-Type': typ, 'Content-Security-Policy': mall.CSP, 'Cache-Control': 'no-store' }, extra || {});
  res.writeHead(kod, h); res.end(kropp);
}

function ljud(req, res, namn) {
  var fil = path.join(ROT, namn);
  if (!fs.existsSync(fil) || !LJUD.test(namn)) return svara(res, 404, 'text/plain; charset=utf-8', 'finns inte');
  var storlek = fs.statSync(fil).size;
  var r = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
  if (r) {
    rangeRakn++;
    var start = r[1] === '' ? Math.max(0, storlek - parseInt(r[2], 10)) : parseInt(r[1], 10);
    var slut = r[1] === '' || r[2] === '' ? storlek - 1 : Math.min(parseInt(r[2], 10), storlek - 1);
    if (start > slut || start >= storlek) { res.writeHead(416, { 'Content-Range': 'bytes */' + storlek }); return res.end(); }
    res.writeHead(206, { 'Content-Type': 'audio/wav', 'Accept-Ranges': 'bytes', 'Content-Range': 'bytes ' + start + '-' + slut + '/' + storlek, 'Content-Length': slut - start + 1 });
    return fs.createReadStream(fil, { start: start, end: slut }).pipe(res);
  }
  res.writeHead(200, { 'Content-Type': 'audio/wav', 'Accept-Ranges': 'bytes', 'Content-Length': storlek });
  fs.createReadStream(fil).pipe(res);
}

var server = http.createServer(function (req, res) {
  var url = decodeURIComponent(req.url.split('?')[0]);
  if (VYER[url]) return svara(res, 200, 'text/html; charset=utf-8', mall.listning(poster(VYER[url]), url));
  var m;
  if ((m = /^\/d\/([^/]+)$/.exec(url))) {
    var fil = path.join(ROT, m[1]);
    if (m[1][0] === '.' || !fs.existsSync(fil) || !/\.md$/i.test(m[1])) return svara(res, 404, 'text/plain; charset=utf-8', 'finns inte');
    var bas = m[1].replace(/\.md$/i, '');
    var lj = fs.existsSync(path.join(ROT, bas + '.wav')) ? bas + '.wav' : (m[1] === 'prov-a.md' ? 'prov-ljud.wav' : null);
    return svara(res, 200, 'text/html; charset=utf-8', mall.dokument(m[1], fs.readFileSync(fil, 'utf8'), lj));
  }
  if ((m = /^\/ljud\/([^/]+)$/.exec(url))) return ljud(req, res, m[1]);
  if (url === '/_prov/range') return svara(res, 200, 'application/json', JSON.stringify({ range: rangeRakn }));
  if (url === '/fenestra.css') return svara(res, 200, 'text/css; charset=utf-8', fs.readFileSync(path.join(STATIK, 'fenestra.css')));
  if (url === '/fenestra.js') return svara(res, 200, 'text/javascript; charset=utf-8', fs.readFileSync(path.join(STATIK, 'fenestra.js')));
  svara(res, 404, 'text/plain; charset=utf-8', 'finns inte');
});

server.listen(PORT, '127.0.0.1', function () { console.log('fenestra fixtur på http://127.0.0.1:' + PORT); });
