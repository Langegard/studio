// Reporter med tre utfall. passed → GRÖNT, failed/timedOut → RÖTT, skipped → OMÄTT.
// Två lägen döljer sin egen blindhet; "kunde inte titta" är inte "godkänt".
class UtfallReporter {
  constructor() { this.rader = []; }
  onTestEnd(test, result) {
    const fel = result.error && result.error.message ? result.error.message.split('\n')[0].slice(0, 160) : '';
    // Kunde provet inte ens titta (webbläsaren startade inte) är det OMÄTT, inte RÖTT.
    const infra = /browserType\.launch|Executable doesn't exist|ECONNREFUSED/.test(fel);
    const utfall = result.status === 'passed' ? 'GRÖNT'
      : (result.status === 'skipped' || infra) ? 'OMÄTT'
      : 'RÖTT';
    const skal = result.status === 'skipped'
      ? (test.annotations.find(a => a.type === 'skip') || {}).description || 'inget skäl angivet'
      : infra ? 'kunde inte titta: ' + fel : fel;
    this.rader.push({ titel: test.title, utfall, skal });
  }
  onEnd() {
    const n = { 'GRÖNT': 0, 'RÖTT': 0, 'OMÄTT': 0 };
    for (const r of this.rader) {
      n[r.utfall]++;
      console.log(`${r.utfall.padEnd(6)} ${r.titel}${r.skal ? '  —  ' + r.skal : ''}`);
    }
    console.log(`\nSumma: ${n['GRÖNT']} grönt · ${n['RÖTT']} rött · ${n['OMÄTT']} omätt`);
    if (n['OMÄTT'] > 0) console.log('OMÄTT är inte godkänt. Varje omätt rad ska förklaras i PR:en.');
  }
}
module.exports = UtfallReporter;
