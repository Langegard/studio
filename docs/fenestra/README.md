# Fenestra — läsyta för journalutkast på telefon

Ren framsida (HTML-mallar, CSS, JS) mot en påhittad fixtur. Ingen bakände byggs här; fixturen
emulerar enbart bakändens *form* (mapplistning, renderad markdown-delmängd, ljud med Range).
Allt fixturmaterial är syntetiskt och imiterar ingen journalanteckning.

## Delar

| Fil | Roll |
|---|---|
| `mall.js` | HTML-mallar som ren funktion av data; lyfts rakt in i bakänden |
| `md.js` | Markdown-delmängd: h1–h4, listor, stycken; allt annat blir text |
| `fenestra.css` | Designtokens, mobil först, 44 px mål, 16 px fält, systemtypsnitt |
| `fenestra.js` | Förbättring (textstorlek, filter). Läsning kräver det inte |
| `fixtur/server.js` | Fixturserver: `/`, `/k`, `/f`, `/d/:namn`, `/ljud/:namn` (Range) |
| `prov/` | Acceptansprov A–I med tre utfall och mutationskontroll |

## Köra

```
cd prov
npm install
npm run prov        # A–I, utfall GRÖNT / RÖTT / OMÄTT per rad
npm run mutation    # visar att B, C, D, G faller när koden avsiktligt bryts
```

Fixturservern startas av proven. För hand: `node fixtur/server.js` och öppna
`http://127.0.0.1:4646/`. Ljudfixturen (`fixtur/prov-ljud.wav`, 3 s syntetisk ton) genereras
vid första start och ligger inte i repot.

Finns en Chromium redan installerad som Playwright inte hittar, peka ut den:
`FENESTRA_CHROMIUM=/sökväg/till/chrome npm run prov`.

Senaste körning i utvecklingsmiljön: 9 grönt · 0 rött · 0 omätt; mutationskontroll 4/4 föll.

## Kriterier → prov

A rullbredd · B mål 44×44 · C fält ≥16 px under `pointer: coarse` · D kontrast räknad mot alla tre
bottnar · E noll externa resurser · F åäö utan familjebyte med negativkontroll · G delmängden och
inget annat · H Range vid spolning · I läsning utan JS.

OMÄTT rapporteras när förutsättningen saknas (t.ex. webbläsare utan `pointer: coarse`). Det är
inte godkänt och ska förklaras i PR:en.
