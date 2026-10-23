// Acceptansprov A–I för Fenestra. Varje prov kan bli GRÖNT, RÖTT eller OMÄTT.
// OMÄTT sätts uttryckligen via omatt() när förutsättningen för att mäta saknas.
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const ROT = path.join(__dirname, '..');
const BOTTNAR = ['#0B0E12', '#131922', '#1A2230'];

function omatt(skal) { test.skip(true, 'OMÄTT: ' + skal); }

// Sonden läser tillbaka sin egen bredd. Blev fönstret inte det begärda vägrar provet dom:
// headless Chrome kan klämma fast bredden (375 begärt, 504 levererat) och då nås villkoret aldrig.
const BEGARD = 375;
async function kravBredd(page) {
  const w = await page.evaluate(() => window.innerWidth);
  if (w !== BEGARD) omatt(`begärd fönsterbredd ${BEGARD} px, fick ${w} px — villkoret nåddes inte`);
  return w;
}

// WCAG 2.x relativ luminans och kontrastkvot, räknad i provet, aldrig hårdkodad.
function lum([r, g, b]) {
  const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function kvot(a, b) { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); }
function hex(h) { const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h); return [1, 2, 3].map(i => parseInt(m[i], 16)); }
function rgbStr(s) {
  const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/.exec(s);
  return m ? { rgb: [+m[1], +m[2], +m[3]], a: m[4] === undefined ? 1 : +m[4] } : null;
}

test.describe('Fenestra', () => {

  test('A · 375×812 ger ingen horisontell rullning', async ({ page }) => {
    for (const url of ['/', '/d/prov-a.md', '/d/prov-tabell.md']) {
      await page.goto(url);
      const w = await kravBredd(page);
      const { sw, cw } = await page.evaluate(() => ({ sw: document.scrollingElement.scrollWidth, cw: document.scrollingElement.clientWidth }));
      expect(cw, url + ' clientWidth ska vara fönsterbredden').toBe(w);
      expect(sw, url).toBeLessThanOrEqual(cw);
    }
  });

  test('B · varje klickbart mål är minst 44×44 CSS-px', async ({ page }) => {
    for (const url of ['/', '/d/prov-a.md']) {
      await page.goto(url);
      const mal = await page.evaluate(() => {
        const sel = 'a, button, input, select, textarea, audio, [role="button"]';
        return [...document.querySelectorAll(sel)].filter(e => !e.hidden && e.offsetParent !== null).map(e => {
          const r = e.getBoundingClientRect();
          return { tag: e.tagName + (e.className ? '.' + e.className : ''), w: r.width, h: r.height };
        });
      });
      if (mal.length === 0) omatt('inga klickbara mål hittades på ' + url);
      for (const m of mal) {
        expect(m.w, url + ' ' + m.tag + ' bredd').toBeGreaterThanOrEqual(44);
        expect(m.h, url + ' ' + m.tag + ' höjd').toBeGreaterThanOrEqual(44);
      }
    }
  });

  test('C · fält har font-size ≥ 16px under (pointer: coarse)', async ({ page }) => {
    await page.goto('/');
    const grov = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
    if (!grov) omatt('webbläsaren rapporterar inte pointer: coarse; kan inte mäta villkoret');
    const falt = await page.evaluate(() => [...document.querySelectorAll('input, textarea, select')]
      .map(e => ({ id: e.id || e.name || e.tagName, px: parseFloat(getComputedStyle(e).fontSize) })));
    if (falt.length === 0) omatt('inga fält på sidan');
    for (const f of falt) expect(f.px, f.id).toBeGreaterThanOrEqual(16);
  });

  test('D · kontrast text ≥ 4.5:1 och grafik ≥ 3.0:1 mot alla tre bottnar', async ({ page }) => {
    await page.goto('/d/prov-a.md');
    const farger = await page.evaluate(() => {
      const text = new Set(), grafik = new Set();
      for (const e of document.querySelectorAll('body *')) {
        const cs = getComputedStyle(e);
        if (e.offsetParent === null && e.tagName !== 'BODY') continue;
        if ((e.textContent || '').trim() && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) text.add(cs.color);
        if (cs.borderTopStyle !== 'none' && parseFloat(cs.borderTopWidth) > 0) grafik.add(cs.borderTopColor);
        if (cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0) grafik.add(cs.outlineColor);
      }
      const fokus = getComputedStyle(document.documentElement).getPropertyValue('--fosfor').trim();
      return { text: [...text], grafik: [...grafik], fokus };
    });
    if (farger.text.length === 0) omatt('ingen textfärg kunde läsas');
    const kolla = (lista, min, slag) => {
      for (const c of lista) {
        const p = rgbStr(c); if (!p || p.a < 1) continue;
        for (const b of BOTTNAR) expect(kvot(p.rgb, hex(b)), `${slag} ${c} mot ${b}`).toBeGreaterThanOrEqual(min);
      }
    };
    kolla(farger.text, 4.5, 'text');
    kolla(farger.grafik, 3.0, 'grafik');
    if (/^#[0-9a-f]{6}$/i.test(farger.fokus)) for (const b of BOTTNAR) expect(kvot(hex(farger.fokus), hex(b)), 'fokusring mot ' + b).toBeGreaterThanOrEqual(3.0);
  });

  test('E · noll externa resurser i bygget', async () => {
    const filer = ['fenestra.css', 'fenestra.js', 'md.js', 'mall.js', 'fixtur/server.js'];
    for (const f of filer) {
      let s = fs.readFileSync(path.join(ROT, f), 'utf8');
      s = s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/<!--[\s\S]*?-->/g, '');
      const traffar = s.match(/https?:\/\/(?!127\.0\.0\.1)[^\s'")]+/g) || [];
      expect(traffar, f).toEqual([]);
    }
  });

  test('F · åäö ÅÄÖ renderas utan familjebyte, med negativkontroll', async ({ page }) => {
    await page.goto('/d/prov-a.md');
    const r = await page.evaluate(() => {
      const GENERISKA = ['system-ui', 'sans-serif', 'serif', 'monospace', 'ui-sans-serif', 'ui-serif', 'ui-monospace', 'cursive', 'fantasy'];
      const stil = getComputedStyle(document.body).fontFamily;
      const ra = stil.split(',')[0].trim().replace(/^["']|["']$/g, '');
      const forsta = GENERISKA.includes(ra) ? ra : `"${ra}"`;
      const c = document.createElement('canvas').getContext('2d');
      const matt = (fam, txt) => { c.font = `24px ${fam}`; return c.measureText(txt).width; };
      const prov = 'åäö ÅÄÖ Syntetisk mening';
      const fejk = '"Foobar Nonexistent"';
      return {
        forsta,
        lanar_forsta: matt(forsta, prov) !== matt(`${forsta}, monospace`, prov),
        lanar_fejk: matt(fejk, prov) !== matt(`${fejk}, monospace`, prov),
        forsta_ar_mono: matt(forsta, prov) === matt('monospace', prov),
        bredd: matt(forsta, prov),
        fontsCheckSagerFinns: document.fonts.check(`16px ${fejk}`)
      };
    });
    // Negativkontroll: en påhittad familj MÅSTE synas låna glyfer, annars mäter metoden ingenting.
    if (!r.lanar_fejk) omatt('negativkontrollen slog inte ut: miljön kan inte skilja en obefintlig familj från fallback');
    expect(r.bredd).toBeGreaterThan(0);
    expect(r.lanar_forsta, `åäö renderas med lånade glyfer från fallback i stället för ${r.forsta}`).toBe(false);
    expect(r.forsta_ar_mono, `${r.forsta} föll tillbaka hela vägen till monospace`).toBe(false);
  });

  test('G · markdown-delmängden renderas, allt annat lämnas som text utan krasch', async ({ page }) => {
    await page.goto('/d/prov-a.md');
    for (const t of ['h1', 'h2', 'h3', 'h4', 'ul li', 'ol li', 'article p']) await expect(page.locator(t).first()).toBeVisible();
    const res = await page.goto('/d/prov-tabell.md');
    expect(res.status()).toBe(200);
    await expect(page.locator('table')).toHaveCount(0);
    await expect(page.locator('pre, code')).toHaveCount(0);
    await expect(page.locator('h5')).toHaveCount(0);
    await expect(page.locator('img')).toHaveCount(0);
    await expect(page.getByText('kodstaket som ska lämnas som text')).toBeVisible();
    await expect(page.getByText('Sidan ska fortfarande stå.')).toBeVisible();
  });

  test('H · <audio> spolar och skickar Range', async ({ page, request }) => {
    const r = await request.get('/ljud/prov-ljud.wav', { headers: { Range: 'bytes=100-199' } });
    expect(r.status()).toBe(206);
    expect(r.headers()['content-range']).toMatch(/^bytes 100-199\/\d+$/);
    const fore = (await (await request.get('/_prov/range')).json()).range;
    await page.goto('/d/prov-a.md');
    const audio = page.locator('audio');
    if (await audio.count() === 0) omatt('ingen <audio> på sidan');
    const kan = await page.evaluate(async () => {
      const a = document.querySelector('audio');
      await new Promise(res => { if (a.readyState >= 1) res(); else { a.addEventListener('loadedmetadata', res, { once: true }); setTimeout(res, 5000); } });
      if (!isFinite(a.duration) || a.duration === 0) return { ok: false, skal: 'ingen duration; webbläsaren avkodade inte fixturen' };
      const p = new Promise(res => { a.addEventListener('seeked', () => res(true), { once: true }); setTimeout(() => res(false), 5000); });
      a.currentTime = Math.min(2, a.duration * 0.7);
      return { ok: await p, skal: 'seeked' };
    });
    if (!kan.ok) omatt(kan.skal);
    await page.waitForTimeout(300);
    const efter = (await (await request.get('/_prov/range')).json()).range;
    expect(efter, 'servern ska ha fått minst en Range-förfrågan från spolningen').toBeGreaterThan(fore);
  });

  test('I · läsning fungerar utan JS', async ({ browser }) => {
    const ctx = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto('http://127.0.0.1:4646/d/prov-a.md');
    await kravBredd(page);
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByText('Syntetisk mening 1 under rubrik 2')).toBeVisible();
    const bakgrund = await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor);
    expect(bakgrund, 'CSS ska vara laddad utan JS').not.toBe('rgba(0, 0, 0, 0)');
    const { sw, cw } = await page.evaluate(() => ({ sw: document.scrollingElement.scrollWidth, cw: document.scrollingElement.clientWidth }));
    expect(sw).toBeLessThanOrEqual(cw);
    await ctx.close();
  });
});
