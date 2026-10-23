#!/usr/bin/env python3
"""diarjamfor — jämför två diariseringar (RTTM) och sätter utredningens band.

Utredning 2 (docs/utredningar/svensk-talarseparerad-transkription-2026.md,
avsnitt 7) vill veta om en port eller en kvantisering ändrar diariseringen:
experiment 1 (portparitet, t.ex. audio.cpp BF16 mot NeMo), 2 (Q8_0 mot BF16)
och 3 (samma sak på CPU). Alla tre slutar i samma fråga: hur mycket skiljer
sig två RTTM-filer åt? Det här verktyget svarar, oberoende av vilken motor som
skrev filerna.

Användning:
    python3 diarjamfor.py --ref bf16.rttm --hyp q8.rttm --experiment 2
    python3 diarjamfor.py --ref bf16/ --hyp q8/ --experiment 2 --ut r.json

Med kataloger paras filerna ihop på filnamn. Mått per fil och totalt:
- turer: antal segment i varje fil och avvikelsen i procent;
- DER med referensen som facit, collar 0, överlapp inräknat, på 10 ms-ramar,
  med optimal talarmatchning (ungersk metod), uppdelad på missat, falsklarm
  och förväxling;
- etikettskillnad: andel ramar där talaretiketterna skiljer sig utan
  matchning. Den är meningsfull när båda körningarna kommer från samma modell,
  som i experiment 2 och 3.

Överlappande segment med samma talare räknas en gång, som i NIST:s md-eval.
pyannote.metrics räknar dem två gånger. Utan sådan självöverlapp ger verktyget
samma DER som pyannote.metrics 3.2.1 och 4.2.0 (collar 0, överlapp inräknat): största
skillnaden över 30 slumpade femminutersmöten var 0,0005 procentenheter.

Band (utredningens tröskelvärden, avsnitt 7):
    exp 1: GRÖNT DER ≤ 2 % och turer ±1 %; RÖTT DER > 6 % eller turer > ±3 %
    exp 2, 3: GRÖNT turer ±2 % och DER ≤ 0,5 %; RÖTT turer > ±6 % eller DER > 1,5 %
    däremellan GRÅTT.

Verktyget läser bara tider och etiketter, aldrig ljud eller text.

Avslutningskod: 0 = GRÖNT eller GRÅTT, 2 = RÖTT, 1 = fel.
"""
import argparse
import json
import math
import os
import sys

GRONT, GRATT, ROTT = "GRÖNT", "GRÅTT", "RÖTT"
RAM = 0.01   # 10 ms, som utredningens "same-label-aktivitetsskillnad på 10 ms"

BAND = {
    # experiment: (DER grön, DER röd, turer grön, turer röd), allt i procent
    1: (2.0, 6.0, 1.0, 3.0),
    2: (0.5, 1.5, 2.0, 6.0),
    3: (0.5, 1.5, 2.0, 6.0),
}


def las_rttm(sokvag):
    """Läser SPEAKER-rader. Ger {uri: [(start, slut, talare), ...]}."""
    ut = {}
    with open(sokvag, encoding="utf-8") as f:
        for nr, rad in enumerate(f, 1):
            delar = rad.split()
            if not delar or delar[0] != "SPEAKER":
                continue
            if len(delar) < 8:
                raise ValueError(f"{sokvag}:{nr}: för få fält i SPEAKER-raden")
            start, langd = float(delar[3]), float(delar[4])
            if langd < 0:
                raise ValueError(f"{sokvag}:{nr}: negativ längd")
            ut.setdefault(delar[1], []).append((start, start + langd, delar[7]))
    return ut


def ramar(segment, antal):
    """Aktivitet per talare som bytearray med en byte per 10 ms-ram."""
    akt = {}
    for start, slut, talare in segment:
        a = akt.setdefault(talare, bytearray(antal))
        i, j = int(round(start / RAM)), int(round(slut / RAM))
        for k in range(max(0, i), min(antal, j)):
            a[k] = 1
    return akt


def ungersk(vinst):
    """Optimal tilldelning som maximerar summan i en (rader × kolumner)-matris.

    Ger en lista av (rad, kolumn). Kuhn–Munkres med potentialer, O(n³).
    """
    n_r, n_k = len(vinst), len(vinst[0]) if vinst else 0
    n = max(n_r, n_k)
    if n == 0:
        return []
    storst = max((max(r) for r in vinst if r), default=0)
    kost = [[storst - (vinst[i][j] if i < n_r and j < n_k else 0) for j in range(n)]
            for i in range(n)]
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, vag = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, anvand = [INF] * (n + 1), [False] * (n + 1)
        while True:
            anvand[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not anvand[j]:
                    c = kost[i0 - 1][j - 1] - u[i0] - v[j]
                    if c < minv[j]:
                        minv[j], vag[j] = c, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if anvand[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = vag[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, n + 1)
            if p[j] - 1 < n_r and j - 1 < n_k]


def jamfor(ref_seg, hyp_seg):
    """DER (collar 0, överlapp inräknat), dess delar, etikettskillnad och turer."""
    slut = max([s for _, s, _ in ref_seg + hyp_seg] + [0.0])
    antal = int(math.ceil(slut / RAM)) + 1
    R, H = ramar(ref_seg, antal), ramar(hyp_seg, antal)
    rn, hn = sorted(R), sorted(H)
    overlapp = [[sum(1 for a, b in zip(R[r], H[h]) if a and b) for h in hn] for r in rn]
    par = [(rn[i], hn[j]) for i, j in ungersk(overlapp)] if rn and hn else []
    nr = [0] * antal
    nh = [0] * antal
    for a in R.values():
        for k, x in enumerate(a):
            nr[k] += x
    for a in H.values():
        for k, x in enumerate(a):
            nh[k] += x
    ratt = [0] * antal
    for r, h in par:
        for k, (a, b) in enumerate(zip(R[r], H[h])):
            if a and b:
                ratt[k] += 1
    tal = sum(nr)
    missat = sum(max(0, a - b) for a, b in zip(nr, nh))
    falsk = sum(max(0, b - a) for a, b in zip(nr, nh))
    forvaxl = sum(min(a, b) - c for a, b, c in zip(nr, nh, ratt))
    aktiva, olika = 0, 0
    etiketter = sorted(set(R) | set(H))
    for k in range(antal):
        r = frozenset(t for t in etiketter if t in R and R[t][k])
        h = frozenset(t for t in etiketter if t in H and H[t][k])
        if r or h:
            aktiva += 1
            olika += r != h
    pct = (lambda x: round(100.0 * x / tal, 3)) if tal else (lambda x: None)
    return {
        "tal_s": round(tal * RAM, 2),
        "der": pct(missat + falsk + forvaxl), "missat": pct(missat),
        "falsklarm": pct(falsk), "forvaxling": pct(forvaxl),
        "etikettskillnad": round(100.0 * olika / aktiva, 3) if aktiva else None,
        "turer_ref": len(ref_seg), "turer_hyp": len(hyp_seg),
        "talare_ref": len(rn), "talare_hyp": len(hn),
        "matchning": {r: h for r, h in par},
    }


def turavvikelse(m):
    if not m["turer_ref"]:
        return None
    return round(100.0 * (m["turer_hyp"] - m["turer_ref"]) / m["turer_ref"], 3)


def band(experiment, der, tur):
    """Utredningens band för en jämförelse. Ger (band, skäl)."""
    dg, dr, tg, tr = BAND[experiment]
    if der is None or tur is None:
        return ROTT, "referensen saknar tal eller turer"
    skal = f"DER {der:.2f} %, turer {tur:+.1f} %"
    if der <= dg and abs(tur) <= tg:
        return GRONT, skal
    if der > dr or abs(tur) > tr:
        return ROTT, skal
    return GRATT, skal


def par_filer(ref, hyp):
    """Parar ihop filer: två filer direkt, eller två kataloger på filnamn."""
    if os.path.isdir(ref) != os.path.isdir(hyp):
        raise ValueError("--ref och --hyp måste vara två filer eller två kataloger")
    if not os.path.isdir(ref):
        return [(os.path.basename(ref), ref, hyp)]
    r = {f for f in os.listdir(ref) if f.endswith(".rttm")}
    h = {f for f in os.listdir(hyp) if f.endswith(".rttm")}
    if r != h:
        raise ValueError(f"filerna stämmer inte: bara i ref {sorted(r - h)}, bara i hyp {sorted(h - r)}")
    if not r:
        raise ValueError("inga .rttm-filer i katalogerna")
    return [(f, os.path.join(ref, f), os.path.join(hyp, f)) for f in sorted(r)]


def segment_for(sokvag):
    alla = las_rttm(sokvag)
    if len(alla) > 1:
        raise ValueError(f"{sokvag} har {len(alla)} uri:er; dela upp filen")
    return next(iter(alla.values()), [])


def main(argv):
    ap = argparse.ArgumentParser(prog="diarjamfor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="referensens RTTM (fil eller katalog)")
    ap.add_argument("--hyp", required=True, help="jämförd RTTM (fil eller katalog)")
    ap.add_argument("--experiment", type=int, choices=sorted(BAND), default=2)
    ap.add_argument("--ut", help="spara resultatet som JSON")
    args = ap.parse_args(argv[1:])

    try:
        par = par_filer(args.ref, args.hyp)
        rader, summa = [], {"tal": 0.0, "fel": 0.0, "m": 0.0, "f": 0.0, "c": 0.0, "tr": 0, "th": 0}
        for namn, r, h in par:
            m = jamfor(segment_for(r), segment_for(h))
            m["fil"], m["turavvikelse"] = namn, turavvikelse(m)
            m["band"], m["skal"] = band(args.experiment, m["der"], m["turavvikelse"])
            rader.append(m)
            if m["der"] is not None:
                vikt = m["tal_s"]
                summa["tal"] += vikt
                summa["fel"] += m["der"] * vikt
                summa["m"] += m["missat"] * vikt
                summa["f"] += m["falsklarm"] * vikt
                summa["c"] += m["forvaxling"] * vikt
            summa["tr"] += m["turer_ref"]
            summa["th"] += m["turer_hyp"]
    except (OSError, ValueError) as e:
        print(f"fel: {e}", file=sys.stderr)
        return 1

    def viktad(n):
        return round(summa[n] / summa["tal"], 3) if summa["tal"] else None
    total = {"fil": "totalt", "der": viktad("fel"), "missat": viktad("m"),
             "falsklarm": viktad("f"), "forvaxling": viktad("c"),
             "turer_ref": summa["tr"], "turer_hyp": summa["th"]}
    total["turavvikelse"] = turavvikelse(total)
    total["band"], total["skal"] = band(args.experiment, total["der"], total["turavvikelse"])

    print(f"Experiment {args.experiment}: {len(rader)} fil(er), referens = {args.ref}")
    print("| Fil | Turer ref → hyp | Avvikelse | DER % | Missat | Falsklarm | Förväxling | Etikettskillnad | Band |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in rader + [total]:
        print("| {} | {} → {} | {} | {} | {} | {} | {} | {} | {} |".format(
            m["fil"], m["turer_ref"], m["turer_hyp"],
            "–" if m["turavvikelse"] is None else f"{m['turavvikelse']:+.1f} %",
            *("–" if m.get(k) is None else f"{m[k]:.2f}"
              for k in ("der", "missat", "falsklarm", "forvaxling", "etikettskillnad")),
            m["band"]))
    if args.ut:
        with open(args.ut, "w", encoding="utf-8") as f:
            json.dump({"experiment": args.experiment, "filer": rader, "totalt": total},
                      f, ensure_ascii=False, indent=1)
    return 2 if total["band"] == ROTT else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
