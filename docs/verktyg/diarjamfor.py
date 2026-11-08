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
- turer: sammanhängande tal av en talare på 10 ms-rutnätet (segment med
  samma talare som överlappar eller möts slås ihop), och avvikelsen i
  procent; antalet RTTM-rader redovisas för sig;
- DER med referensen som facit, collar 0, överlapp inräknat, på 10 ms-ramar,
  med optimal talarmatchning (ungersk metod), uppdelad på missat, falsklarm
  och förväxling;
- etikettskillnad: andel av tidslinjens 10 ms-ramar där någon etiketts
  aktivitet skiljer sig, utan matchning (utredningens "same-label-
  aktivitetsskillnad"). Meningsfull när båda körningarna kommer från samma
  modell, som i experiment 2 och 3.

Med --facit (handmärkt eller forced-alignment-referens) mäts i stället
DER-skillnaden |DER(hyp, facit) − DER(ref, facit)|, som är storheten de
publicerade paritetsmätningarna redovisar. Utan facit är DER hypotesens fel
med referensen som facit. Utredningen anger "DER med BF16 som referens" men
kalibrerade tröskelvärdena på publicerade DER-skillnader; vilken som avses
är inte avgjort, så verktyget redovisar vilken det mätte.

Överlappande segment med samma talare räknas en gång, som i NIST:s md-eval.
pyannote.metrics räknar dem två gånger. Utan sådan självöverlapp ger verktyget
samma DER som pyannote.metrics 3.2.1 och 4.2.0 (collar 0, överlapp inräknat): största
skillnaden över 30 slumpade femminutersmöten var 0,0005 procentenheter.

Band (utredningens tröskelvärden, avsnitt 7):
    exp 1: GRÖNT DER ≤ 2 % och turer ±1 %; RÖTT DER > 6 % eller turer > ±3 %
    exp 2, 3: GRÖNT turer ±2 % och DER ≤ 0,5 %; RÖTT turer > ±6 % eller DER > 1,5 %
    däremellan GRÅTT.

Verktyget läser bara tider och etiketter, aldrig ljud eller text.

Avslutningskod: 0 = GRÖNT eller GRÅTT, 2 = totalen eller någon fil RÖTT,
1 = fel (även felaktiga argument).
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
    with open(sokvag, encoding="utf-8-sig") as f:
        for nr, rad in enumerate(f, 1):
            delar = rad.split()
            if not delar or delar[0] != "SPEAKER":
                continue
            if len(delar) < 8:
                raise ValueError(f"{sokvag}:{nr}: för få fält i SPEAKER-raden")
            try:
                start, langd = float(delar[3]), float(delar[4])
            except ValueError:
                raise ValueError(f"{sokvag}:{nr}: start eller längd är inte ett tal") from None
            if not (math.isfinite(start) and math.isfinite(langd)) or langd < 0 or start < 0:
                raise ValueError(f"{sokvag}:{nr}: ogiltig start eller längd")
            # Hela millisekunder, så att start + längd och nästa rads start hamnar på samma ram.
            s_ms, l_ms = round(start * 1000), round(langd * 1000)
            ut.setdefault(delar[1], []).append((s_ms / 1000, (s_ms + l_ms) / 1000, delar[7]))
    return ut


def ramar(segment, antal):
    """Aktivitet per talare som bytearray med en byte per 10 ms-ram."""
    akt = {}
    for start, slut, talare in segment:
        a = akt.setdefault(talare, bytearray(antal))
        i, j = round(start * 1000) // 10, round(slut * 1000) // 10
        for k in range(max(0, i), min(antal, j)):
            a[k] = 1
    return akt


def turer(akt):
    """Antal sammanhängande tal per talare på rutnätet (ihopslagna segment)."""
    n = 0
    for a in akt.values():
        foreg = 0
        for x in a:
            n += x and not foreg
            foreg = x
    return n


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
    # Ramar 0..antal−1 täcker [0, sista slut); samma heltals-ms-räkning som i ramar().
    antal = max([(round(e * 1000) + 9) // 10 for _, e, _ in ref_seg + hyp_seg] + [1])
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
    olika = 0
    etiketter = sorted(set(R) | set(H))
    tom = bytearray(antal)
    for k in range(antal):
        olika += any(R.get(t, tom)[k] != H.get(t, tom)[k] for t in etiketter)
    pct = (lambda x: round(100.0 * x / tal, 3)) if tal else (lambda x: None)
    ix = {t: i for i, t in enumerate(rn)}
    jx = {t: j for j, t in enumerate(hn)}
    return {
        "tal_s": round(tal * RAM, 2),
        "der": pct(missat + falsk + forvaxl), "missat": pct(missat),
        "falsklarm": pct(falsk), "forvaxling": pct(forvaxl),
        "etikettskillnad": round(100.0 * olika / antal, 3),
        "ramar": {"tal": tal, "missat": missat, "falsk": falsk, "forvaxl": forvaxl,
                  "olika": olika, "alla": antal},
        "turer_ref": turer(R), "turer_hyp": turer(H),
        "rader_ref": len(ref_seg), "rader_hyp": len(hyp_seg),
        "talare_ref": len(rn), "talare_hyp": len(hn),
        "matchning": {r: h for r, h in par if overlapp[ix[r]][jx[h]] > 0},
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
    der, tur = round(der, 2), round(tur, 2)        # bedöm det värde som visas
    skal = f"DER {der:.2f} %, turer {tur:+.2f} %"
    if der <= dg and abs(tur) <= tg:
        return GRONT, skal
    if der > dr or abs(tur) > tr:
        return ROTT, skal
    return GRATT, skal


def par_filer(ref, hyp, facit=None):
    """Parar ihop filer: två (eller tre) filer direkt, eller kataloger på filnamn."""
    sidor = [x for x in (ref, hyp, facit) if x]
    if len({os.path.isdir(x) for x in sidor}) != 1:
        raise ValueError("--ref, --hyp och --facit måste vara filer allihop eller kataloger allihop")
    if not os.path.isdir(ref):
        return [(os.path.basename(ref), ref, hyp, facit)]
    namn = [{f for f in os.listdir(x) if f.endswith(".rttm")} for x in sidor]
    if any(n != namn[0] for n in namn[1:]):
        raise ValueError("filerna i katalogerna stämmer inte: " + "; ".join(
            f"{x}: {sorted(n ^ namn[0])}" for x, n in zip(sidor[1:], namn[1:]) if n != namn[0]))
    if not namn[0]:
        raise ValueError("inga .rttm-filer i katalogerna")
    return [(f, os.path.join(ref, f), os.path.join(hyp, f),
             os.path.join(facit, f) if facit else None) for f in sorted(namn[0])]


def segment_for(sokvag):
    alla = las_rttm(sokvag)
    if len(alla) > 1:
        raise ValueError(f"{sokvag} har {len(alla)} uri:er; dela upp filen")
    return next(iter(alla.values()), [])


class Tolk(argparse.ArgumentParser):
    """Felaktiga argument ger kod 1, så att kod 2 alltid betyder RÖTT."""
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: fel: {message}", file=sys.stderr)
        sys.exit(1)


def main(argv):
    ap = Tolk(prog="diarjamfor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="referensens RTTM (fil eller katalog), t.ex. BF16")
    ap.add_argument("--hyp", required=True, help="jämförd RTTM (fil eller katalog), t.ex. Q8_0")
    ap.add_argument("--facit", help="handmärkt facit; då mäts DER-skillnaden mot facit")
    ap.add_argument("--experiment", type=int, choices=sorted(BAND), default=2)
    ap.add_argument("--ut", help="spara resultatet som JSON")
    args = ap.parse_args(argv[1:])

    try:
        rader = []
        summa = {"tal": 0, "missat": 0, "falsk": 0, "forvaxl": 0, "olika": 0, "alla": 0,
                 "tr": 0, "th": 0, "rr": 0, "rh": 0, "fel_ref": 0, "fel_hyp": 0, "tal_facit": 0}
        for namn, r, h, f in par_filer(args.ref, args.hyp, args.facit):
            ref_seg, hyp_seg = segment_for(r), segment_for(h)
            m = jamfor(ref_seg, hyp_seg)
            m["fil"], m["turavvikelse"] = namn, turavvikelse(m)
            if f:
                fac = segment_for(f)
                mr, mh = jamfor(fac, ref_seg), jamfor(fac, hyp_seg)
                m["der_ref_facit"], m["der_hyp_facit"] = mr["der"], mh["der"]
                m["der_matt"] = (None if mr["der"] is None or mh["der"] is None
                                 else round(abs(mh["der"] - mr["der"]), 3))
                fr = mr["ramar"]
                summa["fel_ref"] += fr["missat"] + fr["falsk"] + fr["forvaxl"]
                hr = mh["ramar"]
                summa["fel_hyp"] += hr["missat"] + hr["falsk"] + hr["forvaxl"]
                summa["tal_facit"] += fr["tal"]
            else:
                m["der_matt"] = m["der"]
            if m["ramar"]["tal"] == 0 and m["talare_hyp"] == 0 and (not f or m.get("der_matt") is None):
                m["band"], m["skal"] = GRONT, "inget tal i någon av filerna"
            else:
                m["band"], m["skal"] = band(args.experiment, m["der_matt"], m["turavvikelse"])
            rader.append(m)
            for k in ("tal", "missat", "falsk", "forvaxl", "olika", "alla"):
                summa[k] += m["ramar"][k]
            summa["tr"] += m["turer_ref"]
            summa["th"] += m["turer_hyp"]
            summa["rr"] += m["rader_ref"]
            summa["rh"] += m["rader_hyp"]

        def andel(k):
            return round(100.0 * summa[k] / summa["tal"], 3) if summa["tal"] else None
        total = {"fil": "totalt", "missat": andel("missat"), "falsklarm": andel("falsk"),
                 "forvaxling": andel("forvaxl"),
                 "der": (round(100.0 * (summa["missat"] + summa["falsk"] + summa["forvaxl"]) / summa["tal"], 3)
                         if summa["tal"] else None),
                 "etikettskillnad": round(100.0 * summa["olika"] / summa["alla"], 3) if summa["alla"] else None,
                 "turer_ref": summa["tr"], "turer_hyp": summa["th"],
                 "rader_ref": summa["rr"], "rader_hyp": summa["rh"],
                 "storsta_turavvikelse": max((abs(m["turavvikelse"]) for m in rader
                                              if m["turavvikelse"] is not None), default=None)}
        total["turavvikelse"] = turavvikelse(total)
        if args.facit:
            t = summa["tal_facit"]
            total["der_matt"] = (round(abs(summa["fel_hyp"] - summa["fel_ref"]) * 100.0 / t, 3)
                                 if t else None)
        else:
            total["der_matt"] = total["der"]
        if summa["tal"] == 0 and all(m["talare_hyp"] == 0 for m in rader):
            total["band"], total["skal"] = GRONT, "inget tal i någon av filerna"
        else:
            total["band"], total["skal"] = band(args.experiment, total["der_matt"], total["turavvikelse"])

        matt = "DER-skillnad mot facit" if args.facit else "DER med ref som facit"
        print(f"Experiment {args.experiment}: {len(rader)} fil(er), referens = {args.ref}, "
              f"bandets DER = {matt}")
        print("| Fil | Turer ref → hyp | Avvikelse | DER % | Missat | Falsklarm | Förväxling "
              "| Bandets DER | Etikettskillnad | Band |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m in rader + [total]:
            print("| {} | {} → {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                m["fil"], m["turer_ref"], m["turer_hyp"],
                "–" if m["turavvikelse"] is None else f"{m['turavvikelse']:+.2f} %",
                *("–" if m.get(k) is None else f"{m[k]:.2f}"
                  for k in ("der", "missat", "falsklarm", "forvaxling", "der_matt", "etikettskillnad")),
                m["band"]))
        if args.ut:
            with open(args.ut, "w", encoding="utf-8") as f:
                json.dump({"experiment": args.experiment, "bandets_der": matt,
                           "filer": rader, "totalt": total}, f, ensure_ascii=False, indent=1)
    except (OSError, ValueError) as e:
        print(f"fel: {e}", file=sys.stderr)
        return 1
    rott = total["band"] == ROTT or any(m["band"] == ROTT for m in rader)
    return 2 if rott else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
