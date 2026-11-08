#!/usr/bin/env python3
"""cpwer — ordfel med talare (cpWER) för talarseparerade transkript.

Utredning 2 (docs/utredningar/svensk-talarseparerad-transkription-2026.md,
avsnitt 7) mäter hela kedjan, transkription plus diarisering, med cpWER
(experiment 8, och som delmått i experiment 7). Experiment 7:s huvudmått,
andel ord med korrekt talare, kräver ordtider och ingår inte här (OMÄTT). cpWER är WER efter att varje talares ord har lagts i
följd och talarna i hypotesen har parats med talarna i facit så att felen
blir så få som möjligt (CHiME-6). Ett ord som hamnar hos fel talare kostar
alltså, även om det är rätt stavat.

Användning:
    python3 cpwer.py --ref facit.tsv --hyp kedja-a.tsv
    python3 cpwer.py --ref facit/ --hyp nemotron/ --jamfor whisperx/

Transkripten är TSV-filer med en replik per rad i tidsordning:
talare<TAB>text. Med kataloger paras filerna ihop på filnamn.

Mått per fil och totalt:
- cpWER: fel med talarparning delat med antal ord i facit;
- WER utan talare: alla repliker i filernas ordning, talarna bortsedda. Den
  beror på hur repliker från olika talare är ordnade (till exempel inskott
  under en annans tur) och är därför ingen undre gräns för cpWER; skillnaden
  mellan dem kan bli negativ och redovisas inte som "talarkostnad".

Med --jamfor sätts experiment 8:s band, och riktningen är fast:
--hyp är Nemotron-kedjan (KB-Whisper + Nemotron BF16), --jamfor är
whisperX-kedjan (KB-Whisper + community-1). GRÖNT om Nemotron-kedjans cpWER
≤ whisperX-kedjans, GRÅTT om den ligger högst 1 procentenhet över, RÖTT
annars. Det är bara cpWER-halvan av kriteriet: utredningens krav på
grafikminne (≤ 5,0 GB för hela kedjan på 6 GB-kortet) saknar ännu verktyg,
eftersom diarbank.py inte kör Nemotron. Bandet sparas därför som band_cpwer.

Normaliseringen och ordfelen är desamma som i asrbank.py, och talarparningen
använder diarjamfor.py:s ungerska metod. De tre filerna hör ihop.

Avslutningskod: 0 = klart (GRÖNT eller GRÅTT med --jamfor), 2 = RÖTT,
1 = fel (även tomt facit och felaktiga argument).
"""
import argparse
import importlib.util
import json
import os
import sys

GRONT, GRATT, ROTT = "GRÖNT", "GRÅTT", "RÖTT"
HAR = os.path.dirname(os.path.realpath(__file__))   # realpath: fungerar även via symbolisk länk


def _modul(namn):
    sokvag = os.path.join(HAR, namn + ".py")
    if not os.path.exists(sokvag):
        sys.exit(f"fel: {namn}.py saknas bredvid cpwer.py; de tre verktygen ska ligga i samma mapp")
    spec = importlib.util.spec_from_file_location(namn, sokvag)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_asr, _dj = _modul("asrbank"), _modul("diarjamfor")
normalisera, ordfel, ungersk = _asr.normalisera, _asr.ordfel, _dj.ungersk


def las(sokvag):
    """TSV: talare<TAB>text. Ger [(talare, normaliserad text)] i tidsordning."""
    ut = []
    with open(sokvag, encoding="utf-8-sig") as f:
        for nr, rad in enumerate(f, 1):
            rad = rad.rstrip("\n")
            if not rad.strip() or rad.startswith("#"):
                continue
            delar = rad.split("\t")
            if len(delar) != 2 or not delar[0].strip():
                raise ValueError(f"{sokvag}:{nr}: väntade talare<TAB>text")
            ut.append((delar[0].strip(), normalisera(delar[1])))
    return ut


def per_talare(repliker):
    ord_ = {}
    for talare, text in repliker:
        ord_.setdefault(talare, []).extend(text.split())
    return {t: " ".join(o) for t, o in ord_.items()}


def cpwer(ref, hyp):
    """Ger fel, ord i facit, parning och WER utan talare för en fil."""
    R, H = per_talare(ref), per_talare(hyp)
    rn, hn = sorted(R), sorted(H)
    n_ref = sum(len(R[t].split()) for t in rn)
    # Kvadratisk kostnadsmatris: riktiga par, och "ingen motpart" för överblivna talare.
    n = len(rn) + len(hn)
    STOR = 10 ** 9
    kost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i < len(rn) and j < len(hn):
                kost[i][j] = ordfel(R[rn[i]], H[hn[j]])[0]
            elif i < len(rn):                                    # facit-talare utan motpart
                kost[i][j] = len(R[rn[i]].split()) if j - len(hn) == i else STOR
            elif j < len(hn):                                    # hypotes-talare utan motpart
                kost[i][j] = len(H[hn[j]].split()) if i - len(rn) == j else STOR
    if n:
        storst = max(max(r) for r in kost)
        par = ungersk([[storst - c for c in rad] for rad in kost])
    else:
        par = []
    fel = sum(kost[i][j] for i, j in par)
    parning = {rn[i]: hn[j] for i, j in par if i < len(rn) and j < len(hn)}
    utan = ordfel(" ".join(t for _, t in ref), " ".join(t for _, t in hyp))[0]
    return {"fel": fel, "ord": n_ref, "fel_utan_talare": utan, "parning": parning,
            "talare_ref": len(rn), "talare_hyp": len(hn)}


def procent(fel, ord_):
    return round(100.0 * fel / ord_, 3) if ord_ else None


def band8(a, b):
    """a = Nemotron-kedjans cpWER, b = whisperX-kedjans."""
    d = round(a - b, 2)
    skal = f"cpWER Nemotron-kedjan {a:.2f} % mot whisperX-kedjan {b:.2f} % ({d:+.2f} pe)"
    if d <= 0:
        return GRONT, skal
    return (GRATT if d <= 1.0 else ROTT), skal


def par_filer(*sidor):
    sidor = [x for x in sidor if x]
    if len({os.path.isdir(x) for x in sidor}) != 1:
        raise ValueError("alla sökvägar måste vara filer eller alla kataloger")
    if not os.path.isdir(sidor[0]):
        return [(os.path.basename(sidor[0]), *sidor)]
    namn = [{f for f in os.listdir(x) if f.endswith(".tsv")} for x in sidor]
    if any(n != namn[0] for n in namn[1:]):
        raise ValueError("filerna i katalogerna stämmer inte överens")
    if not namn[0]:
        raise ValueError("inga .tsv-filer i katalogerna")
    return [(f, *(os.path.join(x, f) for x in sidor)) for f in sorted(namn[0])]


class Tolk(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: fel: {message}", file=sys.stderr)
        sys.exit(1)


def main(argv):
    ap = Tolk(prog="cpwer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="facit (fil eller katalog)")
    ap.add_argument("--hyp", required=True, help="Nemotron-kedjan, eller vilken kedja som helst utan --jamfor")
    ap.add_argument("--jamfor", help="whisperX-kedjan; sätter experiment 8:s band (bara cpWER-halvan)")
    ap.add_argument("--ut", help="spara resultatet som JSON")
    args = ap.parse_args(argv[1:])

    try:
        rader, summa = [], {"ord": 0, "a": 0, "a_utan": 0, "b": 0, "b_utan": 0}
        for namn, r, a, *b in par_filer(args.ref, args.hyp, args.jamfor):
            ref = las(r)
            ma = cpwer(ref, las(a))
            rad = {"fil": namn, "ord": ma["ord"], "cpwer_a": procent(ma["fel"], ma["ord"]),
                   "wer_utan_talare_a": procent(ma["fel_utan_talare"], ma["ord"]),
                   "parning_a": ma["parning"], "talare_ref": ma["talare_ref"], "talare_a": ma["talare_hyp"]}
            summa["ord"] += ma["ord"]
            summa["a"] += ma["fel"]
            summa["a_utan"] += ma["fel_utan_talare"]
            if b:
                mb = cpwer(ref, las(b[0]))
                rad.update({"cpwer_b": procent(mb["fel"], mb["ord"]),
                            "wer_utan_talare_b": procent(mb["fel_utan_talare"], mb["ord"]),
                            "talare_b": mb["talare_hyp"]})
                summa["b"] += mb["fel"]
                summa["b_utan"] += mb["fel_utan_talare"]
            rader.append(rad)
        if not summa["ord"]:
            raise ValueError("facit saknar ord")
        total = {"fil": "totalt", "ord": summa["ord"], "cpwer_a": procent(summa["a"], summa["ord"]),
                 "wer_utan_talare_a": procent(summa["a_utan"], summa["ord"])}
        utfall = None
        if args.jamfor:
            total["cpwer_b"] = procent(summa["b"], summa["ord"])
            total["wer_utan_talare_b"] = procent(summa["b_utan"], summa["ord"])
            utfall = band8(total["cpwer_a"], total["cpwer_b"])

        def p(x):
            return "–" if x is None else f"{x:.2f}"
        kol = "| Fil | Ord | cpWER | WER utan talare |"
        if args.jamfor:
            kol = "| Fil | Ord | cpWER Nemotron | WER utan talare | cpWER whisperX | WER utan talare |"
        print(kol)
        print("|" + "---|" * (kol.count("|") - 1))
        for m in rader + [total]:
            rad = f"| {m['fil']} | {m['ord']} | {p(m['cpwer_a'])} | {p(m['wer_utan_talare_a'])} |"
            if args.jamfor:
                rad += f" {p(m.get('cpwer_b'))} | {p(m.get('wer_utan_talare_b'))} |"
            print(rad)
        if utfall:
            print(f"\n**{utfall[0]}** (experiment 8, bara cpWER-halvan; VRAM-halvan OMÄTT): {utfall[1]}")
        if args.ut:
            with open(args.ut, "w", encoding="utf-8") as f:
                json.dump({"experiment": 8 if utfall else None, "filer": rader, "totalt": total,
                           "band_cpwer": utfall[0] if utfall else None,
                           "delbedomning": "bara cpWER; VRAM-kravet saknar verktyg" if utfall else None,
                           "skal": utfall[1] if utfall else None},
                          f, ensure_ascii=False, indent=1)
    except (OSError, ValueError) as e:
        print(f"fel: {e}", file=sys.stderr)
        return 1
    return 2 if utfall and utfall[0] == ROTT else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
