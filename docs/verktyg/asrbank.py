#!/usr/bin/env python3
"""asrbank — mäter KB-Whisper i olika precision och storlek: WER och RTF.

Utredning 2 (docs/utredningar/svensk-talarseparerad-transkription-2026.md,
avsnitt 7) har två ASR-frågor som ingen har publicerat svar på:
    exp 5: tappar KB-Whisper large ordfel i int8 jämfört med fp16 på GPU?
    exp 6: klarar CPU-noden medium, eller måste den köra small?
Det här verktyget kör varianterna på samma ljud och sätter utredningens band.

Användning:
    python3 asrbank.py --manifest fleurs-sv-test.tsv --experiment 5
    python3 asrbank.py --manifest egna.tsv --experiment 6 --tradar 4
    python3 asrbank.py --manifest m.tsv --varianter large:float16:cuda,large:int8:cuda
    python3 asrbank.py --attrapp --experiment 5        # torrkörning utan beroenden

Manifestet är en TSV med två kolumner: ljudfil och referenstext. Sökvägarna är
relativa till manifestet. Avkodningen är fast: svenska, beam 5,
condition_on_previous_text av (KBLab:s råd mot hallucinationer), ingen VAD.

WER räknas på hela korpusen (summa fel delat med summa referensord) efter
normalisering: gemener, skiljetecken bort, blanksteg ihopslagna. Siffror
normaliseras inte; använd samma referenser för alla varianter. RTF är
beräkningstid delat med ljudlängd (lägre är bättre).

Band (utredningens tröskelvärden):
    exp 5: ΔWER(int8 − fp16) ≤ 0,3 pe GRÖNT; 0,3–1,0 GRÅTT; > 1,0 RÖTT
    exp 6: medium RTF ≤ 0,5 GRÖNT; 0,5–1,0 GRÅTT; > 1,0 RÖTT

Mät bara på grönt ljud, till exempel FLEURS sv_se test (CC BY 4.0) eller
påhittade inspelningar. Bänken sparar inga transkript, bara tal.

Avslutningskod: 0 = klart, 1 = fel.
"""
import argparse
import json
import os
import sys
import threading
import time
import unicodedata

GRONT, GRATT, ROTT, OMATT = "GRÖNT", "GRÅTT", "RÖTT", "OMÄTT"
STORLEKAR = ("tiny", "base", "small", "medium", "large")
STANDARD = {
    5: ["large:float16:cuda", "large:int8:cuda"],
    6: ["small:int8:cpu", "medium:int8:cpu"],
}


# ---------------------------------------------------------------- rena funktioner

def normalisera(text):
    """Gemener, skiljetecken till blanksteg, blanksteg ihopslagna. Å, ä och ö behålls."""
    ut = []
    for tecken in unicodedata.normalize("NFC", text.lower()):
        ut.append(" " if unicodedata.category(tecken).startswith(("P", "S")) else tecken)
    return " ".join("".join(ut).split())


def ordfel(ref, hyp):
    """Levenshtein på ordnivå: antal ersättningar + strykningar + tillägg."""
    r, h = ref.split(), hyp.split()
    foreg = list(range(len(h) + 1))
    for i, rv in enumerate(r, 1):
        rad = [i] + [0] * len(h)
        for j, hv in enumerate(h, 1):
            rad[j] = min(foreg[j] + 1, rad[j - 1] + 1, foreg[j - 1] + (rv != hv))
        foreg = rad
    return foreg[-1], len(r)


def wer(par):
    """Korpus-WER i procent ur (referens, hypotes)-par, efter normalisering."""
    fel = ord_ = 0
    for ref, hyp in par:
        f, n = ordfel(normalisera(ref), normalisera(hyp))
        fel, ord_ = fel + f, ord_ + n
    return round(100.0 * fel / ord_, 3) if ord_ else None


def tolka_variant(text):
    """'large:int8:cuda' → (modell-id, compute_type, enhet). Storlek eller fullt id."""
    delar = text.split(":")
    if len(delar) != 3 or not all(delar):
        raise ValueError(f"variant '{text}': skriv modell:compute_type:enhet")
    modell, prec, enhet = delar
    if modell in STORLEKAR:
        modell = f"KBLab/kb-whisper-{modell}"
    if enhet not in ("cuda", "cpu"):
        raise ValueError(f"variant '{text}': enheten ska vara cuda eller cpu")
    return modell, prec, enhet


def band(experiment, resultat):
    """Utredningens band. resultat: lista av variant-dikt i den ordning de kördes."""
    if experiment == 5:
        fp16 = next((r for r in resultat if r["precision"] in ("float16", "bfloat16")), None)
        int8 = next((r for r in resultat if r["precision"].startswith("int8")), None)
        if not fp16 or not int8 or fp16.get("wer") is None or int8.get("wer") is None:
            return OMATT, "behöver en fp16- och en int8-variant med WER"
        d = round(int8["wer"] - fp16["wer"], 3)
        skal = f"ΔWER {d:+.2f} pe ({int8['precision']} mot {fp16['precision']})"
        if d <= 0.3:
            return GRONT, skal
        return (GRATT if d <= 1.0 else ROTT), skal
    if experiment == 6:
        medium = next((r for r in resultat if r["modell"].endswith("medium")), None)
        small = next((r for r in resultat if r["modell"].endswith("small")), None)
        if not medium or medium.get("rtf") is None:
            return OMATT, "behöver en medium-variant med RTF"
        vinst = ""
        if small and small.get("wer") is not None and medium.get("wer") is not None:
            vinst = f", WER {small['wer'] - medium['wer']:+.2f} pe bättre än small"
        skal = f"medium RTF {medium['rtf']:.2f}{vinst}"
        if medium["rtf"] <= 0.5:
            return GRONT, skal
        return (GRATT if medium["rtf"] <= 1.0 else ROTT), skal
    return OMATT, "inget band för detta experiment"


def las_manifest(sokvag):
    """TSV: ljudfil<TAB>referens. Ger [(absolut sökväg, referens)]."""
    bas = os.path.dirname(os.path.abspath(sokvag))
    ut = []
    with open(sokvag, encoding="utf-8") as f:
        for nr, rad in enumerate(f, 1):
            rad = rad.rstrip("\n")
            if not rad.strip() or rad.startswith("#"):
                continue
            delar = rad.split("\t")
            if len(delar) != 2 or not delar[1].strip():
                raise ValueError(f"{sokvag}:{nr}: väntade ljudfil<TAB>referens")
            ut.append((os.path.join(bas, delar[0]), delar[1]))
    if not ut:
        raise ValueError(f"{sokvag}: inga rader")
    return ut


# ---------------------------------------------------------------- motorer

class Attrapp:
    """Simulerad motor för prov: varje variant gör ett känt antal fel."""
    namn = "attrapp"
    FEL = {("large", "float16"): 0, ("large", "int8"): 1, ("small", "int8"): 3, ("medium", "int8"): 2}
    RTF = {("large", "float16"): 0.05, ("large", "int8"): 0.05, ("small", "int8"): 0.3, ("medium", "int8"): 0.7}

    def __init__(self, modell, prec, enhet, tradar):
        self.nyckel = (modell.rsplit("-", 1)[-1], prec)

    def transkribera(self, ljud, ref):
        """Ger (text, beräkningstid s, ljudlängd s)."""
        ord_ = ref.split()
        for i in range(min(self.FEL.get(self.nyckel, 0), len(ord_))):
            ord_[i] = "x"
        return " ".join(ord_), 10.0 * self.RTF.get(self.nyckel, 1.0), 10.0


class FasterWhisper:
    """Riktig motor: faster-whisper (CTranslate2) med KB-Whisper."""
    namn = "faster-whisper"

    def __init__(self, modell, prec, enhet, tradar, revision=None):
        from faster_whisper import WhisperModel
        kw = {"revision": revision} if revision else {}
        self.m = WhisperModel(modell, device=enhet, compute_type=prec, cpu_threads=tradar, **kw)

    def transkribera(self, ljud, ref):
        """Ger (text, beräkningstid s, ljudlängd s). Referensen används inte här."""
        t0 = time.perf_counter()
        segment, info = self.m.transcribe(ljud, language="sv", beam_size=5,
                                          condition_on_previous_text=False, vad_filter=False)
        text = " ".join(s.text.strip() for s in segment)   # generatorn körs här
        return text, time.perf_counter() - t0, info.duration


class Minne:
    """Toppen för använt grafikminne under en variant, om pynvml finns (hela kortet)."""
    def __init__(self, aktiv):
        self.topp, self._stopp, self.h = None, threading.Event(), None
        if not aktiv:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:                        # noqa: BLE001 — utan pynvml blir det OMÄTT
            self.h = None

    def __enter__(self):
        if self.h is not None:
            self.topp = 0
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stopp.wait(0.2):
            self.topp = max(self.topp, self.nv.nvmlDeviceGetMemoryInfo(self.h).used)

    def __exit__(self, *a):
        self._stopp.set()
        if self.topp is not None:
            self.topp = round(self.topp / 1e9, 2)


# ---------------------------------------------------------------- bänken

def kor_variant(Motor, variant, filer, args):
    modell, prec, enhet = tolka_variant(variant)
    print(f"… {modell} {prec} på {enhet}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    motor = Motor(modell, prec, enhet, args.tradar, **({"revision": args.revision}
                                                         if args.revision and Motor is FasterWhisper else {}))
    laddning = time.perf_counter() - t0
    par, berakning, ljud_s = [], 0.0, 0.0
    with Minne(enhet == "cuda" and Motor is FasterWhisper) as minne:
        for ljud, ref in filer:
            text, sek, langd = motor.transkribera(ljud, ref)
            par.append((ref, text))
            berakning += sek
            ljud_s += langd
    return {"variant": variant, "modell": modell, "precision": prec, "enhet": enhet,
            "filer": len(filer), "ljud_s": round(ljud_s, 2), "laddning_s": round(laddning, 2),
            "wer": wer(par), "rtf": round(berakning / ljud_s, 4) if ljud_s else None,
            "vram_topp_gb": minne.topp}


def main(argv):
    ap = argparse.ArgumentParser(prog="asrbank", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", help="TSV: ljudfil<TAB>referenstext")
    ap.add_argument("--experiment", type=int, choices=sorted(STANDARD), default=5)
    ap.add_argument("--varianter", help="kommaseparerat modell:compute_type:enhet; standard beror på experimentet")
    ap.add_argument("--tradar", type=int, default=0, help="CPU-trådar (0 = CTranslate2 väljer)")
    ap.add_argument("--revision", help="KB-Whisper-revision, t.ex. strict eller subtitle")
    ap.add_argument("--ut", help="spara resultatet som JSON")
    ap.add_argument("--attrapp", action="store_true", help="simulerad motor (prov)")
    args = ap.parse_args(argv[1:])

    varianter = args.varianter.split(",") if args.varianter else STANDARD[args.experiment]
    try:
        for v in varianter:
            tolka_variant(v)
        if args.attrapp:
            Motor = Attrapp
            mening = "det här är en påhittad mening på svenska för att prova bänken"
            filer = [("attrapp.wav", " ".join([mening] * 40))] * 3     # 480 ord per fil
        else:
            if not args.manifest:
                raise ValueError("ange --manifest (eller --attrapp för en torrkörning)")
            Motor = FasterWhisper
            filer = las_manifest(args.manifest)
            saknas = [f for f, _ in filer if not os.path.exists(f)]
            if saknas:
                raise ValueError(f"{len(saknas)} ljudfiler saknas, t.ex. {saknas[0]}")
        resultat = [kor_variant(Motor, v, filer, args) for v in varianter]
    except Exception as e:                       # noqa: BLE001 — skrivs ut, aldrig tyst
        print(f"fel: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    utfall, skal = band(args.experiment, resultat)
    print(f"Experiment {args.experiment}: {len(filer)} filer, {resultat[0]['ljud_s']:.0f} s ljud")
    print("| Variant | WER % | RTF | Laddning s | VRAM topp |")
    print("|---|---|---|---|---|")
    for r in resultat:
        print("| {} | {} | {} | {} | {} |".format(
            r["variant"], OMATT if r["wer"] is None else f"{r['wer']:.2f}",
            OMATT if r["rtf"] is None else f"{r['rtf']:.3f}", r["laddning_s"],
            OMATT if r["vram_topp_gb"] is None else f"{r['vram_topp_gb']} GB"))
    print(f"\n**{utfall}**: {skal}")
    if args.ut:
        with open(args.ut, "w", encoding="utf-8") as f:
            json.dump({"experiment": args.experiment, "motor": Motor.namn, "band": utfall,
                       "skal": skal, "resultat": resultat}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
