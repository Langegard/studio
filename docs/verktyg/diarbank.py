#!/usr/bin/env python3
"""diarbank — kör bänkplanen i utredningen om diarisering på RTX 3080 (avsnitt 4).

Utredningen (docs/utredningar/diarisering-rtx3080.md) beskriver varför pyannotes
community-1 går långsamt på ett Ampere-kort och vilka inställningar som borde ge
5–10× realtid. Det här verktyget mäter dem, en variabel i taget, och sätter
utredningens egna band på varje körning: GRÖNT, GRÅTT, RÖTT eller OMÄTT.

Användning:
    HF_TOKEN=… python3 diarbank.py --ljud klipp.wav [--ref klipp.rttm] [--talare N]
    python3 diarbank.py --lista              # visa planen utan att köra
    python3 diarbank.py --attrapp            # torrkörning utan torch, för prov

Körningar (utredningens beteckningar):
    B    baslinje: community-1 med standardvärden, ljud via filsökväg
    R1   ljudet förladdat i minnet ({"waveform", "sample_rate"})
    R2   TF32 påslaget (pyannotes fix_reproducibility slår annars av det)
    R3   segmentation_batch_size och embedding_batch_size 32, 64, 128
    R4a  segmentation_step 0,1 → 0,2
    R4b  segmentation_step 0,1 → 0,4
    R7   num_speakers känt (bara med --talare)
    R8   kombinationen av alla gröna bland R1–R7
R5 (fp16 i embedding), R6 (annan klustring) och R9–R11 (andra motorer) kräver
kodändringar eller andra paket och ingår inte; de rapporteras som OMÄTT.

Mät bara på grönt ljud: offentliga testklipp (t.ex. AMI ES2004a, CC BY 4.0)
eller påhittade inspelningar. Bänken sparar inga transkript, bara tider,
minne och diariseringsfel. Token för Hugging Face läses ur en miljövariabel
och skrivs aldrig ut.

Avslutningskod: 0 = klart, 2 = rimlighetskontrollen föll (pipelinen ligger
inte på GPU:n), 1 = fel.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

GRONT, GRATT, ROTT, OMATT = "GRÖNT", "GRÅTT", "RÖTT", "OMÄTT"

# Varianter: namn → (beskrivning, inställningar). Allt som inte anges = baslinjen.
VARIANTER = [
    ("B", "baslinje", {}),
    ("R1", "ljud i minnet", {"minne": True}),
    ("R2", "TF32 på", {"tf32": True}),
    ("R3-32", "batch 32", {"batch": 32}),
    ("R3-64", "batch 64", {"batch": 64}),
    ("R3-128", "batch 128", {"batch": 128}),
    ("R4a", "segmentation_step 0,2", {"steg": 0.2}),
    ("R4b", "segmentation_step 0,4", {"steg": 0.4}),
    ("R7", "num_speakers känt", {"talare": True}),
]
EJ_I_BANKEN = {
    "R5": "fp16 i embedding kräver en patch enligt pyannote PR #1996",
    "R6": "annan klustring kräver en annan pipeline-konfiguration",
    "R9": "Nemotron (NeMo, GPU) är en annan motor",
    "R10": "Nemotron (ONNX, CPU) är en annan motor",
    "R11": "whisperX mäts i sin egen miljö",
}


# ---------------------------------------------------------------- rena funktioner

def rtfx(ljud_s, vagg_s):
    """Realtidsfaktor: ljudlängd delat med väggtid (NVIDIA:s definition)."""
    if not ljud_s or not vagg_s or vagg_s <= 0:
        return None
    return ljud_s / vagg_s


def stegtider(handelser, slut):
    """Tid per steg ur hook-anropen.

    handelser: lista av (tidpunkt, stegnamn) i anropsordning. I pyannote 4.0.7
    anropas hook redan när segmenteringen och embeddingarna börjar (completed=0),
    så ett steg räknas från sitt första anrop till nästa stegs första anrop (eller
    till slut). Klustringen har ingen egen hook; den ligger mellan embeddingstegets
    sista anrop och nästa steg och redovisas som "klustring".
    """
    forsta, sista = [], {}
    for t, steg in handelser:
        if steg not in sista:
            forsta.append((t, steg))
        sista[steg] = t
    ut = {}
    for i, (t, steg) in enumerate(forsta):
        nasta = forsta[i + 1][0] if i + 1 < len(forsta) else slut
        ut[steg] = round(max(0.0, nasta - t), 3)
        if steg == "embeddings":
            ut["klustring"] = round(max(0.0, nasta - sista[steg]), 3)
    return ut


def andel(r, b):
    """Hastighetsökning som andel: 0,25 betyder 25 % fler sekunder ljud per sekund."""
    if r is None or b is None or b == 0:
        return None
    return round(r / b - 1.0, 9)   # avrundat, så att +20 % inte blir 0,1999… vid gränsen


def band(kod, mat, bas, cfg):
    """Utredningens utfallsband (tabell 4.4) för en körning. Ger (band, skäl)."""
    def har(*v):
        return all(x is not None for x in v)

    if mat.get("oom"):
        return ROTT, "slut på grafikminne"
    if kod == "B":
        varm, kmv = mat.get("rtfx_varm"), mat.get("kall_minus_varm")
        if varm is None:
            return OMATT, "ingen varm körning"
        if varm < 3:
            return ROTT, f"varm RTFx {varm:.1f}× < 3×"
        if varm >= 10 or (kmv is not None and kmv <= 15):
            return GRONT, f"varm RTFx {varm:.1f}×, kall−varm {fmt(kmv, 's')}"
        return GRATT, f"varm RTFx {varm:.1f}×"

    s = andel(mat.get("rtfx_varm"), bas.get("rtfx_varm"))
    f = None if s is None else s + 1.0
    d = mat.get("der_delta_pp")
    if s is None:
        return OMATT, "ingen jämförbar tid"
    if kod == "R1":
        if s >= 0.20:
            return GRONT, f"{s:+.0%}"
        return (GRATT if s >= -0.10 else ROTT), f"{s:+.0%}"
    if kod == "R2":
        if not har(d):
            return OMATT, f"{s:+.0%}, diariseringsfel omätt"
        if s >= 0.25 and d <= 0.3:
            return GRONT, f"{s:+.0%}, DER-Δ {d:+.2f}"
        if s < 0.05 or d > 0.3:
            return ROTT, f"{s:+.0%}, DER-Δ {d:+.2f}"
        return GRATT, f"{s:+.0%}, DER-Δ {d:+.2f}"
    if kod.startswith("R3"):
        res = mat.get("vram_reserverat_gb")
        if s < 0.05:
            return ROTT, f"{s:+.0%}"
        if s >= 0.20 and res is not None and res < cfg["vram_tak_gb"]:
            return GRONT, f"{s:+.0%}, reserverat {res:.1f} GB"
        return GRATT, f"{s:+.0%}, reserverat {fmt(res, ' GB')}"
    if kod in ("R4a", "R4b"):
        if not har(d):
            return OMATT, f"{f:.2f}×, diariseringsfel omätt"
        gron, rod, dg, dr = ((1.6, 1.2, 0.5, 1.5) if kod == "R4a"
                             else (2.5, 1.5, 1.0, 3.0))
        fel_talare = kod == "R4b" and mat.get("talarfel")
        if f >= gron and d <= dg and not fel_talare:
            return GRONT, f"{f:.2f}×, DER-Δ {d:+.2f}"
        if f < rod or d > dr or fel_talare:
            extra = ", fel antal talare" if fel_talare else ""
            return ROTT, f"{f:.2f}×, DER-Δ {d:+.2f}{extra}"
        return GRATT, f"{f:.2f}×, DER-Δ {d:+.2f}"
    if kod == "R7":
        return (GRONT if s >= 0.15 else GRATT), f"{s:+.0%} (informativt)"
    if kod == "R8":
        varm = mat.get("rtfx_varm")
        if varm < 5:
            return ROTT, f"RTFx {varm:.1f}× < 5×: byt motor (R9/R10)"
        if varm >= 10 and har(d) and d <= 1.0:
            return GRONT, f"RTFx {varm:.1f}×, DER-Δ {d:+.2f}"
        return GRATT, f"RTFx {varm:.1f}×, DER-Δ {fmt(d)}"
    return OMATT, "okänd körning"


def kombination(resultat):
    """R8: slå ihop inställningarna från alla gröna körningar bland R1–R7.

    Bland R3 och R4 tas bara den snabbaste gröna varianten.
    """
    gron = {k: r for k, r in resultat.items() if r.get("band") == GRONT and k != "B"}
    inst = {}
    for familj in ("R1", "R2", "R7"):
        if familj in gron:
            inst.update(gron[familj]["inst"])
    for familj in ("R3", "R4"):
        kandidater = [r for k, r in gron.items() if k.startswith(familj)]
        if kandidater:
            bast = max(kandidater, key=lambda r: r.get("rtfx_varm") or 0)
            inst.update(bast["inst"])
    return inst


def tolkningsregel(resultat):
    """Utredningens tolkningsregel: B rött och R1–R7 alla gråa → felsök miljön."""
    b = resultat.get("B", {}).get("band")
    ovriga = [r.get("band") for k, r in resultat.items() if k not in ("B", "R8")]
    return bool(b == ROTT and ovriga and all(x == GRATT for x in ovriga))


def fmt(x, enhet=""):
    if x is None:
        return OMATT
    return f"{x:.2f}{enhet}" if isinstance(x, float) else f"{x}{enhet}"


# ---------------------------------------------------------------- motorer

class Attrapp:
    """Simulerad motor för prov: inga beroenden, inga väntetider.

    Tiderna är påhittade men följer utredningens referensmätning (RTFx ≈ 2,5 för
    baslinjen) så att alla band kan provas.
    """
    namn = "attrapp"
    FAKTOR = {"minne": 1.05, "tf32": 1.3, ("batch", 32): 1.25, ("batch", 64): 1.3,
              ("steg", 0.2): 1.8, ("steg", 0.4): 2.8, "talare": 1.1}
    DER_PP = {"tf32": 0.1, ("steg", 0.2): 0.3, ("steg", 0.4): 0.8}

    def __init__(self, langd=546.0):
        self.langd = langd
        self.anrop = 0

    def ljudlangd(self, kalla):
        return self.langd

    def uppvarm(self):
        return self.kor(None, {})

    def kor(self, kalla, inst, num_speakers=None):
        self.anrop += 1
        if inst.get("batch") == 128:
            return {"oom": True}
        f, d = 1.0, 0.0
        for k, v in inst.items():
            nyckel = k if k in ("minne", "tf32", "talare") else (k, v)
            f *= self.FAKTOR.get(nyckel, 1.0)
            d += self.DER_PP.get(nyckel, 0.0)
        vagg = self.langd / (2.5 * f)
        if self.anrop == 2:          # första riktiga anropet efter uppvärmningen = kallt
            vagg += 20.0
        return {"vagg": vagg, "hyp": {"der_pp": d, "talare": 4}, "oom": False,
                "handelser": [(0.0, "segmentation"), (vagg * 0.4, "embeddings"),
                              (vagg * 0.9, "discrete_diarization")],
                "vram_alloc_gb": 3.1, "vram_reserverat_gb": 4.0 + (inst.get("batch") or 0) / 32,
                "tf32_i_hook": bool(inst.get("tf32")), "enhet_i_hook": "cuda:0"}

    def der(self, ref, hyp, collar):
        if ref.get("referens"):                   # mot en riktig referens: baslinjen har 5 pe fel
            return max(0.0, 5.0 + hyp["der_pp"] - (1.0 if collar else 0.0)) / 100
        return abs(hyp["der_pp"] - ref["der_pp"]) / 100   # mot baslinjens utdata

    def talare(self, ann):
        return ann["talare"]

    def rttm(self, ann):
        turer = 100 - round(ann["der_pp"] * 10)
        return "".join(f"SPEAKER attrapp 1 {i:.3f} 1.000 <NA> <NA> T{i % 4} <NA> <NA>\n"
                       for i in range(turer))

    def ladda_ref(self, sokvag):
        return {"der_pp": 0.0, "talare": 4, "referens": True}


class Pyannote:
    """Riktig motor: pyannote.audio 4.0.x med community-1 på CUDA."""
    namn = "pyannote"

    def __init__(self, modell, token_variabel, enhet):
        import torch
        from pyannote.audio import Audio, Pipeline
        import pyannote.audio.core.inference as inferens
        import pyannote.audio.core.pipeline as karna
        self.torch, self.Audio = torch, Audio
        # fix_reproducibility() slår av TF32 vid varje anrop; den importeras på två ställen.
        self._moduler = (inferens, karna)
        self._original = (inferens.fix_reproducibility, karna.fix_reproducibility)
        token = os.environ.get(token_variabel) or None
        self.pipeline = Pipeline.from_pretrained(modell, token=token)
        if self.pipeline is None:
            raise RuntimeError(f"kunde inte ladda {modell}; godkänn villkoren på "
                               f"Hugging Face och sätt {token_variabel}")
        self.enhet = torch.device(enhet)
        self.pipeline.to(self.enhet)
        seg = self.pipeline._segmentation
        self._bas = {"seg_batch": self.pipeline.segmentation_batch_size,
                     "emb_batch": self.pipeline.embedding_batch_size,
                     "steg_s": seg.step, "fonster_s": seg.duration}
        self._minne = {}

    def ljudlangd(self, kalla):
        if isinstance(kalla, dict):
            return kalla["waveform"].shape[-1] / kalla["sample_rate"]
        vag, sr = self._ladda(kalla)
        return vag.shape[-1] / sr

    def _ladda(self, sokvag):
        if sokvag not in self._minne:
            self._minne[sokvag] = self.Audio(sample_rate=16000, mono="downmix")(sokvag)
        return self._minne[sokvag]

    def _tf32(self, pa):
        torch = self.torch
        inferens, karna = self._moduler
        if not pa:
            inferens.fix_reproducibility, karna.fix_reproducibility = self._original
            return
        orig = self._original[0]

        def tillat_tf32(enhet):
            orig(enhet)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        inferens.fix_reproducibility = karna.fix_reproducibility = tillat_tf32

    def _stall_in(self, inst):
        p, b = self.pipeline, self._bas
        p.segmentation_batch_size = inst.get("batch", b["seg_batch"])
        p.embedding_batch_size = inst.get("batch", b["emb_batch"])
        steg = inst.get("steg")
        p._segmentation.step = steg * b["fonster_s"] if steg else b["steg_s"]
        self._tf32(inst.get("tf32", False))

    def uppvarm(self):
        """Utredningens K0: 30 s svagt brus, resultatet kastas (4.2 p.1)."""
        brus = {"waveform": self.torch.randn(1, 16000 * 30) * 0.01, "sample_rate": 16000}
        return self.kor(brus, {})

    def kor(self, kalla, inst, num_speakers=None):
        torch = self.torch
        self._stall_in(inst)
        if inst.get("minne") and not isinstance(kalla, dict):
            vag, sr = self._ladda(kalla)
            kalla = {"waveform": vag, "sample_rate": sr}
        kw = {"num_speakers": num_speakers} if inst.get("talare") and num_speakers else {}
        handelser, syn = [], {}

        def krok(steg, *a, **k):
            if not syn:
                syn["tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
                syn["enhet"] = str(next(self.pipeline._segmentation.model.parameters()).device)
            handelser.append((time.perf_counter(), steg))
        cuda = self.enhet.type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        try:
            t0 = time.perf_counter()
            ut = self.pipeline(kalla, hook=krok, **kw)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"oom": True}
        finally:
            self._stall_in({})
        ann = getattr(ut, "speaker_diarization", ut)
        return {"vagg": t1 - t0, "hyp": ann, "oom": False,
                "handelser": [(t - t0, s) for t, s in handelser],
                "vram_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
                "vram_reserverat_gb": torch.cuda.max_memory_reserved() / 1e9 if cuda else None,
                "tf32_i_hook": syn.get("tf32"), "enhet_i_hook": syn.get("enhet")}

    def der(self, ref, hyp, collar):
        from pyannote.metrics.diarization import DiarizationErrorRate
        return float(DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyp))

    def talare(self, ann):
        return len(ann.labels())

    def rttm(self, ann):
        return ann.to_rttm()

    def ladda_ref(self, sokvag):
        from pyannote.database.util import load_rttm
        alla = load_rttm(sokvag)
        if len(alla) == 1:
            return next(iter(alla.values()))
        stam = os.path.splitext(os.path.basename(sokvag))[0]
        if stam in alla:
            return alla[stam]
        raise ValueError(f"{sokvag} har {len(alla)} uri:er; döp filen efter den som gäller")


# ---------------------------------------------------------------- bänken

class Cpu:
    """Medel-CPU-last under en körning, om psutil finns."""
    def __init__(self):
        try:
            import psutil
            self.ps = psutil
        except ImportError:
            self.ps = None
        self.prov, self._stopp = [], threading.Event()

    def __enter__(self):
        if self.ps:
            self.ps.cpu_percent(None)
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stopp.wait(0.5):
            self.prov.append(self.ps.cpu_percent(None))

    def __exit__(self, *a):
        self._stopp.set()
        self.medel = round(statistics.mean(self.prov), 1) if self.prov else None


def mat_variant(motor, kalla, inst, args, ljud_s, ref, bas_hyp, kall=False):
    """Kör en variant och ger (mätvärden, sista utdata).

    Baslinjen (kall=True) mäts direkt efter den globala uppvärmningen: första
    anropet är kallt, resten varma. Övriga varianter får en kastad körning först,
    så att t.ex. cudnn hinner välja kärnor för den nya batchstorleken.
    """
    def kor():
        return motor.kor(kalla, inst, args.talare)
    if not kall and kor().get("oom"):
        return {"inst": inst, "oom": True}, None
    tider, sista, cpu = [], None, []
    for _ in range(args.upprepa + (1 if kall else 0)):
        with Cpu() as c:
            r = kor()
        if r.get("oom"):
            return {"inst": inst, "oom": True}, None
        cpu.append(c.medel)
        tider.append(r["vagg"])
        sista = r
    mat = {"inst": inst, "oom": False}
    if kall:
        mat["vagg_kall"] = round(tider[0], 3)
        mat["rtfx_kall"] = rtfx(ljud_s, tider[0])
        tider = tider[1:]
    varm = statistics.median(tider)
    cpu = [x for x in cpu if x is not None]
    mat.update({
        "vagg_varm": round(varm, 3), "rtfx_varm": rtfx(ljud_s, varm),
        "stegtider": stegtider(sista["handelser"], sista["vagg"]),
        "vram_alloc_gb": sista.get("vram_alloc_gb"),
        "vram_reserverat_gb": sista.get("vram_reserverat_gb"),
        "cpu_medel": round(statistics.mean(cpu), 1) if cpu else None,
        "tf32_i_hook": sista.get("tf32_i_hook"), "enhet_i_hook": sista.get("enhet_i_hook"),
        "talare": motor.talare(sista["hyp"]),
    })
    if kall:
        mat["kall_minus_varm"] = round(mat["vagg_kall"] - varm, 3)
    if ref is not None:
        mat["der_ref_c0"] = motor.der(ref, sista["hyp"], 0.0)
        mat["der_ref_c25"] = motor.der(ref, sista["hyp"], 0.5)   # pyannote: total bredd = ±0,25 s
        mat["talare_ref"] = motor.talare(ref)
    elif bas_hyp is not None:
        mat["der_sjalv_c0"] = motor.der(bas_hyp, sista["hyp"], 0.0)
    if bas_hyp is not None:
        # En variabel i taget: felet räknas bara om ändringen själv flyttade talarantalet.
        mat["talarfel"] = mat["talare"] != motor.talare(bas_hyp)
    return mat, sista["hyp"]


def delta(mat, bas):
    """DER-Δ i procentenheter: mot referensen om den finns, annars mot baslinjens utdata."""
    if mat.get("der_ref_c0") is not None and bas.get("der_ref_c0") is not None:
        return round((mat["der_ref_c0"] - bas["der_ref_c0"]) * 100, 3)
    if mat.get("der_sjalv_c0") is not None:
        return round(mat["der_sjalv_c0"] * 100, 3)
    return None


def spara_rttm(motor, args, kod, hyp):
    """Skriver en variants diarisering till <katalog>/<körning>.rttm, om det begärts."""
    if not args.rttm_katalog or hyp is None:
        return
    os.makedirs(args.rttm_katalog, exist_ok=True)
    with open(os.path.join(args.rttm_katalog, f"{kod}.rttm"), "w", encoding="utf-8") as f:
        f.write(motor.rttm(hyp))


def tabell(resultat):
    rader = ["| Körning | Ändring | RTFx varm | Mot B | DER-Δ (pe) | VRAM res. | Band | Skäl |",
             "|---|---|---|---|---|---|---|---|"]
    bas = resultat.get("B", {})
    for kod, r in resultat.items():
        s = andel(r.get("rtfx_varm"), bas.get("rtfx_varm")) if kod != "B" else None
        rader.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            kod, r.get("beskrivning", ""),
            OMATT if r.get("rtfx_varm") is None else f"{r['rtfx_varm']:.2f}×",
            "–" if s is None else f"{s:+.0%}",
            fmt(r.get("der_delta_pp")), fmt(r.get("vram_reserverat_gb"), " GB"),
            r.get("band"), r.get("skal", "")))
    for kod, skal in EJ_I_BANKEN.items():
        rader.append(f"| {kod} | – | – | – | – | – | {OMATT} | ingår inte: {skal} |")
    return "\n".join(rader)


def main(argv):
    ap = argparse.ArgumentParser(prog="diarbank", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ljud", help="testklipp (wav, 16 kHz mono rekommenderas)")
    ap.add_argument("--ref", help="referens i RTTM; utan den mäts DER mot baslinjens utdata")
    ap.add_argument("--talare", type=int, help="känt antal talare; slår på R7")
    ap.add_argument("--upprepa", type=int, default=3, help="varma mätningar per variant (median)")
    ap.add_argument("--modell", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--token-variabel", default="HF_TOKEN")
    ap.add_argument("--enhet", default="cuda")
    ap.add_argument("--vram-tak-gb", type=float, default=9.0)
    ap.add_argument("--korningar", help="kommaseparerat urval, t.ex. B,R2,R4b")
    ap.add_argument("--ingen-kombination", action="store_true", help="hoppa över R8")
    ap.add_argument("--ut", help="spara resultatet som JSON")
    ap.add_argument("--rttm-katalog", help="spara varje variants diarisering som <körning>.rttm "
                    "(för diarjamfor.py; bara för testklipp, aldrig för riktigt ljud)")
    ap.add_argument("--lista", action="store_true", help="visa planen och avsluta")
    ap.add_argument("--attrapp", action="store_true", help="simulerad motor (prov)")
    args = ap.parse_args(argv[1:])

    valda = [v for v in VARIANTER if v[0] != "R7" or args.talare]
    if args.korningar:
        urval = {k.strip() for k in args.korningar.split(",")} | {"B"}
        valda = [v for v in valda if v[0] in urval]
    if args.lista:
        for kod, besk, inst in valda:
            print(f"{kod:7} {besk}")
        for kod, skal in EJ_I_BANKEN.items():
            print(f"{kod:7} ingår inte: {skal}")
        return 0
    if args.upprepa < 1:
        print("--upprepa måste vara minst 1", file=sys.stderr)
        return 1
    if not args.attrapp and not args.ljud:
        print("ange --ljud (eller --attrapp för en torrkörning)", file=sys.stderr)
        return 1

    try:
        motor = Attrapp() if args.attrapp else Pyannote(args.modell, args.token_variabel, args.enhet)
        kalla = args.ljud or "attrapp.wav"
        ljud_s = motor.ljudlangd(kalla)
        ref = motor.ladda_ref(args.ref) if args.ref else None
        forsta = motor.uppvarm()
    except Exception as e:                       # noqa: BLE001 — skrivs ut, aldrig tyst
        print(f"kunde inte starta: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # Rimlighetskontroll (4.2 p.4): ligger segmenteringsmodellen på GPU:n?
    enhet = forsta.get("enhet_i_hook") or ""
    if args.enhet.startswith("cuda") and not enhet.startswith("cuda"):
        print(f"rimlighetskontrollen föll: segmenteringsmodellen ligger på '{enhet}', "
              "inte på GPU:n. Felsök torch-bygget och drivrutinen först.", file=sys.stderr)
        return 2

    cfg = {"vram_tak_gb": args.vram_tak_gb}
    resultat, bas_hyp = {}, None
    for kod, besk, inst in valda:
        print(f"… {kod} ({besk})", file=sys.stderr, flush=True)
        mat, hyp = mat_variant(motor, kalla, inst, args, ljud_s, ref, bas_hyp, kall=(kod == "B"))
        if kod == "B":
            bas_hyp = hyp
        spara_rttm(motor, args, kod, hyp)
        resultat[kod] = {"beskrivning": besk, **mat}

    def satt_band():
        bas = resultat.get("B", {})
        for kod, r in resultat.items():
            r["der_delta_pp"] = 0.0 if kod == "B" else delta(r, bas)
            r["band"], r["skal"] = band(kod, r, bas, cfg)

    satt_band()
    inst = {} if args.ingen_kombination else kombination(resultat)
    if inst:
        print("… R8 (kombination)", file=sys.stderr, flush=True)
        mat, hyp = mat_variant(motor, kalla, inst, args, ljud_s, ref, bas_hyp)
        spara_rttm(motor, args, "R8", hyp)
        resultat["R8"] = {"beskrivning": "kombination: " + ", ".join(
            f"{k}={v}" for k, v in sorted(inst.items())), **mat}
        satt_band()

    print(tabell(resultat))
    if tolkningsregel(resultat):
        print("\nB är rött och R1–R7 är alla gråa: GPU:n är inte flaskhalsen. Felsök miljön "
              "(torch-bygge, .to(cuda), drivrutin) innan fler parametrar provas.")
    if args.ut:
        with open(args.ut, "w", encoding="utf-8") as f:
            json.dump({"motor": motor.namn, "ljud_s": ljud_s, "upprepa": args.upprepa,
                       "referens": bool(ref), "resultat": resultat,
                       "ej_i_banken": EJ_I_BANKEN}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
