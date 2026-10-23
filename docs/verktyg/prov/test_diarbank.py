"""Prov för diarbank.py: banden ur utredningens tabell 4.4 och en torrkörning.

Kör: python3 -m unittest discover -s docs/verktyg/prov
Proven behöver varken torch eller pyannote; motorn är en attrapp.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HAR = os.path.dirname(os.path.abspath(__file__))
VERKTYG = os.path.join(HAR, "..", "diarbank.py")
spec = importlib.util.spec_from_file_location("diarbank", VERKTYG)
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)

CFG = {"vram_tak_gb": 9.0}
BAS = {"rtfx_varm": 2.0}


def kor(*arg):
    r = subprocess.run([sys.executable, VERKTYG, *arg], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def torrkorning(*arg):
    with tempfile.TemporaryDirectory() as d:
        ut = os.path.join(d, "r.json")
        kod, text, _ = kor("--attrapp", "--upprepa", "1", "--ut", ut, *arg)
        with open(ut, encoding="utf-8") as f:
            return kod, text, json.load(f)


class Band(unittest.TestCase):
    def b(self, kod, **mat):
        return db.band(kod, mat, BAS, CFG)[0]

    def test_baslinjen(self):
        self.assertEqual(db.band("B", {"rtfx_varm": 2.9}, {}, CFG)[0], db.ROTT)
        self.assertEqual(db.band("B", {"rtfx_varm": 10.0}, {}, CFG)[0], db.GRONT)
        self.assertEqual(db.band("B", {"rtfx_varm": 5.0, "kall_minus_varm": 15.0}, {}, CFG)[0], db.GRONT)
        self.assertEqual(db.band("B", {"rtfx_varm": 5.0, "kall_minus_varm": 16.0}, {}, CFG)[0], db.GRATT)
        self.assertEqual(db.band("B", {}, {}, CFG)[0], db.OMATT)

    def test_r1_granser(self):
        self.assertEqual(self.b("R1", rtfx_varm=2.4), db.GRONT)    # +20 %
        self.assertEqual(self.b("R1", rtfx_varm=1.8), db.GRATT)    # −10 %
        self.assertEqual(self.b("R1", rtfx_varm=1.78), db.ROTT)    # −11 %

    def test_r2_kraver_der(self):
        self.assertEqual(self.b("R2", rtfx_varm=2.6), db.OMATT)
        self.assertEqual(self.b("R2", rtfx_varm=2.6, der_delta_pp=0.3), db.GRONT)
        self.assertEqual(self.b("R2", rtfx_varm=2.6, der_delta_pp=0.31), db.ROTT)
        self.assertEqual(self.b("R2", rtfx_varm=2.09, der_delta_pp=0.0), db.ROTT)
        self.assertEqual(self.b("R2", rtfx_varm=2.2, der_delta_pp=0.0), db.GRATT)

    def test_r3_minne_och_oom(self):
        self.assertEqual(self.b("R3-64", rtfx_varm=2.4, vram_reserverat_gb=8.9), db.GRONT)
        self.assertEqual(self.b("R3-64", rtfx_varm=2.4, vram_reserverat_gb=9.0), db.GRATT)
        self.assertEqual(self.b("R3-64", rtfx_varm=2.4), db.GRATT)          # VRAM omätt: aldrig grönt
        self.assertEqual(self.b("R3-128", oom=True), db.ROTT)

    def test_r4b_fel_antal_talare_ar_rott(self):
        self.assertEqual(self.b("R4b", rtfx_varm=6.0, der_delta_pp=0.5), db.GRONT)
        self.assertEqual(self.b("R4b", rtfx_varm=6.0, der_delta_pp=0.5, talarfel=True), db.ROTT)
        self.assertEqual(self.b("R4a", rtfx_varm=3.0, der_delta_pp=1.0), db.GRATT)
        self.assertEqual(self.b("R4a", rtfx_varm=3.2, der_delta_pp=0.5), db.GRONT)

    def test_r8(self):
        self.assertEqual(self.b("R8", rtfx_varm=4.9, der_delta_pp=0.0), db.ROTT)
        self.assertEqual(self.b("R8", rtfx_varm=10.0, der_delta_pp=1.0), db.GRONT)
        self.assertEqual(self.b("R8", rtfx_varm=10.0, der_delta_pp=1.1), db.GRATT)


class Hjalp(unittest.TestCase):
    def test_stegtider(self):
        h = [(0.0, "segmentation"), (1.0, "segmentation"), (4.0, "embeddings"),
             (7.0, "embeddings"), (9.0, "discrete_diarization")]
        self.assertEqual(db.stegtider(h, 10.0),
                         {"segmentation": 4.0, "embeddings": 5.0, "klustring": 2.0,
                          "discrete_diarization": 1.0})

    def test_rtfx(self):
        self.assertEqual(db.rtfx(546, 182), 3.0)
        self.assertIsNone(db.rtfx(546, 0))

    def test_tolkningsregel(self):
        r = {"B": {"band": db.ROTT}, "R1": {"band": db.GRATT}, "R2": {"band": db.GRATT}}
        self.assertTrue(db.tolkningsregel(r))
        r["R2"]["band"] = db.GRONT
        self.assertFalse(db.tolkningsregel(r))

    def test_kombination_tar_snabbaste_grona_per_familj(self):
        r = {"B": {"band": db.ROTT},
             "R2": {"band": db.GRONT, "inst": {"tf32": True}},
             "R3-32": {"band": db.GRONT, "inst": {"batch": 32}, "rtfx_varm": 3.0},
             "R3-64": {"band": db.GRONT, "inst": {"batch": 64}, "rtfx_varm": 3.2},
             "R4a": {"band": db.GRATT, "inst": {"steg": 0.2}}}
        self.assertEqual(db.kombination(r), {"tf32": True, "batch": 64})


class Torrkorning(unittest.TestCase):
    def test_lista(self):
        kod, text, _ = kor("--lista")
        self.assertEqual(kod, 0)
        self.assertIn("R5", text)
        self.assertNotIn("R7", text)            # R7 kräver --talare

    def test_kraver_ljud_utan_attrapp(self):
        self.assertEqual(kor()[0], 1)

    def test_hela_planen(self):
        kod, text, j = torrkorning()
        self.assertEqual(kod, 0)
        r = j["resultat"]
        self.assertEqual(r["B"]["band"], db.ROTT)                 # som referensmätningen, 2,5×
        self.assertEqual(r["R3-128"]["band"], db.ROTT)            # OOM
        self.assertEqual(r["R2"]["band"], db.GRONT)
        self.assertEqual(r["R8"]["band"], db.GRONT)
        self.assertEqual(r["R8"]["inst"], {"tf32": True, "batch": 64, "steg": 0.4})
        self.assertIn("| R5 |", text)
        self.assertNotIn("hyp", r["B"])                            # inga utdata sparas

    def test_talare_och_referens(self):
        kod, _, j = torrkorning("--talare", "4", "--ref", "attrapp.rttm")
        r = j["resultat"]
        self.assertTrue(j["referens"])
        self.assertIn("R7", r)
        self.assertAlmostEqual(r["R2"]["der_delta_pp"], 0.1, places=6)
        self.assertAlmostEqual(r["B"]["der_ref_c0"], 0.05, places=6)

    def test_urval_tar_alltid_med_baslinjen(self):
        kod, _, j = torrkorning("--korningar", "R2", "--ingen-kombination")
        self.assertEqual(sorted(j["resultat"]), ["B", "R2"])

    def test_rttm_katalog_kan_jamforas_med_diarjamfor(self):
        with tempfile.TemporaryDirectory() as d:
            kod, _, _ = kor("--attrapp", "--upprepa", "1", "--korningar", "R4b",
                            "--ingen-kombination", "--rttm-katalog", d)
            self.assertEqual(kod, 0)
            self.assertEqual(sorted(os.listdir(d)), ["B.rttm", "R4b.rttm"])
            jamfor = os.path.join(HAR, "..", "diarjamfor.py")
            r = subprocess.run([sys.executable, jamfor, "--ref", os.path.join(d, "B.rttm"),
                                "--hyp", os.path.join(d, "R4b.rttm"), "--experiment", "2"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)          # 8 % färre turer: rött i experiment 2
            self.assertIn("-8.0 %", r.stdout)

    def test_attrappen_laddar_inte_torch(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
