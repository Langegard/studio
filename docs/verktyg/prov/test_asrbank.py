"""Prov för asrbank.py: normalisering, WER, band och en torrkörning.

Kör: python3 -m unittest discover -s docs/verktyg/prov
Proven behöver varken faster-whisper eller ljud; motorn är en attrapp.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HAR = os.path.dirname(os.path.abspath(__file__))
VERKTYG = os.path.join(HAR, "..", "asrbank.py")
spec = importlib.util.spec_from_file_location("asrbank", VERKTYG)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def kor(*arg):
    r = subprocess.run([sys.executable, VERKTYG, *arg], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


class Text(unittest.TestCase):
    def test_normalisering_behaller_svenska_bokstaver(self):
        self.assertEqual(ab.normalisera("Hej, Åsa!  Är   ÖL gott?"), "hej åsa är öl gott")

    def test_normalisering_tar_bort_tankstreck_och_citattecken(self):
        self.assertEqual(ab.normalisera("»Ja« – sa hon."), "ja sa hon")

    def test_ordfel(self):
        self.assertEqual(ab.ordfel("a b c", "a b c"), (0, 3))
        self.assertEqual(ab.ordfel("a b c", "a x c"), (1, 3))       # ersättning
        self.assertEqual(ab.ordfel("a b c", "a c"), (1, 3))         # strykning
        self.assertEqual(ab.ordfel("a b c", "a b b c"), (1, 3))     # tillägg
        self.assertEqual(ab.ordfel("", "a"), (1, 0))

    def test_korpus_wer_viktas_per_ord(self):
        par = [("en två tre fyra", "en två tre fyra"), ("fem sex", "fem sju")]
        self.assertAlmostEqual(ab.wer(par), 100 / 6, places=2)


class Varianter(unittest.TestCase):
    def test_storlek_blir_kb_whisper(self):
        self.assertEqual(ab.tolka_variant("medium:int8:cpu"), ("KBLab/kb-whisper-medium", "int8", "cpu"))

    def test_fullt_modell_id_behalls(self):
        self.assertEqual(ab.tolka_variant("org/modell:float16:cuda")[0], "org/modell")

    def test_trasiga_varianter(self):
        for fel in ("large:int8", "large::cuda", "large:int8:tpu"):
            with self.assertRaises(ValueError):
                ab.tolka_variant(fel)


class Band(unittest.TestCase):
    def r(self, modell, prec, wer=None, rtf=None):
        return {"modell": modell, "precision": prec, "wer": wer, "rtf": rtf}

    def test_experiment_5(self):
        f = self.r("KBLab/kb-whisper-large", "float16", wer=5.4)
        for int8_wer, vantat in ((5.7, ab.GRONT), (5.71, ab.GRATT), (6.4, ab.GRATT), (6.41, ab.ROTT)):
            i = self.r("KBLab/kb-whisper-large", "int8", wer=int8_wer)
            self.assertEqual(ab.band(5, [f, i])[0], vantat, int8_wer)

    def test_experiment_5_utan_int8_ar_omatt(self):
        self.assertEqual(ab.band(5, [self.r("x", "float16", wer=5.0)])[0], ab.OMATT)

    def test_experiment_6(self):
        s = self.r("KBLab/kb-whisper-small", "int8", wer=7.3, rtf=0.2)
        for rtf, vantat in ((0.5, ab.GRONT), (0.51, ab.GRATT), (1.0, ab.GRATT), (1.01, ab.ROTT)):
            m = self.r("KBLab/kb-whisper-medium", "int8", wer=6.6, rtf=rtf)
            self.assertEqual(ab.band(6, [s, m])[0], vantat, rtf)


class Manifest(unittest.TestCase):
    def test_relativa_sokvagar_och_kommentarer(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.tsv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("# kommentar\nljud/a.wav\thej hopp\n\n")
            self.assertEqual(ab.las_manifest(p), [(os.path.join(d, "ljud/a.wav"), "hej hopp")])

    def test_rad_utan_referens_ar_fel(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.tsv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("a.wav\n")
            with self.assertRaises(ValueError):
                ab.las_manifest(p)


class Torrkorning(unittest.TestCase):
    def test_experiment_5(self):
        with tempfile.TemporaryDirectory() as d:
            ut = os.path.join(d, "r.json")
            kod, text, _ = kor("--attrapp", "--experiment", "5", "--ut", ut)
            self.assertEqual(kod, 0)
            with open(ut, encoding="utf-8") as f:
                j = json.load(f)
        self.assertEqual(j["band"], ab.GRONT)
        self.assertEqual([r["precision"] for r in j["resultat"]], ["float16", "int8"])
        self.assertNotIn("text", json.dumps(j))            # inga transkript sparas
        self.assertIn("**GRÖNT**", text)

    def test_experiment_6(self):
        kod, text, _ = kor("--attrapp", "--experiment", "6")
        self.assertEqual(kod, 0)
        self.assertIn("**GRÅTT**: medium RTF 0.70", text)

    def test_utan_manifest_ar_fel(self):
        self.assertEqual(kor()[0], 1)

    def test_trasig_variant_ar_fel(self):
        kod, _, fel = kor("--attrapp", "--varianter", "large:int8")
        self.assertEqual(kod, 1)
        self.assertIn("modell:compute_type:enhet", fel)


if __name__ == "__main__":
    unittest.main()
