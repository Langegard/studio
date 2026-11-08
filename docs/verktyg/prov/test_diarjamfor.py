"""Prov för diarjamfor.py mot små, handräknade RTTM-filer.

Kör: python3 -m unittest discover -s docs/verktyg/prov
"""
import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HAR = os.path.dirname(os.path.abspath(__file__))
VERKTYG = os.path.join(HAR, "..", "diarjamfor.py")
spec = importlib.util.spec_from_file_location("diarjamfor", VERKTYG)
dj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dj)


def rttm(segment, uri="mote"):
    return "".join(f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {t} <NA> <NA>\n"
                   for s, e, t in segment)


def skriv(katalog, namn, segment):
    sokvag = os.path.join(katalog, namn)
    with open(sokvag, "w", encoding="utf-8") as f:
        f.write(rttm(segment))
    return sokvag


def kor(*arg):
    r = subprocess.run([sys.executable, VERKTYG, *arg], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


class Matt(unittest.TestCase):
    def test_identiska(self):
        s = [(0, 5, "A"), (5, 9, "B"), (9, 12, "A")]
        m = dj.jamfor(s, s)
        self.assertEqual((m["der"], m["etikettskillnad"]), (0.0, 0.0))

    def test_omdopta_etiketter_ger_ingen_der_men_full_etikettskillnad(self):
        m = dj.jamfor([(0, 5, "A"), (5, 10, "B")], [(0, 5, "X"), (5, 10, "Y")])
        self.assertEqual(m["der"], 0.0)
        self.assertEqual(m["etikettskillnad"], 100.0)
        self.assertEqual(m["matchning"], {"A": "X", "B": "Y"})

    def test_missat(self):
        m = dj.jamfor([(0, 10, "A")], [(0, 9, "A")])
        self.assertAlmostEqual(m["der"], 10.0, places=2)
        self.assertAlmostEqual(m["missat"], 10.0, places=2)

    def test_falsklarm(self):
        m = dj.jamfor([(0, 10, "A")], [(0, 10, "A"), (10, 12, "B")])
        self.assertAlmostEqual(m["falsklarm"], 20.0, places=2)

    def test_forvaxling(self):
        m = dj.jamfor([(0, 5, "A"), (5, 10, "B")], [(0, 10, "X")])
        self.assertAlmostEqual(m["forvaxling"], 50.0, places=2)
        self.assertAlmostEqual(m["der"], 50.0, places=2)

    def test_overlapp_raknas(self):
        m = dj.jamfor([(0, 10, "A"), (5, 10, "B")], [(0, 10, "A")])
        self.assertAlmostEqual(m["missat"], 100 * 5 / 15, places=2)

    def test_sjalvoverlapp_raknas_en_gang(self):
        m = dj.jamfor([(0, 10, "A"), (5, 12, "A")], [(0, 12, "X")])
        self.assertEqual(m["der"], 0.0)
        self.assertAlmostEqual(m["tal_s"], 12.0, places=2)

    def test_turer_raknas_pa_rutnatet(self):
        m = dj.jamfor([(0, 5, "A"), (5, 10, "A"), (10, 12, "B")], [(0, 10, "A"), (10, 12, "B")])
        self.assertEqual((m["turer_ref"], m["rader_ref"]), (2, 3))
        self.assertEqual(dj.turavvikelse(m), 0.0)

    def test_turavvikelse(self):
        ref = [(i, i + 1, "AB"[i % 2]) for i in range(100)]
        hyp = ref[:93]
        m = dj.jamfor(ref, hyp)
        self.assertEqual(dj.turavvikelse(m), -7.0)


class Ungersk(unittest.TestCase):
    def test_mot_uttommande_sokning(self):
        slump = random.Random(7)
        for _ in range(200):
            nr, nk = slump.randint(1, 5), slump.randint(1, 5)
            m = [[slump.randint(0, 50) for _ in range(nk)] for _ in range(nr)]
            par = dj.ungersk(m)
            fatt = sum(m[i][j] for i, j in par)
            if nr <= nk:
                bast = max(sum(m[i][p[i]] for i in range(nr))
                           for p in itertools.permutations(range(nk), nr))
            else:
                bast = max(sum(m[p[j]][j] for j in range(nk))
                           for p in itertools.permutations(range(nr), nk))
            self.assertEqual(fatt, bast)
            self.assertEqual(len({i for i, _ in par}), len(par))
            self.assertEqual(len({j for _, j in par}), len(par))


class Band(unittest.TestCase):
    def test_experiment_2(self):
        self.assertEqual(dj.band(2, 0.5, -2.0)[0], dj.GRONT)
        self.assertEqual(dj.band(2, 0.51, 0.0)[0], dj.GRATT)
        self.assertEqual(dj.band(2, 0.2, -6.0)[0], dj.GRATT)
        self.assertEqual(dj.band(2, 0.2, -6.1)[0], dj.ROTT)
        self.assertEqual(dj.band(2, 1.6, 0.0)[0], dj.ROTT)

    def test_experiment_1(self):
        self.assertEqual(dj.band(1, 2.0, 1.0)[0], dj.GRONT)
        self.assertEqual(dj.band(1, 4.0, 2.0)[0], dj.GRATT)
        self.assertEqual(dj.band(1, 6.1, 0.0)[0], dj.ROTT)

    def test_utan_tal_ar_rott(self):
        self.assertEqual(dj.band(2, None, None)[0], dj.ROTT)


class Kommandorad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_kataloger_och_totalt(self):
        ref = [(i, i + 1, "AB"[i % 2]) for i in range(100)]
        for sida in ("ref", "hyp"):
            os.mkdir(os.path.join(self.d, sida))
        for n in ("a.rttm", "b.rttm"):
            skriv(os.path.join(self.d, "ref"), n, ref)
        skriv(os.path.join(self.d, "hyp"), "a.rttm", ref)
        skriv(os.path.join(self.d, "hyp"), "b.rttm", ref[:97])
        ut = os.path.join(self.d, "r.json")
        kod, text, _ = kor("--ref", os.path.join(self.d, "ref"), "--hyp", os.path.join(self.d, "hyp"),
                           "--experiment", "2", "--ut", ut)
        self.assertEqual(kod, 2)                              # en fil är RÖTT
        with open(ut, encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["totalt"]["turer_ref"], 200)
        self.assertEqual(j["totalt"]["turavvikelse"], -1.5)
        self.assertAlmostEqual(j["totalt"]["missat"], 1.5, places=2)
        self.assertEqual(j["totalt"]["band"], dj.GRATT)       # DER 1,5 % > 0,5 men ≤ 1,5
        self.assertEqual([m["band"] for m in j["filer"]], [dj.GRONT, dj.ROTT])
        self.assertEqual(j["totalt"]["storsta_turavvikelse"], 3.0)
        self.assertIn("| totalt |", text)

    def test_felaktiga_argument_ger_kod_1(self):
        self.assertEqual(kor("--ref", "a", "--hyp", "b", "--experiment", "4")[0], 1)

    def test_bom_tappar_inte_forsta_raden(self):
        ref = os.path.join(self.d, "r.rttm")
        with open(ref, "w", encoding="utf-8-sig") as f:
            f.write(rttm([(0, 2, "A"), (2, 4, "B")]))
        self.assertEqual(len(dj.segment_for(ref)), 2)

    def test_facit_mater_der_skillnaden(self):
        fac = skriv(self.d, "f.rttm", [(0, 10, "A"), (10, 20, "B")])
        ref = skriv(self.d, "r.rttm", [(0, 10, "A"), (10, 19, "B")])      # 5 % fel mot facit
        hyp = skriv(self.d, "h.rttm", [(0, 10, "A"), (10, 18.9, "B")])    # 5,5 % fel mot facit
        ut = os.path.join(self.d, "j.json")
        kod, text, _ = kor("--ref", ref, "--hyp", hyp, "--facit", fac, "--ut", ut)
        with open(ut, encoding="utf-8") as f:
            j = json.load(f)
        self.assertAlmostEqual(j["totalt"]["der_matt"], 0.5, places=2)
        self.assertEqual(j["totalt"]["band"], dj.GRONT)
        self.assertIn("DER-skillnad mot facit", text)

    def test_tyst_mot_tyst_ar_overensstammelse(self):
        tom = skriv(self.d, "t.rttm", [])
        self.assertEqual(kor("--ref", tom, "--hyp", tom)[0], 0)

    def test_rott_ger_kod_2(self):
        ref = skriv(self.d, "r.rttm", [(0, 10, "A")])
        hyp = skriv(self.d, "h.rttm", [(0, 5, "A")])
        self.assertEqual(kor("--ref", ref, "--hyp", hyp)[0], 2)

    def test_olika_filer_i_katalogerna_ar_fel(self):
        for sida in ("ref", "hyp"):
            os.mkdir(os.path.join(self.d, sida))
        skriv(os.path.join(self.d, "ref"), "a.rttm", [(0, 1, "A")])
        skriv(os.path.join(self.d, "hyp"), "b.rttm", [(0, 1, "A")])
        kod, _, fel = kor("--ref", os.path.join(self.d, "ref"), "--hyp", os.path.join(self.d, "hyp"))
        self.assertEqual(kod, 1)
        self.assertIn("stämmer inte", fel)

    def test_andra_radtyper_ignoreras_och_trasiga_rader_ar_fel(self):
        ref = os.path.join(self.d, "r.rttm")
        with open(ref, "w", encoding="utf-8") as f:
            f.write("SPKR-INFO mote 1 <NA> <NA> <NA> unknown A <NA> <NA>\n" + rttm([(0, 2, "A")]))
        self.assertEqual(kor("--ref", ref, "--hyp", ref)[0], 0)
        trasig = os.path.join(self.d, "t.rttm")
        with open(trasig, "w", encoding="utf-8") as f:
            f.write("SPEAKER mote 1 0.0\n")
        self.assertEqual(kor("--ref", trasig, "--hyp", ref)[0], 1)


if __name__ == "__main__":
    unittest.main()
