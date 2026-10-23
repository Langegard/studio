"""Prov för cpwer.py: talarparning, talarkostnad, band och kommandorad.

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
VERKTYG = os.path.join(HAR, "..", "cpwer.py")
spec = importlib.util.spec_from_file_location("cpwer", VERKTYG)
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def kor(*arg):
    r = subprocess.run([sys.executable, VERKTYG, *arg], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def skriv(katalog, namn, repliker):
    sokvag = os.path.join(katalog, namn)
    with open(sokvag, "w", encoding="utf-8") as f:
        f.write("".join(f"{t}\t{x}\n" for t, x in repliker))
    return sokvag


def brute(ref, hyp):
    """Minsta fel över alla delvisa en-till-en-parningar (för små fall)."""
    R, H = cp.per_talare(ref), cp.per_talare(hyp)
    rn, hn = sorted(R), sorted(H)
    bast = None
    for k in range(min(len(rn), len(hn)) + 1):
        for rs in itertools.combinations(rn, k):
            for hs in itertools.permutations(hn, k):
                fel = sum(cp.ordfel(R[r], H[h])[0] for r, h in zip(rs, hs))
                fel += sum(len(R[r].split()) for r in rn if r not in rs)
                fel += sum(len(H[h].split()) for h in hn if h not in hs)
                bast = fel if bast is None else min(bast, fel)
    return bast


class Matt(unittest.TestCase):
    def test_identiska_och_omdopta(self):
        ref = [("A", "hej hur mår du"), ("B", "bra tack")]
        self.assertEqual(cp.cpwer(ref, ref)["fel"], 0)
        m = cp.cpwer(ref, [("X", "hej hur mår du"), ("Y", "bra tack")])
        self.assertEqual((m["fel"], m["parning"]), (0, {"A": "X", "B": "Y"}))

    def test_ratt_ord_hos_fel_talare_kostar(self):
        ref = [("A", "a b c"), ("B", "d e f")]
        m = cp.cpwer(ref, [("X", "a b c d e f")])
        self.assertEqual((m["fel"], m["ord"], m["fel_utan_talare"]), (6, 6, 0))

    def test_mot_uttommande_sokning(self):
        slump = random.Random(11)
        ord_ = "ja nej kanske hej tack bra dag kväll".split()
        for _ in range(150):
            def replik():
                return (slump.choice("ABC"), " ".join(slump.choice(ord_) for _ in range(slump.randint(1, 4))))
            ref = [replik() for _ in range(slump.randint(1, 5))]
            hyp = [(t.lower(), x) for t, x in (replik() for _ in range(slump.randint(1, 5)))]
            self.assertEqual(cp.cpwer(ref, hyp)["fel"], brute(ref, hyp))


class Band(unittest.TestCase):
    def test_experiment_8(self):
        self.assertEqual(cp.band8(10.0, 10.0)[0], cp.GRONT)
        self.assertEqual(cp.band8(11.0, 10.0)[0], cp.GRATT)
        self.assertEqual(cp.band8(11.01, 10.0)[0], cp.ROTT)
        self.assertEqual(cp.band8(None, 10.0)[0], cp.ROTT)


class Kommandorad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_kataloger_och_band(self):
        for k in ("facit", "a", "b"):
            os.mkdir(os.path.join(self.d, k))
        facit = [("A", "det här är ett påhittat möte"), ("B", "ja det stämmer")]
        skriv(os.path.join(self.d, "facit"), "m1.tsv", facit)
        skriv(os.path.join(self.d, "a"), "m1.tsv", [("1", "det här är ett påhittat möte"), ("2", "ja det stämmer")])
        skriv(os.path.join(self.d, "b"), "m1.tsv", [("1", "det här är ett möte"), ("2", "ja det")])
        ut = os.path.join(self.d, "r.json")
        kod, text, _ = kor("--ref", os.path.join(self.d, "facit"), "--hyp", os.path.join(self.d, "a"),
                           "--jamfor", os.path.join(self.d, "b"), "--ut", ut)
        self.assertEqual(kod, 0)
        with open(ut, encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["totalt"]["cpwer_a"], 0.0)
        self.assertAlmostEqual(j["totalt"]["cpwer_b"], 200 / 9, places=2)
        self.assertEqual(j["band"], cp.GRONT)
        self.assertIn("**GRÖNT**", text)

    def test_rott_ger_kod_2(self):
        facit = skriv(self.d, "f.tsv", [("A", "ett två tre fyra")])
        a = skriv(self.d, "a.tsv", [("1", "ett")])
        b = skriv(self.d, "b.tsv", [("1", "ett två tre fyra")])
        self.assertEqual(kor("--ref", facit, "--hyp", a, "--jamfor", b)[0], 2)

    def test_trasig_rad_och_fel_argument_ger_kod_1(self):
        trasig = os.path.join(self.d, "t.tsv")
        with open(trasig, "w", encoding="utf-8") as f:
            f.write("bara text utan talare\n")
        self.assertEqual(kor("--ref", trasig, "--hyp", trasig)[0], 1)
        self.assertEqual(kor("--ref", trasig, "--hyp", trasig, "--experiment", "3")[0], 1)


if __name__ == "__main__":
    unittest.main()
