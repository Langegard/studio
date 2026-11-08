"""Prov för usbdjup.py mot syntetiska UsbTreeView-rapporter.

Kör: python3 -m unittest discover -s docs/verktyg/prov
"""
import os
import subprocess
import sys
import unittest

HAR = os.path.dirname(os.path.abspath(__file__))
VERKTYG = os.path.join(HAR, "..", "usbdjup.py")


def kor(fixtur):
    r = subprocess.run([sys.executable, VERKTYG, os.path.join(HAR, fixtur)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


class Djup(unittest.TestCase):
    def setUp(self):
        self.kod, self.ut = kor("djup.xml")

    def test_avslutningskod_2_vid_for_djupt(self):
        self.assertEqual(self.kod, 2)

    def test_pekar_ut_led_6(self):
        self.assertIn("1-2-3-4-5-6-7 sitter på led 6, max är 5", self.ut)

    def test_minsta_flytt_ar_narmaste_forfadern(self):
        rad = next(r for r in self.ut.splitlines() if "minsta flytt" in r)
        self.assertIn("1-2-3-4-5-6 ", rad)

    def test_usb3_halva_utan_last_flaggas(self):
        self.assertIn("USB 3-hubben på 1-20 bär ingenting", self.ut)

    def test_ostangda_noder_tal(self):
        # Rapporten slutar mitt i trädet, som UsbTreeViews egna gör ibland.
        self.assertIn("1-2-3-4-5-6-7", self.ut)

    def test_inga_enhetsnamn_lacker(self):
        # Bandregeln: utskriften får bära kedjor och krets-ID, aldrig namn.
        for ord_ in ("HEMLIG", "Privat", "Telefon", "PROV"):
            self.assertNotIn(ord_, self.ut)


class Platt(unittest.TestCase):
    def test_inget_fel_ger_0(self):
        kod, ut = kor("platt.xml")
        self.assertEqual(kod, 0)
        self.assertIn("Inget sitter djupare än 5 hubbar", ut)


class Felhantering(unittest.TestCase):
    def test_saknad_fil_ger_1(self):
        kod, _ = kor("finns-inte.xml")
        self.assertEqual(kod, 1)


if __name__ == "__main__":
    unittest.main()
