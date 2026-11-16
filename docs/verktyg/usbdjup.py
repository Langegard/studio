#!/usr/bin/env python3
"""usbdjup — läser en UsbTreeView-rapport (XML) och räknar hubbdjup.

USB tillåter högst fem hubbar i rad mellan rotporten och en enhet. Sitter en
hubb på led sex vägrar Windows den ("Device nested too deeply") och allt bakom
den blir osynligt. Det här verktyget visar varje hubbkedja med sitt djup,
pekar ut det som är för djupt och föreslår den minsta flytt som löser det.

Användning:
    python3 usbdjup.py rapport.xml

Rapporten sparas i UsbTreeView med File → Save Report (XML). UsbTreeViews XML
är inte alltid välformad (sista noderna stängs inte), så den läses med en
tolerant tokeniserare i stället för en XML-parser.

Utskriften innehåller bara hubbar, portkedjor, krets-ID och status — aldrig
enhetsnamn, serienummer eller Bluetooth-namn — så den kan klistras in var som
helst utan att bära med sig husdata.

Avslutningskod: 0 = inget för djupt, 2 = minst en hubb för djupt, 1 = fel.
"""
import html
import re
import sys

MAX_HUBBAR = 5          # USB 2.0 kap. 4.1.1 / USB 3.x route string: fem hubbar
HUBBTYPER = {"6", "7"}  # UsbTreeView: 6 = hubb, 7 = hubb med fel
FART = {"0": "LS", "1": "FS", "2": "HS (USB 2)", "3": "SS (USB 3)"}

_tok = re.compile(r'<node\s+([^>]*)>|</node>|<text>(.*?)</text>', re.S)
_attr = re.compile(r'(\w+)="([^"]*)"')


def falt(text, namn):
    m = re.search(r'^' + re.escape(namn) + r'\s*:\s*(.*)$', text, re.M)
    return m.group(1).strip() if m else ""


def las(sokvag):
    with open(sokvag, encoding="utf-8-sig", errors="replace") as f:
        s = f.read()
    noder, vantande = [], None
    for m in _tok.finditer(s):
        if m.group(0).startswith("<node"):
            vantande = dict(_attr.findall(m.group(1)))
        elif m.group(2) is not None and vantande is not None:
            t = html.unescape(m.group(2)).replace("\r", "")
            kedja = falt(t, "Port Chain")
            if kedja:
                noder.append({
                    "typ": vantande.get("type", ""),
                    "fart": vantande.get("speed", ""),
                    "kedja": kedja,
                    "status": falt(t, "Connection Status"),
                    "vid": falt(t, "idVendor").split(" ")[0],
                    "pid": falt(t, "idProduct").split(" ")[0],
                    "tillverkare": (re.search(r'\((.*)\)', falt(t, "idVendor")) or [None, ""])[1],
                })
            vantande = None
    return noder


def led(kedja):
    """Antal portnummer efter bussnumret = vilket led hubben/enheten sitter på."""
    return len(kedja.split("-")) - 1


def anslutna(noder):
    return [n for n in noder if not n["status"].startswith("0x00")]


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 1
    try:
        noder = anslutna(las(argv[1]))
    except OSError as e:
        print(f"Kan inte läsa rapporten: {e}", file=sys.stderr)
        return 1
    hubbar = sorted((n for n in noder if n["typ"] in HUBBTYPER),
                    key=lambda n: [int(x) for x in n["kedja"].split("-")])
    if not hubbar:
        print("Inga hubbar hittades. Är det en UsbTreeView-rapport i XML?")
        return 1

    print(f"{'Led':>3}  {'Portkedja':<28} {'Krets':<10} {'Fart':<11} Status")
    for h in hubbar:
        d = led(h["kedja"])
        flagga = "  ❌" if d > MAX_HUBBAR or "too deeply" in h["status"].lower() else ""
        krets = f"{h['vid'][2:]}:{h['pid'][2:]}" if h["vid"] else "?"
        print(f"{d:>3}  {'  ' * (d - 1) + h['kedja']:<28} {krets:<10} "
              f"{FART.get(h['fart'], '?'):<11} {h['status'][:40]}{flagga}")

    # USB 3-halvor utan något nedströms: trafiken går då helt på USB 2-vägen.
    for h in hubbar:
        if h["fart"] == "3" and not any(
                n["kedja"].startswith(h["kedja"] + "-") for n in noder):
            print(f"\nObs: USB 3-hubben på {h['kedja']} bär ingenting nedströms. "
                  "Allt bakom den går på USB 2-halvan i ett annat träd.")

    djupa = [h for h in hubbar
             if led(h["kedja"]) > MAX_HUBBAR or "too deeply" in h["status"].lower()]
    if not djupa:
        print(f"\nInget sitter djupare än {MAX_HUBBAR} hubbar. Felet ligger någon annanstans.")
        return 0

    per_kedja = {h["kedja"]: h for h in hubbar}
    for h in djupa:
        d = led(h["kedja"])
        behov = d - MAX_HUBBAR
        delar = h["kedja"].split("-")
        forfader = ["-".join(delar[:i + 1]) for i in range(1, len(delar) - 1)]
        print(f"\n❌ {h['kedja']} sitter på led {d}, max är {MAX_HUBBAR}. "
              f"Kedjan behöver bli {behov} led kortare.")
        print("   Flytta en av hubbarna ovanför direkt till en rotport på datorn:")
        kandidater = [f for f in reversed(forfader) if led(f) - 1 >= behov and f in per_kedja]
        for f in kandidater:
            p = per_kedja[f]
            minst = " ← minsta flytt" if f == kandidater[0] else ""
            tillv = ", " + p["tillverkare"] if p["tillverkare"] else ""
            print(f"   - hubben på {f} ({p['vid'][2:]}:{p['pid'][2:]}{tillv}): "
                  f"sparar {led(f) - 1} led{minst}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
