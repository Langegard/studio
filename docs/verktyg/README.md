# Verktyg

Små, generiska verktyg som följer med utredningarna. Inget av dem bär husdata: inga
värdnamn, adresser eller sökvägar. Det kontrolleras i CI.

| Verktyg | Vad det gör | Hör till |
|---|---|---|
| [`usbdjup.py`](usbdjup.py) | Läser en UsbTreeView-rapport (XML), visar varje hubbkedja med sitt djup, flaggar allt som sitter djupare än fem hubbar och föreslår den minsta flytt som löser det. | [USB-topologi](../utredningar/usb-topologi.md) |
| [`usbserver-installera.sh`](usbserver-installera.sh) | Gör en Linux-nod till USB/IP-server och lägger till kommandot `usbdela`. Vägrar alltid lagringsenheter och hubbar. | [USB-topologi](../utredningar/usb-topologi.md), avsnitt 5 |

## usbdjup

```
# I UsbTreeView: File → Save Report (XML)
python3 usbdjup.py rapport.xml
```

Avslutningskod: `0` om inget sitter för djupt, `2` om minst en hubb gör det, `1` vid fel.
Utskriften innehåller bara portkedjor, krets-ID (VID:PID) och status, aldrig enhetsnamn.
Den kan därför klistras in i en chatt eller en PR.

Prov: `python3 -m unittest discover -s docs/verktyg/prov`. Proven körs mot syntetiska
rapporter, en med sex hubbar i rad och ostängda XML-noder (som UsbTreeViews egna ibland
har) och en platt. De kontrollerar också att ett påhittat enhetsnamn inte läcker ut i
utskriften.

## usbserver-installera.sh

```
sudo ./usbserver-installera.sh            # bara Tailscale når port 3240
sudo ./usbserver-installera.sh --lan 192.168.1.0/24
```

Därefter `sudo usbdela lista`, `sudo usbdela dela vvvv:pppp` och `sudo usbdela sluta vvvv:pppp`.
Syntaxkontrollerat och shellcheck-rent. Ännu inte kört på riktig maskin, så börja med en
enhet som inte gör något om den krånglar.
