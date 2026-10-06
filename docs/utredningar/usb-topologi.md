# USB-topologi: djupgränsen, de två träden och delad kringutrustning

*Teknisk utredning efter ett verkligt fel ("Too many USB hubs") på en arbetsstation. Källor lästa 2026-10-06. Markering: **[D]** = dokumenterat av andra (med källa), **[I]** = egen slutsats. Fallet är avidentifierat: inga värdnamn, enhetsnamn eller sökvägar.*

## Sammanfattning

1. **Gränsen är fem hubbar.** USB tillåter sju led från värd till enhet. Det motsvarar högst fem hubbar i rad mellan rotporten och en enhet [1][2] **[D]**. I USB 3 är gränsen inbyggd i adressen: en 20-bitars *route string* med fem fält på fyra bitar, ett per hubbled [3] **[D]**.
2. **Felet handlar om djup, inte om ström.** Windows rapporterar statusen `DeviceHubNestedTooDeeply` (0x07). Strömfel har egna statusar, `DeviceNotEnoughPower` och `DeviceCausedOvercurrent` [4] **[D]**. Hubbar med egen ström ändrar därför ingenting här **[I]**.
3. **Hubbar gömmer sig.** En USB 3-hubb är två hubbar i samma låda, en för USB 2 och en för SuperSpeed [5] **[D]**. Stora hubbar består ofta av flera kretsar i serie. Dockor, USB-switchar och skärmar har egna hubbkretsar **[I]**. Antalet lådor på bordet säger alltså lite om djupet.
4. **Det finns två träd.** SuperSpeed kräver en obruten USB 3-väg hela vägen från värd till enhet [3] **[D]**. Ett enda USB 2-led i kedjan, till exempel en USB 2-switch, tvingar allt nedströms in i USB 2-trädet. Där räknas djupet för sig **[I]**.
5. **I fallet** satt den sjätte hubben bakom två USB-switchar och en stor tvåkretshubb. Den billigaste lösningen var att flytta *en* låda till en rotport. Det kostade ingenting, och delningen mellan datorerna fanns kvar (avsnitt 4) **[I]**.
6. **För att dela kringutrustning mellan flera datorer** finns en trappa (avsnitt 5): platta ut kedjan, byt två switchar mot en, låt nätverket sköta mus och tangentbord och, sist, USB över IP **[I]**.

Verktyg: [`docs/verktyg/usbdjup.py`](../verktyg/usbdjup.py) läser en UsbTreeView-rapport och pekar ut den minsta flytt som löser felet.

---

## 1. Gränsen

USB är ett träd med värden som rot. Specifikationen tillåter högst sju led i varje gren: värden, upp till fem hubbar och sist enheten [1][2] **[D]**. En sjätte hubb får ingen plats i adressen. Det gäller bokstavligt för USB 3, där paket styrs med fem portnummer på fyra bitar vardera, och varje hubb vet sitt djup (0–4) och läser "sitt" fält [3] **[D]**.

Liknelse: adressen är ett kuvert med plats för fem "c/o"-rader. Den sjätte hubben kan inte adresseras, så den svarar aldrig. Windows ger den statusen *nested too deeply*, och allt som sitter bakom den blir osynligt **[I]**.

## 2. Att läsa en UsbTreeView-rapport

UsbTreeView (Uwe Sieber, `winget install -e --id UweSieber.UsbTreeView` [6]) visar trädet och kan spara det som XML. Tre fält räcker:

| Fält | Betydelse |
|---|---|
| `Port Chain` | Vägen från rotporten, till exempel `1-9-3-4`. Antalet tal efter bussnumret är ledet: `1-9-3-4` sitter på led 3 **[I]**. |
| `Connection Status` | `0x01` ansluten, `0x07` för djupt nästlad [4] **[D]**. |
| `speed` | 0 = LS, 1 = FS, 2 = HS (USB 2), 3 = SS (USB 3) **[I]**, avläst ur rapportens hubbar. |

En USB 3-hubb syns två gånger: dess USB 2-halva på en USB 2-rotport och dess SuperSpeed-halva på en annan [5] **[D]**. De två halvorna har samma VID och revision men olika PID **[I]**.

**Bandregeln:** en rå rapport innehåller Bluetooth-namn, serienummer och användarsökvägar. Den ska inte klistras in någonstans utanför huset. Utskriften från `usbdjup.py` bär bara portkedjor, VID:PID och status, och CI kontrollerar att den inte läcker enhetsnamn **[I]**.

## 3. Var hubbarna gömmer sig

| Källa | Så syns den i rapporten | Kostar |
|---|---|---|
| USB 3-hubb | En HS-hubb och en SS-hubb med samma revision [5] **[D]** | 1 led i varje träd |
| Stor hubb (7 portar eller fler) | Två eller tre hubbkretsar i rad, ofta samma VID **[I]** | 2–3 led |
| USB-C-docka | VIA Labs-hubb (2109:xxxx) med en *Billboard*-enhet, ofta nätverkskort **[I]** | 1–2 led |
| USB-switch (för att dela mellan datorer) | Billig hubbkrets, till exempel Terminus FE1.1 (1A40:0101) **[I]** | 1 led per switch |
| Skärm eller tangentbord med USB-portar | Egen hubb bakom enheten **[I]** | 1 led |

## 4. Fallstudie (avidentifierad)

Rapporten visade en enda värdkontroller med 15 hubbar. Alla tunga grenar gick genom en rotport, och USB 3-halvan av den första hubben bar ingenting **[I]**:

| Led | Portkedja | Krets | Trolig roll |
|---|---|---|---|
| 1 | 1-9 | 05E3:0610 | USB 2-halvan av en USB 3-hubb (SS-halvan på 1-20, tom) |
| 2 | 1-9-3 | 05E3:0610 | nästa nivå: dockor, kamera, nätverk |
| 3 | 1-9-3-4 | 1A40:0101 | switch 1 (seriell adapter och tangentbordsemulator sitter direkt i den) |
| 4 | 1-9-3-4-1 | 05E3:0610 | hubb med tangentbord |
| 5 | 1-9-3-4-1-4 | 05E3:0610 | hubb |
| **6** | 1-9-3-4-1-4-4 | 1A40:0101 | **switch 2: vägras, 0x07** |

En annan gren (1-7) hade två hubbar som bara drev en ljudadapter, så det fanns gott om plats där **[I]**.

**Åtgärd:** flytta en hubb ovanför det trasiga ledet direkt till en rotport. `usbdjup.py` listar alternativen från minsta flytt (närmaste förfadern, som sparar 4 led) till största (switch 1, som sparar 2 led men tar med hela grenen). Flytten av switchens datorsladd behåller delningen mellan datorerna **[I]**.

## 5. Dela kringutrustning mellan flera datorer utan djupa kedjor

| Steg | Åtgärd | Kostnad | Löser | Risker och förbehåll |
|---|---|---|---|---|
| 1 | Platta ut: koppla enheter direkt i switchens utgångar och flytta grenar till lediga rotportar | 0 kr | Djupet | Inga **[I]** |
| 2 | En USB 3-switch med fyra ingångar i stället för två switchar i serie | några hundralappar | 1 led eller mer, och en USB 3-väg | Välj en modell med USB 3 hela vägen, annars hamnar allt i USB 2-trädet [3] **[D]/[I]** |
| 3 | Mus och tangentbord över nätverket: Input Leap (Windows, Linux, macOS) [7] eller Mouse Without Borders i PowerToys | 0 kr | Tar bort switchen för mus och tangentbord helt | Input Leaps officiella repo arkiverades i juli 2026. Fortsättningen Leapdesk KVM finns, men underhållet är ej verifierat [7][8] **[D]** |
| 4 | USB över IP: en alltid-på Linux-nod lånar ut enheter. `usbip` i Linuxkärnan, `usbipd-win` som server på Windows [9], `usbip-win2` som klient på Windows med WHLK- eller attesteringssignerade drivrutiner [10] | 0 kr | Djupet försvinner hos klienten | En klient per enhet åt gången. Syns inte i BIOS. **Okrypterat och utan lösenord**, så bara över Tailscale eller i hemnätet. Kameror fungerar dåligt **[I]** |
| 5 | VirtualHere i stället för usbip | ca 49–50 USD per server, engångslicens som gäller för alltid [11] **[D]** | Som steg 4, med bättre stöd för kameror | Licensen är knuten till en server **[D]** |
| — | Ett PCIe-kort med USB | några hundralappar | Ger ett nytt träd, men gränsen gäller per träd | Hjälper bara om det låter dig flytta grenar **[I]** |

[`usbserver-installera.sh`](../verktyg/usbserver-installera.sh) installerar steg 4 och vägrar alltid att låna ut lagringsenheter och hubbar.

## 6. Checklista

- [ ] Spara en UsbTreeView-rapport som XML och kör `python3 usbdjup.py rapport.xml`.
- [ ] Gör den minsta flytt verktyget föreslår och kör sedan rapporten igen. Avslutningskod 0 betyder klart.
- [ ] Kontrollera att USB 3-hubbarnas SS-halvor faktiskt bär något. Om inte ligger ett USB 2-led i vägen.
- [ ] Låna aldrig ut lagringsenheter över nätverket. Dela diskar som nätverksmappar.

## Källförteckning

1. Lightware, "USB Tiers Explained" – https://www.lightware.com/en/news/usb-tiers-explained-lightware-usbxpert
2. Microchip Developer Help, "USB Hubs" – https://developerhelp.microchip.com/xwiki/bin/view/applications/usb/hubs/
3. Electronic Design, "What's the Difference Between USB 2.0 and 3.0 Hubs?" – https://www.electronicdesign.com/technologies/embedded/article/21799762/whats-the-difference-between-usb-20-and-30-hubs
4. Microsoft Learn, `_USB_CONNECTION_STATUS` (usbioctl.h) – https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/usbioctl/ne-usbioctl-_usb_connection_status
5. Wikipedia, "USB 3.0" (avsnittet om hubbar) – https://en.wikipedia.org/wiki/USB_3.0
6. winget-pkgs, UweSieber.UsbTreeView – https://winstall.app/apps/UweSieber.UsbTreeView
7. Input Leap – https://github.com/input-leap/input-leap
8. Leapdesk KVM – https://github.com/bryceishunter/leapdesk-kvm
9. usbipd-win – https://github.com/dorssel/usbipd-win
10. usbip-win2 – https://github.com/vadimgrn/usbip-win2
11. VirtualHere, köp och licenser – https://www.virtualhere.com/purchase
