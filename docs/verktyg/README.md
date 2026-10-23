# Verktyg

Små, generiska verktyg som följer med utredningarna. Inget av dem bär husdata: inga
värdnamn, adresser eller sökvägar. Det kontrolleras i CI.

| Verktyg | Vad det gör | Hör till |
|---|---|---|
| [`usbdjup.py`](usbdjup.py) | Läser en UsbTreeView-rapport (XML), visar varje hubbkedja med sitt djup, flaggar allt som sitter djupare än fem hubbar och föreslår den minsta flytt som löser det. | [USB-topologi](../utredningar/usb-topologi.md) |
| [`usbserver-installera.sh`](usbserver-installera.sh) | Gör en Linux-nod till USB/IP-server och lägger till kommandot `usbdela`. Vägrar alltid lagringsenheter och hubbar. | [USB-topologi](../utredningar/usb-topologi.md), avsnitt 5 |
| [`diarbank.py`](diarbank.py) | Kör bänkplanen för diarisering (B, R1–R4, R7 och kombinationen R8) en variabel i taget och sätter utredningens band GRÖNT/GRÅTT/RÖTT/OMÄTT på varje körning. | [Diarisering på RTX 3080](../utredningar/diarisering-rtx3080.md), avsnitt 4 |
| [`diarjamfor.py`](diarjamfor.py) | Jämför två diariseringar (RTTM), till exempel Q8_0 mot BF16 eller en port mot sin referens. Mäter turer, DER (missat, falsklarm, förväxling) och etikettskillnad på 10 ms, och sätter banden för experiment 1–3. | [Svensk talarseparerad transkription](../utredningar/svensk-talarseparerad-transkription-2026.md), avsnitt 7 |
| [`asrbank.py`](asrbank.py) | Mäter KB-Whisper i olika precision och storlek (WER och RTF) och sätter banden för utredning 2:s experiment 5 (fp16 mot int8) och 6 (small mot medium på CPU). | [Svensk talarseparerad transkription](../utredningar/svensk-talarseparerad-transkription-2026.md), avsnitt 7 |

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

## diarbank

```
pip install "pyannote.audio==4.0.7" psutil      # med ett CUDA-bygge av torch
export HF_TOKEN=…                               # efter att villkoren för community-1 är godkända
python3 diarbank.py --lista                     # visa planen
python3 diarbank.py --ljud klipp.wav --ref klipp.rttm --ut resultat.json
```

Bänken följer utredningens avsnitt 4 rad för rad:
- **Uppvärmning:** 30 s svagt brus, som kastas.
- **Baslinjen:** först en kall körning, sedan median av `--upprepa` varma (standard 3).
- **En variabel i taget:** varje variant får en kastad körning först.

Den mäter:
- väggtid och **RTFx**, det vill säga ljudlängd delad med väggtid;
- tid per steg ur pyannotes hook, och klustringen för sig;
- toppen för allokerat och reserverat grafikminne;
- CPU-last om `psutil` finns;
- antal talare;
- **DER** med `pyannote.metrics`, collar 0 och ±0,25 s (pyannotes `collar=0.5`).

Utan `--ref` mäts DER mot baslinjens egen utdata (självkonsistens).

**R2 (TF32)** kopplar in sig efter pyannotes `fix_reproducibility()`, som annars stänger av TF32 vid varje anrop. **R4 (steget)** skriver direkt till segmenteringens `step` i sekunder. Pipelinens attribut läses bara när den skapas, så det räcker inte att ändra det. **R8** kombinerar alla gröna varianter. Från R3 och R4 tas bara den snabbaste. R5, R6 och R9–R11 ingår inte och redovisas som OMÄTT.

**Testklipp:** bara grönt ljud. Utredningen föreslår AMI-mötet ES2004a, som är CC BY 4.0. Referenser i RTTM finns i [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup). Bänken sparar inga transkript och ingen utdata, bara tider, minne och fel. Token läses ur en miljövariabel och skrivs aldrig ut.

**Turparitet:** med `--rttm-katalog d/` sparas varje variants diarisering som `d/<körning>.rttm`. Sedan avgör `python3 diarjamfor.py --ref d/B.rttm --hyp d/R4b.rttm` om ändringen flyttade talarturerna. Spara bara RTTM för testklipp: RTTM från riktigt ljud är husdata.

**Tidsåtgång:** med standardvärdena blir det ungefär 40 körningar. Går ett 9-minutersklipp i referensmätningens takt, cirka 3 minuter per körning, tar bänken ett par timmar. Börja med `--upprepa 1`, eller ett urval med `--korningar B,R2,R4b`.

**Avslutningskod:**
- `0`: klart.
- `2`: rimlighetskontrollen föll, eftersom segmenteringsmodellen inte ligger på GPU:n. Felsök då torch-bygget och drivrutinen innan något annat mäts.
- `1`: fel.

**Läge:** syntaxkontrollerad, och provad mot en attrapp som återger referensmätningens takt. Attrappen täcker banden, kombinationen, urvalet och DER mot referens och mot baslinjen. **Bänken är inte körd på en GPU än. Dess första riktiga mätning är OMÄTT** tills någon har gjort den. Kodvägen mot pyannote följer källan till 4.0.7.

Prov: `python3 -m unittest discover -s docs/verktyg/prov`. Proven behöver varken torch eller pyannote.

## diarjamfor

```
python3 diarjamfor.py --ref bf16.rttm --hyp q8.rttm --experiment 2
python3 diarjamfor.py --ref bf16/ --hyp q8/ --experiment 2 --ut resultat.json   # kataloger, paras på filnamn
```

Verktyget är oberoende av körmotor. Det enda det behöver är två RTTM-filer, till exempel från audio.cpp, NeMo, parakeet.cpp eller pyannote.

**Mått, per fil och viktat totalt:**
- **turer:** antal segment och avvikelsen i procent;
- **DER:** med referensen som facit, collar 0 och överlapp inräknat, uppdelad på missat, falsklarm och förväxling;
- **etikettskillnad:** andel 10 ms-ramar där etiketterna skiljer sig utan matchning. Måttet är bara meningsfullt när båda körningarna kommer från samma modell.

Talarna matchas optimalt med den ungerska metoden. Den är provad mot uttömmande sökning i 200 slumpfall.

**Korskontroll:** utan självöverlapp ger verktyget samma DER som pyannote.metrics 3.2.1. Största skillnaden över 30 slumpade femminutersmöten var 0,0005 procentenheter. Självöverlapp hos samma talare räknas en gång, som i NIST:s md-eval. Ett möte på 15 minuter med fem talare tar under en sekund.

**Band (utredningens avsnitt 7):**
- **Experiment 1:** GRÖNT vid DER ≤ 2 % och turer inom ±1 %. RÖTT vid DER > 6 % eller turer utanför ±3 %.
- **Experiment 2 och 3:** GRÖNT vid turer inom ±2 % och DER ≤ 0,5 %. RÖTT vid turer utanför ±6 % eller DER > 1,5 %.
- Däremellan GRÅTT.

**Avslutningskod:** `0` vid GRÖNT eller GRÅTT, `2` vid RÖTT, `1` vid fel. Verktyget läser bara tider och etiketter, aldrig ljud eller text.

Prov: 16 st i `prov/test_diarjamfor.py`. De kräver inga beroenden.

## asrbank

```
pip install faster-whisper pynvml                      # pynvml bara för grafikminnet
python3 asrbank.py --manifest fleurs-sv-test.tsv --experiment 5            # large fp16 mot int8 på GPU
python3 asrbank.py --manifest egna.tsv --experiment 6 --tradar 4           # small mot medium på CPU
python3 asrbank.py --manifest m.tsv --varianter large:float16:cuda,large:int8:cuda --revision strict
```

**Manifestet** är en TSV med två kolumner, `ljudfil<TAB>referenstext`, och sökvägarna är relativa till manifestet. Utredningen föreslår FLEURS sv_se test (CC BY 4.0), där KBLab redovisar 5,4 % WER för large i fp16. Ljudet och transkriptionerna finns i datasetrepot `google/fleurs` på Hugging Face, under `data/sv_se/`. Kontrollera kolumnernas namn där innan Du bygger manifestet [I].

**Avkodningen är fast:** svenska, beam 5, `condition_on_previous_text=False` (KBLab:s råd mot hallucinationer) och ingen VAD. Bara variabeln ändras mellan körningarna.

**WER** räknas på hela korpusen efter normalisering: gemener, skiljetecken bort, blanksteg ihopslagna. Å, ä och ö behålls, men siffror normaliseras inte.

**RTF** är beräkningstid delad med ljudlängd. Laddningstiden redovisas för sig. Grafikminnet är toppen för hela kortet, som pynvml ser det, och blir OMÄTT utan pynvml.

**Band (utredningens avsnitt 7):**
- **Experiment 5:** ΔWER(int8 − fp16) ≤ 0,3 procentenheter ger GRÖNT, 0,3–1,0 GRÅTT och över 1,0 RÖTT, och då är fp16 obligatoriskt på GPU.
- **Experiment 6:** medium RTF ≤ 0,5 ger GRÖNT, 0,5–1,0 GRÅTT och över 1,0 RÖTT, och då gäller small på CPU-noden.

Bänken sparar inga transkript. **Läge:** provad mot en attrapp, **inte körd med riktiga modeller än (OMÄTT)**. API-anropen följer faster-whisper 1.2.1.

Prov: 18 st i `prov/test_asrbank.py`. De kräver inga beroenden.
