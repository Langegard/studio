# Svensk lokal talarseparerad transkription — läget i september 2026

*Teknisk omvärldsbevakning. Alla källor lästa 2026-09-29. Markering: **[D]** = dokumenterat/uppmätt av andra (med källa), **[I]** = egen slutsats/inferens. Saknas belägg skrivs "ej verifierat".*

## Sammanfattning

1. **ASR:** KB-Whisper från KBLab (Kungliga biblioteket) är den tydliga standarden för svensk lokal taligenkänning: `kb-whisper-large` når 5,4 / 4,1 / 5,2 % WER på FLEURS / Common Voice / NST mot 7,8 / 9,5 / 11,3 % för OpenAI whisper-large-v3, och redan `kb-whisper-small` slår OpenAI:s large-v3 **[D]**. Alla storlekar är Apache-2.0 och levereras i HF-, CTranslate2-, GGML- och ONNX-format **[D]**.
2. **Diarisering:** NVIDIA Nemotron 3 Diarization (100 M parametrar, OpenMDW-1.1, släppt 2026-09-23) är i dag bästa öppna diariserare enligt både NVIDIA:s egna tabeller och den oberoende VoiceArena-benchen (14,72 % DER mot 30,61 % för pyannote community-1) **[D]**.
3. **Kvantiseringsparitet:** Q8_0-GGUF ger samma DER som BF16 i alla publicerade mätningar, men **inte** samma talarturer: paketförfattaren till audio.cpp-GGUF:en rapporterar exakt samma fenomen som referensfyndet (161 → 152 turer på en 311-sekunders inspelning), och en annan GGUF-konvertering rapporterar ~6 % färre segment för Q8_0 än F16 **[D]**. Referensfyndet är alltså reproducerat av andra och bör ses som en egenskap hos kvantiseringen av sigmoid-huvudet, inte ett lokalt fel **[I]**.
4. **Pipelines:** whisperX pekar redan ut `KBLab/wav2vec2-large-voxrex-swedish` som svensk alignmentmodell och stödjer KB-Whisper via CTranslate2 **[D]**; NeMo-Speech.cpp, audio.cpp och parakeet.cpp erbjuder Nemotron-diarisering utan Python **[D]**.
5. **Facit:** RixVox-v2 (ODC-By, ~23 000 h) har talar-id och start/slut per segment, men segmenten är ASR-orienterade 30-sekundersbitar, inte RTTM-facit för diarisering **[D]**; ett sådant måste byggas **[I]**.
6. **Hårdvara:** 6 GB räcker för KB-Whisper large i int8 + Nemotron; 10 GB räcker för fp16 + Nemotron BF16; en 4B-klassad CPU-nod bör köra KB-Whisper medium/small (ct2 int8 eller GGML q5_0) + Nemotron Q8_0/ONNX-int8 **[I]**, med sifferunderlag i avsnitt 5.

---

## 1. ASR för svenska (lokalt)

### 1.1 KB-Whisper-familjen

KBLab släppte i februari 2025 fem finjusterade Whisper-modeller (tiny, base, small, medium, large-v3) tränade på >50 000 timmar svenskt tal i två steg: 56 514 h "Stage 1" (lågt kvalitetsfilter) och 8 533 h "Stage 2" (BLEU ≥ 0,7 m.m.), med undertexter, riksdagsinspelningar (RixVox-v2), ISOF-dialektmaterial och NST som källor ([modellkort kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large), [Interspeech-artikel "Swedish Whispers"](https://arxiv.org/abs/2505.17538)) **[D]**.

**WER (%), Stage 2 "standard", enligt modellkortet **[D]**:**

| Storlek | Param. | FLEURS KBLab / OpenAI | Common Voice KBLab / OpenAI | NST KBLab / OpenAI |
|---|---|---|---|---|
| tiny | 39 M | 13,2 / 59,2 | 12,9 / 67,8 | 11,2 / 85,2 |
| base | 74 M | 9,1 / 39,6 | 8,7 / 52,1 | 7,8 / 53,4 |
| small | 244 M | 7,3 / 20,6 | 6,4 / 26,4 | 6,6 / 26,4 |
| medium | 769 M | 6,6 / 12,1 | 5,4 / 15,8 | 5,8 / 17,1 |
| large-v3 | 1,55 B | 5,4 / 7,8 | 4,1 / 9,5 | 5,2 / 11,3 |

Källa: [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large); parameterantal från [artikeln](https://arxiv.org/abs/2505.17538). Utvärderingen använder FLEURS train+test, Common Voice 16.1 train+val+test och NST-testsetet som hold-out **[D]**.

**Varianter (revisioner):** varje storlek har tre Stage-2-checkpoints: `standard` (default), `subtitle` (kondenserad stil) och `strict` (mer ordagrann). Skillnaderna i WER är små (large: 5,4 / 5,3 / 5,3 på FLEURS) ([modellkort](https://huggingface.co/KBLab/kb-whisper-large)) **[D]**. För talarseparerad transkription är `strict` sannolikt bäst, eftersom kondenserad undertextstil tappar ord som annars skulle tilldelas talare **[I]**.

**Format som KBLab själva publicerar per storlek **[D]** ([filträd large](https://huggingface.co/KBLab/kb-whisper-large/tree/main), [small](https://huggingface.co/KBLab/kb-whisper-small/tree/main), [medium](https://huggingface.co/KBLab/kb-whisper-medium/tree/main), [tiny](https://huggingface.co/KBLab/kb-whisper-tiny/tree/main), [base](https://huggingface.co/KBLab/kb-whisper-base/tree/main)):**

| Storlek | HF safetensors | CTranslate2 `model.bin` (fp16) | GGML `ggml-model.bin` (f16) | GGML `ggml-model-q5_0.bin` | ONNX-mapp |
|---|---|---|---|---|---|
| tiny | 115 MB | 75,5 MB | 77,7 MB | 29,9 MB | ja |
| base | 198 MB | 145 MB | 148 MB | 55,3 MB | ja |
| small | 563 MB | 484 MB | 488 MB | 175 MB | ja |
| medium | 1,63 GB | 1,53 GB | 1,53 GB | 539 MB | ja |
| large-v3 | 3,22 GB | 3,09 GB | 3,10 GB | 1,08 GB | ja (fp16, int8, q4, q4f16, bnb4, uint8) |

Filstorlekarna är exakta bytevärden från filträden avrundade **[D]**. ONNX-mappen för large innehåller bl.a. `encoder_model_int8.onnx` (645 MB) och `decoder_model_merged_int8.onnx` (1,18 GB) **[D]**.

- **CTranslate2/faster-whisper ("ct2"):** `model.bin` + `vocabulary.json` ligger i huvudrepot, så `WhisperModel("KBLab/kb-whisper-large", compute_type="float16"|"int8")` fungerar direkt ([modellkort](https://huggingface.co/KBLab/kb-whisper-large)) **[D]**. Det finns även community-ct2-speglar av `strict`/`subtitle`-revisionerna (t.ex. [q-henric/kb-whisper-large-strict-ct2](https://huggingface.co/q-henric/kb-whisper-large-strict-ct2)) **[D]**.
- **Distil:** KBLab har **inte** publicerat någon distillerad KB-Whisper; en Hub-sökning på "kb-whisper" (2026-09-29) visar inga `distil`-artefakter från KBLab ([sökresultat via Hub-API](https://huggingface.co/models?search=kb-whisper)) **[D]**. KBLab:s egen `easytranscriber` exemplifierar med `distil-whisper/distil-large-v3.5`, som är engelskt ([easytranscriber](https://github.com/kb-labb/easytranscriber)) **[D]**.
- **MLX/CoreML:** endast community-konverteringar (t.ex. [bratland/kb-whisper-large-mlx](https://huggingface.co/bratland/kb-whisper-large-mlx), [jegeblad/kb-whisper-large-mlx-q8](https://huggingface.co/jegeblad/kb-whisper-large-mlx-q8), [mickekringai/kb-whisper-coreml](https://huggingface.co/mickekringai/kb-whisper-coreml)) **[D]**; kvalitet ej verifierat.
- **Licens:** `apache-2.0` i modellkortets metadata för samtliga fem storlekar ([kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large)) **[D]**.
- **Hastighet/VRAM:** KBLab publicerar ingen RTF- eller VRAM-data (ej verifierat). Eftersom arkitekturen är identisk med OpenAI:s Whisper gäller generella faster-whisper/whisper.cpp-mätningar (avsnitt 5) **[I]**.
- **Kvantiserad WER:** KBLab publicerar ingen WER för int8/q5_0-artefakterna — ej verifierat. Modellkortet varnar bara att `int8` "may reduce accuracy" ([modellkort](https://huggingface.co/KBLab/kb-whisper-large)) **[D]**.

### 1.2 OpenAI Whisper som baslinje

whisper-large-v3 ligger på 7,8 / 9,5 / 11,3 % WER på svenska FLEURS/CV/NST i KBLab:s mätning **[D]**. large-v3 tränades på 1 M h svagt märkt + 4 M h pseudomärkt ljud, varav 2 119 h svenska enligt KBLab:s artikel ([HF-kort large-v3](https://huggingface.co/openai/whisper-large-v3), [artikel](https://arxiv.org/abs/2505.17538)) **[D]**. **large-v3-turbo** har 4 dekoderlager i stället för 32, är MIT-licensierad och "performs similarly to large-v2" över språk, men OpenAI ger ingen svensk siffra ([HF-kort turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [OpenAI-diskussion #2363](https://github.com/openai/whisper/discussions/2363)) **[D]**; svensk WER för turbo: ej verifierat. I faster-whisper mättes turbo till 2 537 MB VRAM mot 4 521 MB för large-v3 (fp16) ([faster-whisper #1030](https://github.com/SYSTRAN/faster-whisper/issues/1030)) **[D]**.

### 1.3 Andra lokala ASR-alternativ med svenska (2025–2026)

| Modell | Svenska? | Publicerad svensk WER | Licens | Kommentar |
|---|---|---|---|---|
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ja (25 EU-språk) | FLEURS sv 15,08; CoVoST2 sv 20,16 **[D]** | CC-BY-4.0 **[D]** | Snabb transducer; GGUF via NeMo-Speech.cpp/parakeet.cpp **[D]** |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | ja (25 EU-språk) | FLEURS sv 9,57; CoVoST2 sv 13,32 **[D]** | CC-BY-4.0 **[D]** | ASR + översättning |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | ja, tier "broad-coverage" (sv-SE) **[D]** | ej publicerad — ej verifierat | OpenMDW-1.1 **[D]** | Streaming, kopplas officiellt till Nemotron 3 Diarization **[D]** |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (och 0.6B) | ja (30 språk inkl. sv) **[D]** | ej publicerad per språk — ej verifierat | Apache-2.0 **[D]** | ForcedAligner stödjer 11 språk, **inte** svenska **[D]** |
| [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | nej — språklistan är en/fr/de/es/it/pt/nl/hi **[D]** | – | Apache-2.0 | Ej relevant för svenska |
| [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en) | nej (en/fr) **[D]** | – | – | – |
| Moonshine, Granite Speech | svenska ej listad i sökträffar — ej verifierat | – | – | Utelämnas |

Slutsats **[I]**: ingen multilingual modell närmar sig KB-Whisper på svenska; Canary-1B-v2 (9,57 FLEURS) är den enda som ligger i närheten av KB-Whisper *base*. KB-Whisper är förstahandsval; Parakeet/Canary är relevanta främst för att de körs nativt i NeMo-Speech.cpp/parakeet.cpp tillsammans med Nemotron-diarisering (avsnitt 3).

---

## 2. Diarisering

### 2.1 NVIDIA Nemotron 3 Diarization

**Arkitektur **[D]**:** 31-lagers Transformer-encoder med RoPE, 10 ms mel-features stackas ×8 till 80 ms encoderram, Conv1D-huvud uppsamplar till 10 ms, 100 M parametrar; Sortformer-princip (utgångskanaler ordnade efter talarens första ankomst) med Arrival-Order Speaker Cache (AOSC) + FIFO för streaming; max 8 talare; latensprofiler 0,32 s – 30,4 s ([modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization), [Streaming Sortformer-artikeln](https://arxiv.org/abs/2507.18446), [Sortformer-artikeln](https://arxiv.org/abs/2409.06656)). Släppt 2026-09-23 **[D]**.

**Licens **[D]**:** [OpenMDW License Agreement v1.1](https://openmdw.ai/license/1-1/) (metadata `license: openmdw-1.1`), "ready for commercial or non-commercial use". OpenMDW förvaltas av Linux Foundation och ger royaltyfri rätt under upphovsrätt, patent och databasrätt med endast krav på att bevara licenstext och attribution ([LF-pressmeddelande](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families), [OpenMDW/openmdw på GitHub](https://github.com/OpenMDW/openmdw)) **[D]**.

**Träningsdata **[D]**:** ~10 000 h riktiga samtal (Fisher, AMI, ICSI, VoxConverse, AISHELL-4, DIHARD III dev, CALLHOME, AliMeeting, DiPCo, NOTSOFAR1, DISPLACE, licensierad David AI-data, 5 000 h YODAS-v2) + 82 611 h simulerade blandningar. Språk: engelska, mandarin, hindi m.fl.; **svenska nämns inte** ([modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization)). Diarisering är i princip språkoberoende, men detta är ej verifierat för svenska av NVIDIA **[I]**.

**BF16-referensprestanda (DER %, offline 30,4 s, overlap inkluderad, collar 0 utom CALLHOME 0,25) **[D]**:**

| Testset | Nemotron 3 | Streaming Sortformer 4spk-v2.1 |
|---|---|---|
| DIHARD III full | 12,73 | 19,09 |
| CALLHOME-part2 | 9,10 | 10,32 |
| AMI test MHM (forced-aligned RTTM) | 9,25 | 15,81 |
| AMI test SDM | 11,14 | 21,42 |
| AliMeeting near / far | 6,40 / 10,47 | 11,57 / 13,69 |
| NOTSOFAR1 MHM / SC | 6,7 / 11,00 | 21,77 / 30,49 |

Källa: [modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization). RTFx (ljudtid/beräkningstid) offline, batch 1, eager: 1 340; kompilerad: 4 385; batch 32: 12 196–15 113, på RTX PRO 5000 i BF16 **[D]**. NVIDIA påpekar att DER bara är jämförbar med samma referens-RTTM, collar och overlap-inställning ([diarization_evaluation.md](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/diarization_evaluation.md)) **[D]**.

**Oberoende bench:** VoiceArena Diarization-Bench v1 (139 far-field engelska samtal, 21,97 h, 280+ talare, collar 0 ms, overlap räknas): Nemotron 3 14,72 % DER (#1), DiariZen WavLM Large 19,34, pyannoteAI Precision-3 20,56, Precision-2 23,41, Streaming Sortformer 4spk v2 24,61, pyannote Community-1 30,61 ([VoiceArena](https://voicearena.com/diarization-bench), återgivet i [Bonenk/Nemotron-3-Diarization-GGUF](https://huggingface.co/Bonenk/Nemotron-3-Diarization-GGUF)) **[D]**.

**Runtimes **[D]**:**

| Runtime | Format/precision | Källa |
|---|---|---|
| NeMo Speech (Python, Apache-2.0) | `.nemo` (199 MB), BF16/FP32 | [modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization), [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) |
| HF Transformers (nativt stöd, `AutoModelForAudioFrameClassification`) | safetensors 397 MB fp32 | [modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| NeMo-Speech.cpp (NVIDIA, C++/ggml, Apache-2.0) | GGUF; NVIDIA lägger själva `Nemotron-3-Diarization.q8_0.gguf` (107 MB) i officiella repot | [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp), [PR #50](https://github.com/NVIDIA/NeMo-Speech.cpp/pull/50), [filträd](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/main) |
| audio.cpp (Apache-2.0) | GGUF BF16 189,5 MiB / Q8_0 101,7 MiB; CUDA, Vulkan, Metal, CPU | [audio-cpp/Nemotron-3-Diarization-GGUF](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF), [audio.cpp](https://github.com/0xShug0/audio.cpp), [LICENSE](https://github.com/0xShug0/audio.cpp/blob/main/LICENSE) |
| parakeet.cpp (MIT) | GGUF F16 192 MB / Q8_0 104 MB; offline endast | [mudler/Nemotron-3-Diarization-GGUF](https://huggingface.co/mudler/Nemotron-3-Diarization-GGUF), [PR #71](https://github.com/mudler/parakeet.cpp/pull/71) |
| transcribe.cpp/Glimpse | GGUF F32/F16/Q8_0 | [Glimpse-Dictation/Nemotron-3-Diarization-gguf](https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf) |
| ONNX Runtime (inkl. WASM i webbläsare) | fp32 398 MB, fp16 199 MB, int8/"quantized" 120 MB, q4 83 MB | [onnx-community/Nemotron-3-Diarization-ONNX](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX), [NealCaren/…-ONNX](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX), [s0undy/…-onnx-int8](https://huggingface.co/s0undy/nemotron-3-diarization-onnx-int8) |
| MLX (Apple) | 8-bit 107 MB; INT8 g64 ~102 MiB | [mlx-community/…-8bit](https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit), [aufklarer/…-MLX-INT8](https://huggingface.co/aufklarer/Nemotron-3-Diarization-100M-MLX-INT8) |

### 2.2 Kvantiseringsparitet: q8_0/int8 mot bf16/fp32

Detta är kärnfrågan för referensfyndet (Q8_0 tappade 9 av 161 turer på 311 s). Publicerade belägg, sorterade efter styrka:

1. **audio.cpp:s egna GGUF-kort rapporterar samma sak **[D]**.** Kortet skriver: *"Q8_0 is not exact-parity-safe … it can change speaker boundaries and turn counts."* I deras CUDA-validering mot C++-BF16 (RTX 5090, profil `very_high`) behöll fem korta inspelningar sina turantal med mestadels 10 ms-förskjutningar; en 97,6-sekunders inspelning behöll 31 turer (0,174 % av tidslinjen skilde); **en 311-sekunders inspelning gick från 161 till 152 turer** och 1,174 % av tidslinjen skilde. Upprepade körningar inom samma precision gav identiska turer ([audio-cpp/Nemotron-3-Diarization-GGUF](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF)). Kortet betonar att procenten är same-label-aktivitetsskillnad på 10 ms-rutnät, **inte DER**. Q8_0 kvantiserar 130 av 362 tensorer; 229 förblir BF16, 2 F16 och melfiltret F32 **[D]**.
2. **parakeet.cpp-GGUF rapporterar ~6 % färre segment **[D]**.** Paritet mot PyTorch: F16 583 segment mot 581 i referensen, **Q8_0 547 segment**; kortet förklarar det med "quantization noise in the sigmoid speaker head" och rekommenderar F16 för maximal noggrannhet ([mudler/Nemotron-3-Diarization-GGUF](https://huggingface.co/mudler/Nemotron-3-Diarization-GGUF)). PR:n som lade in stödet hävdar samtidigt att "Q8_0 keeps the same segments" på deras kortare testklipp ([PR #71](https://github.com/mudler/parakeet.cpp/pull/71)) — effekten syns alltså främst på längre material **[I]**.
3. **DER-nivån påverkas inte mätbart **[D]**.** Glimpse-porten på AMI IHM-test (16 möten, 9,06 h, NVIDIA:s forced-alignment-RTTM, collar 0): referens fp32 9,22 % DER, port F32 9,23, F16 9,23, **Q8_0 9,23** — men med Missed 4,79 mot 4,68 och False alarm 3,58 mot 3,67 ([Glimpse-Dictation-kortet](https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf)). audio.cpp PR #677 mätte på en 120-sekunders ström DER mot NeMo-Python: CPU F32 4,2 %, CUDA F32 4,4 %, CUDA BF16 5,1 %, CUDA Q8_0 4,3 % ([PR #677](https://github.com/0xShug0/audio.cpp/pull/677)) **[D]**. ONNX-int8 (dynamisk, per kanal) gav 99,994 % ramöverensstämmelse och segmentgränser inom 10 ms på ett 87-sekunders 4-talarklipp ([NealCaren-kortet](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX)); MLX-INT8 gav 0,01–0,07 % beslutsavvikelse vid tröskel 0,25 ([aufklarer-kortet](https://huggingface.co/aufklarer/Nemotron-3-Diarization-100M-MLX-INT8)) **[D]**.
4. **Svensk data:** en ONNX-int8-build som används i en svensk Android-app uppges "match fp32 within noise" på ett svenskt testset byggt av Riksdagsdebatter märkta med RixVox-v2 plus simulerade samtal ([s0undy-kortet](https://huggingface.co/s0undy/nemotron-3-diarization-onnx-int8)); det länkade benchmark-repot kunde inte hämtas (HTTP 404) — siffrorna är därför **ej verifierade**.
5. **NVIDIA:** den officiella `q8_0.gguf` publiceras utan någon paritets- eller DER-uppgift — ej verifierat **[D]**.

**Tolkning **[I]**:** Q8_0 är "DER-paritetssäker" men inte "tur-paritetssäker". Mönstret (färre segment, något högre Missed, något lägre FA) tyder på att 8-bitars brus i sigmoid-huvudet systematiskt drar sannolikheter under tröskeln 0,5 vid korta inskott och gränser. Referensfyndets 9/161 (5,6 %) ligger mellan audio.cpp:s 5,6 % och parakeet.cpp:s 6,2 % och är sannolikt samma effekt. Eftersom paketen skiljer sig i *vilka* tensorer som kvantiseras (audio.cpp: 130 tensorer; parakeet.cpp: endast linjära vikter; Glimpse: silence-embedding hålls F32) finns det utrymme för en Q8-variant som lämnar huvudet (encoder_proj + Conv1D-upsampler) okvantiserat — men ingen sådan är publicerad; ej verifierat. Praktisk följd: använd BF16 där turantal spelar roll (t.ex. talartilldelning av ord); besparingen med Q8_0 är bara ~90 MB vikter och ~130 MiB VRAM ([audio-cpp-kortet](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF)) **[D]**.

**Kända portproblem (ej kvantisering) **[D]**:** audio.cpp PR #677 hittade tre paritetsfel mot NeMo (ceil vs floor för antal melramar, oputsade paddade encoderrader, fel ordning på pre-emphasis och tail-padding) som gav 4 flippade ramar på ett 20-sekundersklipp före fixen och 0 efter ([PR #677](https://github.com/0xShug0/audio.cpp/pull/677)); de sista ~70 ms av en fil kan fortfarande skilja mot NeMo p.g.a. paddningsstrategi ([audio.cpp-modelldok](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/nemotron_3_diar.md)). Publicerade GGUF:er från före 2026-09-24 kräver en runtime som accepterar det äldre schemat ([PR #703](https://github.com/0xShug0/audio.cpp/pull/703)). Lärdom **[I]**: jämför alltid BF16-GGUF mot NeMo/Transformers-referensen *före* Q8-testet, annars blandas portfel och kvantiseringsfel.

### 2.3 pyannote community-1 (etablerat alternativ)

- Pipeline: segmentering (powerset) + WeSpeaker-embeddingar + VBx-klustring; utdata inkluderar *exclusive speaker diarization* för enklare koppling till ASR-tidsstämplar ([modellkort](https://huggingface.co/pyannote/speaker-diarization-community-1)) **[D]**.
- Licens: pipelinen CC-BY-4.0 (gated, kräver HF-token men går att köra offline efter nedladdning); koden pyannote.audio MIT ([modellkort](https://huggingface.co/pyannote/speaker-diarization-community-1), [pyannote-audio](https://github.com/pyannote/pyannote-audio)) **[D]**.
- DER (collar 0, overlap räknas): AMI IHM 17,0, AMI SDM 19,9, DIHARD 3 20,2, VoxConverse 11,2, CALLHOME 26,7 ([modellkort](https://huggingface.co/pyannote/speaker-diarization-community-1)) **[D]**. Jämfört med Nemotron 3 på samma korpora men *andra* referens-RTTM (AMI 9,25/11,14; DIHARD 12,73) — ej direkt jämförbart **[D]**, men VoiceArena mäter båda på samma facit: 30,61 mot 14,72 **[D]**.
- Hastighet: 31 s per timme ljud på H100 ([pyannote-audio README](https://github.com/pyannote/pyannote-audio)) **[D]**.
- **VRAM-varning:** pyannote.audio 4.0.3 rapporterades toppa på 9,54 GB för en 72-minutersfil (både community-1 och 3.1) mot 1,59 GB i 3.3.2; orsaken pekas ut till `discrete_diarization`/exclusive-beräkningen ([issue #1963](https://github.com/pyannote/pyannote-audio/issues/1963)) **[D]**. Om detta är åtgärdat i 4.0.4–4.0.7 (senaste 4.0.7) kunde inte verifieras — ej verifierat. På 6 GB-kortet är detta en reell risk för långa filer **[I]**.

### 2.4 Övriga alternativ (kort)

- [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1): föregångaren, max 4 talare, klart sämre (AMI MHM 15,81) **[D]**.
- DiariZen (BUT FIT): #2 på VoiceArena (19,34) **[D]**; lokal körbarhet/licens ej undersökt här.
- whisper.cpp `tinydiarize` (`-tdrz`): experimentell, endast engelsk `small.en-tdrz`, ger bara `[SPEAKER_TURN]`-markörer ([whisper.cpp README](https://github.com/ggml-org/whisper.cpp)) **[D]** — ej användbart för svenska **[I]**.

---

## 3. Kombinerade pipelines

| Kombination | Status | Belägg |
|---|---|---|
| **whisperX** = faster-whisper (KB-Whisper ct2) + wav2vec2-alignment + pyannote community-1 | Fungerar; KBLab ger kodexempel med `KBLab/wav2vec2-large-voxrex-swedish` som alignmentmodell, och whisperX har `"sv": "KBLab/wav2vec2-large-voxrex-swedish"` som default i `DEFAULT_ALIGN_MODELS_HF` | [kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large), [whisperX alignment.py](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py), [whisperX README](https://github.com/m-bain/whisperX) **[D]** |
| faster-whisper + pyannote (egen limning) | Samma komponenter som whisperX utan alignment; ord→talare via segmentöverlapp | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) **[D]** |
| **KB-Whisper + Nemotron 3** | Ingen färdig integration publicerad (ej verifierat); byggs genom att köra Nemotron (audio.cpp/NeMo-Speech.cpp/ONNX) → 10 ms-talaraktivitet, KB-Whisper (ct2) → ordtidsstämplar (helst wav2vec2-alignade), och tilldela varje ord den talare med störst överlapp | Komponentkällor ovan; kombinationslogiken är **[I]** |
| **NeMo-Speech.cpp `transcribe --diarize`** | Ordnivå-talartaggar automatiskt; ASR-modeller: Nemotron 3.5 ASR 0.6B (40 locales, sv-SE "broad-coverage"), Parakeet TDT 0.6B v3 (25 EU-språk inkl. sv), Nemotron-Speech Streaming (endast en), Parakeet CTC 1.1B (endast en) | [docs/cli.md](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md), [docs/asr/models.md](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/models.md), [Nemotron 3.5 ASR-kort](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) **[D]** |
| **NeMo end-to-end (Python)** | `speech_to_text_multitalker_streaming_infer.py`: Nemotron 3 Diar + Nemotron 3.5 ASR (masked ASR, 32 locales inkl. sv) eller Multitalker Parakeet (endast engelska) | [ASR_INTEGRATION_GUIDE.md](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md) **[D]** |
| **parakeet.cpp SAS** | `transcribe_and_diarize` med valfri Parakeet-GGUF (TDT v3 har svenska) + Nemotron F16/Q8; endast offline-diarisering | [mudler-kortet](https://huggingface.co/mudler/Nemotron-3-Diarization-GGUF), [PR #71](https://github.com/mudler/parakeet.cpp/pull/71) **[D]** |
| whisper.cpp + audio.cpp | Två separata CLI:er (GGML för KB-Whisper, GGUF för Nemotron); ingen gemensam ord→talare-logik | [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [audio.cpp](https://github.com/0xShug0/audio.cpp) **[D]**; sammanfogning **[I]** |
| **KBLab easyaligner / easytranscriber** | GPU-Viterbi forced alignment med wav2vec2-emissioner (MIT); easytranscriber påstår 35–102 % snabbare än whisperX med svenska modeller; pyannote används som VAD, inte diarisering | [easyaligner](https://github.com/kb-labb/easyaligner), [KBLab-blogg 2026-04-16](https://huggingface.co/blog/KBLab/easyaligner), [easytranscriber](https://github.com/kb-labb/easytranscriber) **[D]** |

**Ordnivå-alignment för svenska:** KBLab:s [wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) (CC0-1.0, WER 8,49 % på Common Voice utan LM) är den modell både whisperX och KBLab själva använder **[D]**; tidigare varianter finns ([wav2vec2-large-xlsr-53-swedish](https://huggingface.co/KBLab/wav2vec2-large-xlsr-53-swedish), [wav2vec2-base-voxpopuli-sv-swedish](https://huggingface.co/KBLab/wav2vec2-base-voxpopuli-sv-swedish)) **[D]**. Qwen3-ForcedAligner stödjer inte svenska **[D]**.

**Kända fallgropar **[D]** om inte annat anges:**
- whisperX kan inte tidsätta ord med tecken utanför wav2vec2-vokabulären (siffror, valutasymboler) och hanterar överlappande tal dåligt ([whisperX README](https://github.com/m-bain/whisperX)).
- whisperX 3.8.x kräver `pyannote-audio>=4.0.0`, `torch~=2.8`, `ctranslate2>=4.5` ([pyproject.toml](https://github.com/m-bain/whisperX/blob/main/pyproject.toml)); faster-whisper kräver cuBLAS/cuDNN 9 för CUDA 12 ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)).
- pyannote 4.0.x VRAM-toppar på långa filer (avsnitt 2.3).
- Nemotron kräver 16 kHz mono; ONNX-exporten täcker bara nätet — melfrontend och speaker-cache måste implementeras utanför grafen ([NealCaren-kortet](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX)).
- Nemotron ordnar turer kronologiskt i C++ men grupperat per talare i Python ([audio.cpp PR #660](https://github.com/0xShug0/audio.cpp/pull/660)).
- `subtitle`-revisionen av KB-Whisper kondenserar text; för talartilldelning bör `strict` eller `standard` användas **[I]**.
- Whisper-hallucinationer vid tystnad: KBLab tränade in 100 h tystnad för att dämpa detta, och modellkortet föreslår `condition_on_previous_text=False` ([artikel](https://arxiv.org/abs/2505.17538), [modellkort](https://huggingface.co/KBLab/kb-whisper-large)).

---

## 4. Facit och utvärdering

### 4.1 RixVox-v2

- ~23 000 h svenskt riksdagstal 1966–2024, byggt genom matchning och forced alignment av protokoll mot mediefiler; 3,6 M rader, 944 parquet-filer, 2 716 GB; licens **ODC-By 1.0** ([datasetkort](https://huggingface.co/datasets/KBLab/rixvox-v2), [Hub-metadata](https://huggingface.co/datasets/KBLab/rixvox-v2)) **[D]**.
- Fält relevanta för diarisering: `speaker_id` (SWERIK/Riksdagen), `name`, `start`, `end` (sekunder i källfilen `audio_file`), `speech_id`, `protocol_id`, `text_timestamps` (meningsnivå) **[D]**.
- **Begränsning:** segmenten är ≤30 s, ASR-orienterade och innehåller inte överlappande tal eller täta talarbyten; start/slut är alignade mot protokolltext, inte mot ljudgränser för talaraktivitet. Datasetets föregångare beskrivs av KBLab som lämpat för "creation of synthetic diarization datasets" snarare än som färdigt diariseringsfacit ([RixVox v1-kort](https://huggingface.co/datasets/KBLab/rixvox)) **[D]**. Pipelinen som skapade RixVox använde pyannote för att förfina talgränser ([kb-labb/rixvox](https://github.com/kb-labb/rixvox)) **[D]** — facitgränserna är alltså delvis modellgenererade **[I]**.
- **Använt som diariseringsfacit av andra:** ja, i den svenska Android-appens Nemotron-int8-benchmark ("Riksdagen debates labelled with RixVox-v2, and simulated conversations") ([s0undy-kortet](https://huggingface.co/s0undy/nemotron-3-diarization-onnx-int8)) **[D]**, detaljer ej verifierade.
- **Rekommenderat bygge **[I]**:** (a) återskapa RTTM per debatt från `speaker_id`+`start`/`end` i samma `audio_file` — ger riktiga byten mellan talare men glesa; (b) syntetiska samtal genom att konkatenera/överlappa RixVox-v2-segment från olika `speaker_id` (samma metod som NVIDIA använder med FastMSS, [modellkort](https://huggingface.co/nvidia/Nemotron-3-Diarization)); (c) en liten handmärkt svensk mötesmängd (t.ex. 5 × 10 min) för ekologisk validitet.

### 4.2 Andra svenska set

- **RixVox v1**: 5 493 h, 835 044 observationer från 1 194 talare, CC-BY-4.0, aeneas-alignerat på meningsnivå ([datasetkort](https://huggingface.co/datasets/KBLab/rixvox)) **[D]**.
- **Nord-Parl-TTS** (2025): ~5 090 h svenskt riksdagstal processat med talardiarisering för TTS-ändamål ([arXiv 2509.17988](https://arxiv.org/abs/2509.17988), endast abstrakt/sökträff läst) **[D, svagt]**; lämplighet som diariseringsfacit ej verifierat.
- Ett publikt svenskt **möteskorpus med RTTM-facit** kunde inte hittas (sökningar 2026-09-29) — ej verifierat att något saknas, men inget påträffades.
- Utvärderingsverktyg: NeMo:s [`score_diarization.py`](https://github.com/NVIDIA-NeMo/Speech) och kravlistan på rapporterade parametrar (collar, overlap, streaming-geometri, precision) i [diarization_evaluation.md](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/diarization_evaluation.md) **[D]**; pyannote.metrics för DER/JER **[D]**.

---

## 5. Hårdvaruprofiler

**Sifferunderlag (generiskt Whisper; KB-Whisper är storleksidentisk) **[D]**:**

| Källa | Mätning |
|---|---|
| [faster-whisper README](https://github.com/SYSTRAN/faster-whisper) | large-v2 GPU: fp16 4 525 MB VRAM (1 min 03 s för 13 min ljud), int8 2 926 MB (59 s); batch 8: fp16 6 090 MB, int8 4 500 MB. small CPU: fp32 2 257 MB RAM, int8 1 477 MB; whisper.cpp fp32 1 049 MB |
| [faster-whisper #1030](https://github.com/SYSTRAN/faster-whisper/issues/1030) | large-v3 fp16 4 521 MB VRAM, 52 s / 13 min ljud (RTF ≈ 0,067); large-v3-turbo 2 537 MB, 19 s |
| [whisper.cpp README](https://github.com/ggml-org/whisper.cpp) | RAM: tiny ~273 MB, base ~388 MB, small ~852 MB, medium ~2,1 GB, large ~3,9 GB (f16); q5_0 minskar disk/RAM ytterligare |
| [audio-cpp GGUF-kort](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF) | Nemotron på RTX 5090, inkl. CUDA-kontext: BF16 820–1 076 MiB, Q8_0 740–948 MiB; NeMo-Python 1 504–2 786 MiB; aggregerad RTF ≈ 0,0009–0,0011 (≈ 900–1 100× realtid) |
| [Glimpse-kortet](https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf) | Nemotron Q8_0 på Apple M2 Pro CPU: 59× realtid; Metal 339× |
| [parakeet.cpp PR #71](https://github.com/mudler/parakeet.cpp/pull/71) | 12,3 min 3-talarklipp på 6,8 s CPU (≈ 108× realtid) |
| [NealCaren-kortet](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX) | ONNX int8 i WASM (M3 Max): ~25× realtid |
| [pyannote #1963](https://github.com/pyannote/pyannote-audio/issues/1963) | community-1 i 4.0.3: 9,54 GB VRAM-topp (72 min fil) |

### (a) GPU 6 GB VRAM **[I]** (bygger på **[D]**-siffrorna ovan)

- **ASR:** KB-Whisper large ct2 **int8** (~2,9 GB) eller **medium fp16** (medium ≈ hälften av large → ~2,3 GB, ej uppmätt). Batchad whisperX (`batch_size=16`) på large fp16 riskerar OOM; kör `--compute_type int8` och `--batch_size 4`, som whisperX-README rekommenderar för lite minne **[D]**.
- **Diarisering:** Nemotron BF16-GGUF via audio.cpp (~0,8–1,1 GB inkl. kontext) — **inte** Q8_0, eftersom besparingen är ~130 MiB och turparitet brister. Kör sekventiellt (diarisering först, frigör, sedan ASR) om båda ska ligga på GPU.
- **Undvik:** pyannote 4.0.x på filer >30 min tills VRAM-regressionen är bekräftat löst.
- Förväntad total: ~4 GB topp vid sekventiell körning; ~5 GB om Nemotron ligger kvar laddad.

### (b) GPU 10 GB VRAM **[I]**

- KB-Whisper large ct2 **fp16** (~4,5 GB, batch 1; ~6,1 GB batch 8) + Nemotron BF16 (~1 GB) samtidigt: ~5,5–7 GB.
- Alternativt NeMo-Python-referensen för Nemotron (1,5–2,8 GB) för paritetstester.
- pyannote community-1 fungerar som jämförelse på korta filer; för långa filer kan 9,5 GB-toppen slå i taket — chunka eller köra separat.
- Här bör även BF16-vs-NeMo-paritet mätas, eftersom det är den enda profilen som rymmer båda.

### (c) CPU-nod, ~4B-klass **[I]**

- **ASR:** KB-Whisper **medium** via whisper.cpp GGML q5_0 (539 MB disk, ~2 GB RAM) eller ct2 int8 (~1,5 GB modell). Om realtidsfaktorn blir >1 på nodens kärnor: KB-Whisper **small** (852 MB RAM; whisper.cpp fp32 small tog 2 min 05 s för 13 min ljud på CPU i faster-whisper-benchen, RTF ≈ 0,16 **[D]**). WER-kostnaden small→medium→large är 7,3 → 6,6 → 5,4 på FLEURS **[D]**.
- **Diarisering:** Nemotron Q8_0-GGUF på CPU (audio.cpp/NeMo-Speech.cpp) eller ONNX-int8; båda är ≥25× realtid på laptop-CPU **[D]**, så BF16/F16 är också realistiskt på CPU (F16 48× på M2 Pro **[D]**) — välj F16/BF16 om turparitet prioriteras.
- Alignment: wav2vec2-large-voxrex (~1,2 GB fp32) på CPU är den tyngsta posten; mät RTF innan den görs obligatorisk.
- Realtidsfaktor för hela kedjan på CPU: ej verifierat; bör mätas (experiment 8).

---

## 6. Jämförelsetabell

| System / kombination | Svensk WER (testset) | DER (testset) | VRAM / RAM | RTF | Licens(er) | Mognad |
|---|---|---|---|---|---|---|
| KB-Whisper large ct2 fp16 | 5,4 FLEURS / 4,1 CV / 5,2 NST **[D]** | – | ~4,5 GB VRAM (large-v2/v3 fp16, generiskt) **[D]** | ≈0,067 (large-v3 fp16, generiskt) **[D]** | Apache-2.0 | Hög |
| KB-Whisper large ct2 int8 | int8-WER ej verifierat (fp16: 5,4) | – | ~2,9 GB VRAM **[D]** | ≈ fp16 (59 s vs 63 s / 13 min) **[D]** | Apache-2.0 | Hög |
| KB-Whisper large GGML q5_0 (whisper.cpp) | q5_0-WER ej verifierat | – | ~3,9 GB RAM (f16); q5_0 lägre, ej verifierat | ej verifierat | Apache-2.0 / MIT (whisper.cpp) | Hög |
| KB-Whisper medium | 6,6 / 5,4 / 5,8 **[D]** | – | ~2,1 GB RAM (whisper.cpp f16) **[D]** | ej verifierat | Apache-2.0 | Hög |
| KB-Whisper small | 7,3 / 6,4 / 6,6 **[D]** | – | ~852 MB RAM (whisper.cpp); 1 477 MB (ct2 int8 CPU) **[D]** | ≈0,16 CPU (whisper.cpp fp32, generiskt) **[D]** | Apache-2.0 | Hög |
| KB-Whisper base / tiny | 9,1 / 8,7 / 7,8; 13,2 / 12,9 / 11,2 **[D]** | – | ~388 / ~273 MB RAM **[D]** | ej verifierat | Apache-2.0 | Hög |
| OpenAI whisper-large-v3 | 7,8 / 9,5 / 11,3 **[D]** | – | 4 521 MB fp16 **[D]** | ≈0,067 **[D]** | MIT | Hög |
| OpenAI whisper-large-v3-turbo | ej verifierat | – | 2 537 MB fp16 **[D]** | ≈0,024 **[D]** | MIT | Hög |
| Parakeet TDT 0.6B v3 | FLEURS sv 15,08; CoVoST2 sv 20,16 **[D]** | – | ej verifierat | ej verifierat | CC-BY-4.0 | Hög |
| Canary 1B v2 | FLEURS sv 9,57; CoVoST2 sv 13,32 **[D]** | – | ej verifierat | ej verifierat | CC-BY-4.0 | Medel |
| Qwen3-ASR 1.7B | sv stöds; WER ej verifierat | – | ej verifierat | ej verifierat | Apache-2.0 | Medel |
| Nemotron 3.5 ASR streaming 0.6B | sv "broad-coverage"; WER ej verifierat | – | ej verifierat | ej verifierat | OpenMDW-1.1 | Ny (2026-06) |
| Nemotron 3 Diarization BF16 (NeMo) | – | DIHARD III 12,73; AMI MHM 9,25; CALLHOME 9,10 **[D]**; VoiceArena 14,72 **[D]** | 1,5–2,8 GB VRAM **[D]** | ≈0,0007 (RTFx 1 340) **[D]** | OpenMDW-1.1 | Ny (2026-09-23), aktiv |
| Nemotron 3 Diarization BF16-GGUF (audio.cpp) | – | ≈NeMo (4–5 % DER-avvikelse på 120 s ström) **[D]** | 820–1 076 MiB **[D]** | ≈0,001 (RTX 5090) **[D]** | OpenMDW-1.1 + Apache-2.0 | Ny, snabbt föränderlig |
| Nemotron 3 Diarization Q8_0-GGUF | – | AMI IHM 9,23 (ref 9,22) **[D]**; **turer −5,6 % (161→152) / segment −6 %** **[D]** | 740–948 MiB **[D]** | ≈0,0009–0,0011 **[D]** | OpenMDW-1.1 | Ny; **ej turparitetssäker** |
| Nemotron 3 Diarization ONNX int8 | – | 99,994 % ramöverensstämmelse mot fp32 **[D]**; DER ej verifierat | ~103–120 MB modell; RAM ej verifierat | ≈0,04 (WASM, M3 Max) **[D]** | OpenMDW-1.1 | Community |
| pyannote community-1 | – | AMI IHM 17,0; DIHARD3 20,2; VoxConverse 11,2 **[D]**; VoiceArena 30,61 **[D]** | 1,59 GB (3.3.2) → 9,54 GB (4.0.3, 72 min) **[D]** | 31 s/h på H100 (≈0,009) **[D]** | CC-BY-4.0 (pipeline), MIT (kod) | Hög, men VRAM-regression |
| whisperX (KB-Whisper ct2 + voxrex + community-1) | = KB-Whisper | = community-1 | summa av delarna; batch 16 fp16 ≈ 6 GB+ **[D]** | ej verifierat för svenska | Apache-2.0 + CC0 + CC-BY-4.0 + BSD-2 | Hög |
| NeMo-Speech.cpp `transcribe --diarize` (Parakeet TDT v3 / Nemotron 3.5 ASR + Nemotron 3 Diar) | = Parakeet v3 (15,08 FLEURS) resp. ej verifierat | = Nemotron 3 | ej verifierat | ej verifierat | Apache-2.0 + CC-BY-4.0/OpenMDW | Ny |
| KB-Whisper + Nemotron 3 (egen limning) | = KB-Whisper | = Nemotron 3 | ≈ ASR + 1 GB **[I]** | ej verifierat | Apache-2.0 + OpenMDW-1.1 | Ingen färdig integration |

---

## 7. Rekommenderad provordning

Alla experiment: 16 kHz mono, samma ljudfiler, fasta versioner/commits loggade. DER rapporteras med collar 0 och overlap inkluderad enligt NVIDIA:s konvention ([diarization_evaluation.md](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/diarization_evaluation.md)). Tröskelvärdena är **[I]** och satta utifrån de publicerade paritetsdata ovan.

1. **Portparitet BF16-GGUF mot NeMo/Transformers-referens (Nemotron).** *Syfte:* separera portfel från kvantiseringsfel. *Variabel:* runtime (audio.cpp BF16 vs NeMo Python bf16), samma offline-geometri (chunk 340, rc 40, fifo 40, spkcache 264, update 300). *Förväntat:* nära identiska turer; publicerat 4–5 % DER-avvikelse på strömmar, lägre offline. *Grönt:* DER(GGUF vs NeMo) ≤ 2 % och turantal ±1 %. *Grått:* 2–6 % eller ±1–3 %. *Rött:* > 6 % eller > 3 % turavvikelse → uppgradera runtime/GGUF innan vidare tester.
2. **Kvantiseringsparitet Q8_0 mot BF16 (Nemotron) på ≥5 svenska filer, 5–15 min vardera.** *Syfte:* kvantifiera referensfyndet. *Variabel:* endast vikternas precision (samma runtime, backend, geometri). *Mått:* turantal, same-label-aktivitetsskillnad på 10 ms, DER med BF16 som referens, Missed/FA-split. *Förväntat:* DER-skillnad ≪ 1 % men 3–6 % färre turer, högre Missed. *Grönt:* turavvikelse ≤ 2 % och DER-diff ≤ 0,5 %. *Grått:* 2–6 % eller DER-diff 0,5–1,5 %. *Rött:* > 6 % eller DER-diff > 1,5 % → Q8_0 utesluts för produktion.
3. **Q8_0 mot BF16 på CPU-backend.** *Syfte:* audio.cpp-kortet varnar att Q8-resultaten "do not validate … other backends". *Variabel:* backend (CPU vs CUDA) för båda precisionerna. *Förväntat:* CPU F32/BF16 ≈ CUDA BF16; Q8_0-avvikelsen lika stor eller större på CPU. *Grönt/grått/rött:* som exp. 2.
4. **Svenskt facit: Nemotron BF16 mot pyannote community-1.** *Syfte:* bekräfta att VoiceArena-ordningen håller för svenska. *Variabel:* diariserare. *Facit:* RixVox-v2-härledda RTTM (debatter) + 5 handmärkta möten. *Förväntat:* Nemotron lägre DER, särskilt talarförväxling. *Grönt:* Nemotron ≤ pyannote − 3 % absolut. *Grått:* inom ±3 %. *Rött:* Nemotron > pyannote + 3 % → undersök språk-/domäneffekt.
5. **KB-Whisper precision: ct2 fp16 mot int8 (large).** *Syfte:* KBLab har inga int8-siffror. *Variabel:* `compute_type`. *Testset:* FLEURS sv test (publikt facit 5,4) + egna filer. *Förväntat:* ΔWER < 0,3. *Grönt:* ΔWER ≤ 0,3 absolut. *Grått:* 0,3–1,0. *Rött:* > 1,0 → fp16 obligatoriskt på GPU.
6. **KB-Whisper storlek på CPU-noden: small vs medium (whisper.cpp q5_0 och ct2 int8).** *Syfte:* välja CPU-modell. *Variabel:* modellstorlek (format hålls konstant per körning). *Förväntat:* medium ~0,7–1,3 WER-poäng bättre, RTF 2–3× högre. *Grönt:* medium RTF ≤ 0,5 på noden. *Grått:* 0,5–1,0. *Rött:* > 1,0 → small.
7. **Ordtidsstämplar: KB-Whisper egna tidsstämplar mot wav2vec2-voxrex-alignment (whisperX/easyaligner).** *Syfte:* mäta om alignment behövs för talartilldelning. *Variabel:* alignment på/av. *Mått:* andel ord med korrekt talare mot handmärkt facit, samt cpWER. *Grönt:* alignment ger ≥ 2 % fler korrekt tilldelade ord. *Grått:* 0–2 %. *Rött:* ingen vinst eller sämre → hoppa över alignment på CPU-noden.
8. **Hel kedja end-to-end: KB-Whisper (strict) + Nemotron BF16 mot whisperX (KB-Whisper + community-1).** *Syfte:* helhetsmått. *Variabel:* diariserare i kedjan. *Mått:* cpWER och DER på de handmärkta mötena; topp-VRAM och RTF per profil. *Grönt:* Nemotron-kedjan ≤ whisperX-kedjan i cpWER och topp-VRAM ≤ 5,0 GB på 6 GB-kortet. *Grått:* cpWER inom ±1 % eller VRAM 5,0–5,8 GB. *Rött:* OOM eller cpWER > whisperX + 1 %.
9. **Turparitet med `strict` mot `standard` KB-Whisper.** *Syfte:* mäta om verbatim-stilen ger bättre talartilldelning. *Variabel:* revision. *Grönt:* strict ≥ 1 % fler korrekt tilldelade ord utan WER-försämring > 0,3. *Grått:* neutralt. *Rött:* WER > +0,5.
10. **pyannote 4.0.7 VRAM-regressionstest** (endast 10 GB-kortet). *Variabel:* fillängd 10/30/60 min. *Grönt:* topp ≤ 3 GB vid 60 min. *Grått:* 3–8 GB. *Rött:* > 8 GB → pyannote utesluts på 6 GB-kortet.

---

## Källförteckning

Alla länkar lästa 2026-09-29.

1. KBLab, modellkort *KB-Whisper Large* — https://huggingface.co/KBLab/kb-whisper-large
2. Vesterbacka m.fl. (KBLab), *Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition*, Interspeech 2025 — https://arxiv.org/abs/2505.17538
3. KBLab, filträd kb-whisper-small — https://huggingface.co/KBLab/kb-whisper-small/tree/main
4. KBLab, filträd kb-whisper-medium — https://huggingface.co/KBLab/kb-whisper-medium/tree/main
5. KBLab, filträd kb-whisper-tiny — https://huggingface.co/KBLab/kb-whisper-tiny/tree/main
6. KBLab, filträd kb-whisper-base — https://huggingface.co/KBLab/kb-whisper-base/tree/main
7. KBLab, modellkort *Wav2vec 2.0 large VoxRex Swedish* — https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish
8. KBLab, easyaligner (GitHub) — https://github.com/kb-labb/easyaligner
9. KBLab, blogg *easyaligner: Forced alignment of text and audio, made easy* (2026-04-16) — https://huggingface.co/blog/KBLab/easyaligner
10. KBLab, easytranscriber (GitHub) — https://github.com/kb-labb/easytranscriber
11. KBLab, datasetkort *RixVox-v2* — https://huggingface.co/datasets/KBLab/rixvox-v2
12. KBLab, datasetkort *RixVox* (v1) — https://huggingface.co/datasets/KBLab/rixvox
13. KBLab, rixvox-pipeline (GitHub, arkiverat) — https://github.com/kb-labb/rixvox
14. OpenAI, modellkort whisper-large-v3 — https://huggingface.co/openai/whisper-large-v3
15. OpenAI, modellkort whisper-large-v3-turbo — https://huggingface.co/openai/whisper-large-v3-turbo
16. OpenAI, diskussion #2363 (large-v3-turbo) — https://github.com/openai/whisper/discussions/2363
17. SYSTRAN, faster-whisper README (benchmarks, licens) — https://github.com/SYSTRAN/faster-whisper
18. SYSTRAN, faster-whisper issue #1030 (turbo-benchmark) — https://github.com/SYSTRAN/faster-whisper/issues/1030
19. ggml-org, whisper.cpp README (minnestabell, kvantisering, tinydiarize) — https://github.com/ggml-org/whisper.cpp
20. m-bain, whisperX README — https://github.com/m-bain/whisperX
21. m-bain, whisperX `alignment.py` (svensk default-modell) — https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
22. m-bain, whisperX `pyproject.toml` — https://github.com/m-bain/whisperX/blob/main/pyproject.toml
23. NVIDIA, modellkort *Nemotron 3 Diarization* — https://huggingface.co/nvidia/Nemotron-3-Diarization
24. NVIDIA, filträd Nemotron-3-Diarization (inkl. q8_0.gguf) — https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/main
25. NVIDIA, *Diarization Evaluation* (subcard) — https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/diarization_evaluation.md
26. NVIDIA, *ASR Integration Guide* (subcard) — https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md
27. Medennikov m.fl., *Streaming Sortformer* — https://arxiv.org/abs/2507.18446
28. Park m.fl., *Sortformer* — https://arxiv.org/abs/2409.06656
29. OpenMDW License Agreement v1.1 — https://openmdw.ai/license/1-1/
30. OpenMDW, licensrepo (GitHub) — https://github.com/OpenMDW/openmdw
31. Linux Foundation, pressmeddelande OpenMDW-1.1 / NVIDIA — https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families
32. NVIDIA, NeMo-Speech.cpp README — https://github.com/NVIDIA/NeMo-Speech.cpp
33. NVIDIA, NeMo-Speech.cpp PR #50 (Nemotron 3 Diarization) — https://github.com/NVIDIA/NeMo-Speech.cpp/pull/50
34. NVIDIA, NeMo-Speech.cpp docs/cli.md — https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md
35. NVIDIA, NeMo-Speech.cpp docs/asr/models.md — https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/models.md
36. NVIDIA, NeMo Speech (Python-ramverk) — https://github.com/NVIDIA-NeMo/Speech
37. NVIDIA, modellkort nemotron-3.5-asr-streaming-0.6b — https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
38. NVIDIA, modellkort parakeet-tdt-0.6b-v3 — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
39. NVIDIA, modellkort canary-1b-v2 — https://huggingface.co/nvidia/canary-1b-v2
40. NVIDIA, modellkort diar_streaming_sortformer_4spk-v2.1 — https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
41. audio-cpp, *Nemotron-3-Diarization-GGUF* (paritets- och Q8-kaveat) — https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF
42. 0xShug0, audio.cpp README — https://github.com/0xShug0/audio.cpp
43. 0xShug0, audio.cpp LICENSE (Apache-2.0) — https://github.com/0xShug0/audio.cpp/blob/main/LICENSE
44. 0xShug0, audio.cpp PR #660 (Nemotron-stöd) — https://github.com/0xShug0/audio.cpp/pull/660
45. 0xShug0, audio.cpp PR #677 (NeMo-paritet, Q8_0-DER) — https://github.com/0xShug0/audio.cpp/pull/677
46. 0xShug0, audio.cpp PR #703 (GGUF-kompatibilitet) — https://github.com/0xShug0/audio.cpp/pull/703
47. 0xShug0, audio.cpp docs/models/nemotron_3_diar.md — https://github.com/0xShug0/audio.cpp/blob/main/docs/models/nemotron_3_diar.md
48. mudler, *Nemotron-3-Diarization-GGUF* för parakeet.cpp (Q8_0 −6 % segment) — https://huggingface.co/mudler/Nemotron-3-Diarization-GGUF
49. mudler, parakeet.cpp PR #71 — https://github.com/mudler/parakeet.cpp/pull/71
50. mudler, parakeet.cpp README — https://github.com/mudler/parakeet.cpp
51. Glimpse-Dictation, *Nemotron-3-Diarization-gguf* (AMI-DER per precision) — https://huggingface.co/Glimpse-Dictation/Nemotron-3-Diarization-gguf
52. onnx-community, *Nemotron-3-Diarization-ONNX* — https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX
53. NealCaren, *Nemotron-3-Diarization-ONNX* (int8-paritet) — https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX
54. s0undy, *nemotron-3-diarization-onnx-int8* (svenskt testset) — https://huggingface.co/s0undy/nemotron-3-diarization-onnx-int8
55. aufklarer, *Nemotron-3-Diarization-100M-MLX-INT8* — https://huggingface.co/aufklarer/Nemotron-3-Diarization-100M-MLX-INT8
56. mlx-community, *Nemotron-3-Diarization-8bit* — https://huggingface.co/mlx-community/Nemotron-3-Diarization-8bit
57. Bonenk, *Nemotron-3-Diarization-GGUF* (återger VoiceArena-tabellen) — https://huggingface.co/Bonenk/Nemotron-3-Diarization-GGUF
58. VoiceArena, Diarization-Bench — https://voicearena.com/diarization-bench
59. pyannote, modellkort *speaker-diarization-community-1* — https://huggingface.co/pyannote/speaker-diarization-community-1
60. pyannote, pyannote-audio README (MIT, hastighet) — https://github.com/pyannote/pyannote-audio
61. pyannote, issue #1963 (VRAM-regression 4.0.3) — https://github.com/pyannote/pyannote-audio/issues/1963
62. pyannote, releases (4.0.0–4.0.7) — https://github.com/pyannote/pyannote-audio/releases
63. Qwen, modellkort Qwen3-ASR-1.7B — https://huggingface.co/Qwen/Qwen3-ASR-1.7B
64. Mistral, modellkort Voxtral-Mini-3B-2507 — https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
65. Kyutai, modellkort stt-2.6b-en — https://huggingface.co/kyutai/stt-2.6b-en
66. Nord-Parl-TTS (arXiv, endast abstrakt) — https://arxiv.org/abs/2509.17988
67. Hub-sökning "kb-whisper" (community-ct2/MLX/CoreML-speglar) — https://huggingface.co/models?search=kb-whisper
