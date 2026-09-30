# Varför pyannote community-1 går långsamt på en RTX 3080 – och hur man når 5–10× realtid

*Webbaserad teknisk utredning. Alla källor lästa 2026-09-29. Versioner som nämns är de som var aktuella det datumet (pyannote.audio 4.0.7, Nemotron 3 Diarization släppt 2026-09-23).*

## Sammanfattning

Referensmätningen (1,2–3,1× realtid på RTX 3080) ligger en storleksordning under pyannotes egna ~31–37 s/timme (≈100×) för community-1, angivet utan hårdvara [[27]](https://github.com/pyannote/pyannote-audio) [[35]](https://www.pyannote.ai/blog/community-1). Troligaste orsaker:
1. **Stor fast overhead per anrop** – skillnaden mellan de två klippen antyder (slutledning, §1.9) ca 100 s fast kostnad utöver modelladdningen.
2. **Redundant embedding‑beräkning** (10 s fönster/1 s steg, en embedding per chunk × talare = ~21× ljudvolymen) [[12]](https://arxiv.org/abs/2507.16136); pyannotes egen "fast path"‑PR visar 52 s → ~10 s på H100 [[10]](https://github.com/pyannote/pyannote-audio/pull/1996).
3. **TF32 stängs av vid varje anrop och ingen fp16/autocast finns** – tensorkärnorna används inte [[6]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/reproducibility.py) [[4]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/inference.py).
4. **CPU‑bundna steg** (VBx/AHC i numpy/scipy, dekodning, Python‑loopar) [[5]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/clustering.py) [[15]](https://github.com/pyannote/pyannote-audio/issues/1453), samt möjligt miljöfel (pipeline i praktiken på CPU; ett ärende med två RTX 3080 visar 0 % GPU‑last) [[21]](https://github.com/pyannote/pyannote-audio/issues/1702).

Störst hävstång: (a) mät varmt, med ljud i minnet och verifiera `cuda`‑placering [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1); (b) `segmentation_step` 0,1→0,2–0,4 (15–38× vid stride 4, DER +0,01–0,06) [[12]](https://arxiv.org/abs/2507.16136) plus fp16 på embedding enligt PR #1996 [[10]](https://github.com/pyannote/pyannote-audio/pull/1996); (c) byt motor till Nemotron 3 Diarization (OpenMDW‑1.1, 100 M param.): 1 340–15 113× realtid på GPU offline [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization), 55–90× på laptop‑CPU via ONNX int8 [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX).

---

## 1. Varför pyannote är långsamt på GPU

Legend: **[D]** = dokumenterat/mätt av andra (med källa). **[I]** = min slutledning.

### 1.1 Pipelinens steg och standardvärden

community-1 består av powerset‑segmentering, WeSpeaker‑embeddings och VBx‑klustring [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1). Pipelinens `config.yaml` sätter `clustering: VBxClustering`, `segmentation_batch_size: 32`, `embedding_batch_size: 32`, `embedding_exclude_overlap: true` samt VBx‑parametrarna `threshold: 0.6`, `Fa: 0.07`, `Fb: 0.8` [[2]](https://huggingface.co/pyannote/speaker-diarization-community-1/blob/main/config.yaml) **[D]**. 3.1 använde i stället `AgglomerativeClustering` (centroid, `min_cluster_size: 12`, tröskel ≈0,7046) med samma batchstorlekar [[29]](https://huggingface.co/pyannote/speaker-diarization-3.1) **[D]**.

I koden är klassens egna defaultvärden `segmentation_batch_size=1`, `embedding_batch_size=1`, `segmentation_step=0.1` (steget uttryckt som andel av fönsterlängden) [[3]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py) **[D]**. Ändringen till default 1 infördes i 3.0.0 [[23]](https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md). Den som instansierar pipelinen utan `config.yaml` får alltså batch 1 **[I: relevant bara om man bygger pipelinen manuellt]**.

### 1.2 Glidande fönster i segmenteringen

Segmenteringsmodellen tar 10 s mono‑ljud och ger 7 powerset‑klasser (upp till 3 samtidiga talare) [[30]](https://huggingface.co/pyannote/segmentation-3.0) **[D]**. Med `segmentation_step=0.1` blir steget 1 s, dvs. varje sekund ljud passerar nätet 10 gånger; SDBench beskriver just "10‑second window size and 1‑second stride" som källa till "considerable computational overhead" [[12]](https://arxiv.org/abs/2507.16136) **[D]**. `Inference` bygger chunkarna med `unfold`, batchar dem (`batch_size` default 32 i klassen) och kör i `torch.inference_mode()`; ingen autocast/fp16 finns [[4]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/inference.py) **[D]**.

### 1.3 Embedding‑steget (WeSpeaker ResNet34) – den största kostnaden

Embeddingar beräknas per (chunk, talare) med masker; SDBench räknar ut att pyannote 3.1 kör embedding‑nätet på 630 s ljud för ett 30 s‑klipp, ett 21× overhead, eftersom nätet körs för var och en av 3 talare i varje 10 s‑fönster oavsett om talaren är aktiv [[12]](https://arxiv.org/abs/2507.16136) **[D]**. Ett GitHub‑ärende beskriver samma sak: varje chunk körs "3 times" genom embedding‑modellen och en delning i ResNet‑backbone + maskpooling skulle ge "almost 3x speedup"; ärendet fick etiketten *wontfix* [[11]](https://github.com/pyannote/pyannote-audio/issues/1634) **[D]**.

WeSpeaker‑wrappern beräknar fbank med `torch.vmap` på GPU (fallback till CPU endast för MPS) och exponerar redan en delad API `forward_frames`/`forward_embedding` [[33]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/embedding/wespeaker/__init__.py) **[D]**. PR #1996 använder den: alla chunk‑vågformer skivas en gång i stället för `audio.crop()` per chunk, (chunk, talare)‑par med tom mask hoppas över ("~60 % fewer embedding calls"), `forward_frames` körs en gång per chunk, och en ny `embedding_precision=torch.float16` kör autocast på `forward_frames` med pooling i fp32. Uppmätt på H100, 57 min/10 talare: 52 s original → 20 s fast path → ~10 s med fp16; `forward_frames` 12 s → 3 s [[10]](https://github.com/pyannote/pyannote-audio/pull/1996) **[D]**. PR:n är **inte** mergad (öppen, märkt *wontfix* 2026‑09‑22) och aktiveras bara för in‑memory‑ljud **[D]**.

Batchstorlek hjälper inte automatiskt: med `wespeaker-voxceleb-resnet34-LM` gav batch 32/64/128 71/69/70 s trots högre GPU‑last [[17]](https://github.com/pyannote/pyannote-audio/issues/1566) **[D]**, och underhållaren själv noterade att batch 32 var långsammare än batch 1 på Mac [[18]](https://github.com/pyannote/pyannote-audio/issues/1195) **[D]**. **[I]** Det tyder på att flaskhalsen ofta ligger utanför GPU‑kärnorna (Python‑loop, maskhantering, dataflytt), inte i beräkningsvolymen.

### 1.4 Klustring – CPU‑bunden

`clustering.py` implementerar AHC med `scipy.cluster.hierarchy.linkage/fcluster`, KMeans via scikit‑learn, samt VBx som initieras med AHC och kör `cluster_vbx(..., maxIters=20)` efter en PLDA‑transform; allt i numpy/scipy på CPU, ingen GPU‑väg [[5]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/clustering.py) **[D]**. Användare med långa filer rapporterar att CPU ligger på ~100 % medan GPU vilar (10 min ljud på 12 min på en V100, dvs. 50× under uppgivet riktmärke) [[15]](https://github.com/pyannote/pyannote-audio/issues/1453) och att 2 h ljud tog ~1 h med "GPU under very light load" [[16]](https://github.com/pyannote/pyannote-audio/issues/1403) **[D]**. **[I]** För referensmätningens 2,5–9 min klipp är antalet embeddingar litet (≈150–550 chunkar × ≤3 talare), så klustringen bör ta sekunder, inte minuter – den förklarar inte huvuddelen av gapet men blir viktig för timmeslånga filer.

### 1.5 Torch‑inställningar: TF32, fp16, cudnn.benchmark

`fix_reproducibility()` sätter `torch.backends.cuda.matmul.allow_tf32 = False` och `torch.backends.cudnn.allow_tf32 = False` med hänvisning till reproducerbarhetsproblem (issue #1370, där byte av CUDA‑version gav 12 % relativ DER‑skillnad) [[6]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/reproducibility.py) [[8]](https://github.com/pyannote/pyannote-audio/issues/1370) **[D]**, och den anropas i början av varje `Pipeline.__call__` [[7]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/pipeline.py) **[D]**. PyTorch har `matmul.allow_tf32=False` som default sedan 1.12 men `cudnn.allow_tf32=True` [[9]](https://docs.pytorch.org/docs/stable/notes/cuda.html) **[D]** – pyannote stänger alltså av även cuDNN‑TF32 för faltningarna i ResNet34. **[I]** På Ampere (RTX 3080) innebär det ren fp32 utan tensorkärnor för hela pipelinen. Att sätta flaggorna före `pipeline(...)` hjälper inte eftersom de återställs i `__call__`; man måste antingen patcha funktionen eller anropa `pipeline.apply(...)` direkt (behöver verifieras lokalt). `cudnn.benchmark` sätts inte av pyannote **[D, frånvaro i koden]**; effekten är omätt för denna pipeline **[ej verifierat]**.

### 1.6 Ljudinläsning och CPU↔GPU‑överföring

4.0.0 bytte från `torchaudio` till `torchcodec` (ffmpeg krävs) och tog bort sox/soundfile‑backend; bara ffmpeg eller in‑memory‑ljud stöds [[23]](https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md) **[D]**. 4.0.2 pinnade torch/torchcodec/torchaudio "to avoid segmentation fault" [[23]](https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md). Modellkortet rekommenderar att ladda ljudet i minnet ("may result in faster processing") [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1) **[D]**, men i 3.1.1 rapporterades det motsatta (tensor/BytesIO ~1,7× långsammare än sökväg, på CPU) [[20]](https://github.com/pyannote/pyannote-audio/issues/1692) **[D]** – utfallet är versions‑ och plattformsberoende och måste mätas. Den legacy‑väg som körs för filbaserat ljud gör `self._audio.crop(file, chunk, mode="pad")` per chunk [[3]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py) **[D]**; **[I]** det är ett stort antal små CPU‑operationer och host→device‑kopior.

### 1.7 Hook/progress

`apply()` anropar `hook("segmentation"…)`, `hook("speaker_counting"…)`, `hook("embeddings"…)`, `hook("discrete_diarization"…)` [[3]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py) **[D]**. Ingen källa kvantifierar hook‑kostnaden **[ej verifierat]**; **[I]** den är sannolikt försumbar men hooken är samtidigt det enklaste sättet att tidsätta stegen i bänkplanen.

### 1.8 ONNX/TensorRT

Segmentation‑3.0 exporteras problemfritt till ONNX (t.ex. [[60]](https://github.com/pengzhendong/pyannote-onnx), Apache‑2.0, och sherpa‑onnx' paket [[58]](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)) **[D]**. 3.0 använde onnxruntime för embeddingar, vilket orsakade CPU‑körning när `onnxruntime` (ej `-gpu`) låg kvar; 3.0.1 bytte till `onnxruntime-gpu` och 3.1 tog bort onnxruntime helt till förmån för ren PyTorch [[53]](https://github.com/m-bain/whisperX/issues/499) [[23]](https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md) **[D]**. En studie av DIART‑pipelinen fann enligt sin sammanfattning att ONNX‑konvertering försämrade latensen ~40 % medan destillation/kvantisering hjälpte [[62]](https://arxiv.org/abs/2408.02341) **[D, endast abstract läst]**. Ingen TensorRT‑export av community‑1 kunde verifieras **[ej verifierat]**.

### 1.9 Vad referensmätningen själv säger **[I]**

Om 2,5 min (150 s) tar ~125 s (1,2×) och 9,1 min (546 s) tar ~176 s (3,1×), är den marginella kostnaden ≈51 s för 396 s extra ljud (≈7,8× realtid) och den fasta kostnaden ≈105 s – långt mer än de 9 s modell­laddningen. Kandidater: första CUDA‑anropets kernel‑/cuDNN‑initiering, `torch.vmap`‑fbank vid första körning, ffmpeg‑dekodning via torchcodec, PLDA/VBx‑initiering, eller att någon del faktiskt körs på CPU. Bänkplanens uppvärmningsregel (§4) är utformad för att skilja dessa.

---

## 2. Inställningar som dokumenterat ger 5–10× realtid

| Åtgärd | Rapporterad effekt | Hårdvara / källa | Kommentar |
|---|---|---|---|
| Kör pipelinen på GPU (`pipeline.to(torch.device("cuda"))`) | 45 min video: 24 → 3,3 min (3.0.1→3.1.1, MPS) | Mac GPU [[19]](https://github.com/pyannote/pyannote-audio/issues/1626) | Grundkrav; kontrollera att parametrarna verkligen ligger på `cuda`. |
| Ljud i minnet (`{"waveform","sample_rate"}`) | "may result in faster processing"; krav för fast path i PR #1996 | [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1) [[10]](https://github.com/pyannote/pyannote-audio/pull/1996) | Motsatt resultat rapporterat på CPU i 3.1.1 [[20]](https://github.com/pyannote/pyannote-audio/issues/1692). |
| Större `segmentation_step` (stride 2 s / 4 s) | 15,3–38,3× snabbare vid stride 4; DER +0,01 (AMI‑IHM) till +0,06 (VoxConverse) | M2 Ultra, fp16, pyannote 3.1 [[12]](https://arxiv.org/abs/2507.16136) | Störst hävstång; störst DER‑risk i "in the wild"‑ljud. Underhållaren rekommenderade redan 2021 att öka steget [[22]](https://github.com/pyannote/pyannote-audio/discussions/778). |
| Per‑chunk embedding (en backbone‑körning per chunk) | 1,2× i SpeakerKit; "may be even higher in Pyannote v3.1" | [[12]](https://arxiv.org/abs/2507.16136) | Motsvarar `forward_frames`‑delningen i PR #1996. |
| Fast path + `embedding_precision=torch.float16` | 52 s → 20 s → ~10 s för 57 min (≈5×) | H100 [[10]](https://github.com/pyannote/pyannote-audio/pull/1996) | Ej mergad; kräver patch. DER ej rapporterad. |
| Stride 3 s + per‑chunk embedding + relativ `min_cluster_size` (f=0,01) | upp till 12,2× snabbare på AMI; VoxConverse‑DER återställd 0,113→0,079 | Apple MPS, CAM++‑baserad pipeline [[14]](https://arxiv.org/abs/2606.08505) | Visar att `min_cluster_size` måste skalas om när antalet embeddingar minskar. |
| Tillåt TF32 | pyannote: "can be re‑enabled"; PyTorch beskriver TF32 som snabbare fp32‑GEMM på Ampere | [[6]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/reproducibility.py) [[9]](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Ingen publicerad pyannote‑mätning **[ej verifierat]**; mät själv (§4). |
| Ange `num_speakers`/`min/max_speakers` | Stöds av API:t | [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1) | Effekt på tid ej dokumenterad **[ej verifierat]**. |

**Ärlig sammanfattning:** de enda publicerade mätningarna med ≥5× vinst kommer från datacenter‑GPU (H100) eller Apple‑Silicon; ingen källa dokumenterar RTX 3080 specifikt. **[I]** Kombinationen stride 0,2–0,4 + fp16 på embedding + in‑memory‑ljud bör dock ge >5× på en 3080 om GPU:n idag inte är flaskhalsen, vilket bänkplanen testar.

---

## 3. Alternativ och licenser

| System | Typ | Licens (kod / vikter) | Hastighet (dokumenterad) | Noteringar |
|---|---|---|---|---|
| pyannote.audio 3.1 (`speaker-diarization-3.1`) | segmentering + WeSpeaker + AHC | Kod MIT [[28]](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE); pipeline MIT [[29]](https://huggingface.co/pyannote/speaker-diarization-3.1); WeSpeaker‑vikter CC‑BY‑4.0 [[31]](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) [[32]](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) | RTF 45× i genomsnitt på RTX A6000 (out‑of‑the‑box) [[13]](https://arxiv.org/abs/2509.26177) | Gated på HF (acceptera villkor + token). |
| pyannote.audio 4.0.x / community‑1 | segmentering + WeSpeaker + VBx, "exclusive" utdata | Kod MIT; pipeline CC‑BY‑4.0 [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1) | ~31 s/h (AMI‑IHM), 37 s/h (DIHARD 3); hårdvara ej angiven [[27]](https://github.com/pyannote/pyannote-audio) [[35]](https://www.pyannote.ai/blog/community-1) | Kräver Python ≥3.10, torch ≥2.8, torchcodec ≥0.7, ffmpeg [[25]](https://github.com/pyannote/pyannote-audio/blob/develop/pyproject.toml). Senaste 4.0.7 (2026‑06‑30) [[26]](https://pypi.org/project/pyannote-audio/). |
| pyannoteAI precision‑2/‑3 | kommersiell moln‑API (self‑hosting endast enterprise) | Proprietär; HF‑kortet är bara ett API‑skal [[34]](https://huggingface.co/pyannote/speaker-diarization-precision-2) | 14 s/h, "2,6× snabbare" än community på DIHARD 3 [[27]](https://github.com/pyannote/pyannote-audio); lägst DER i två oberoende benchmark [[13]](https://arxiv.org/abs/2509.26177) [[12]](https://arxiv.org/abs/2507.16136) | Prissättning ca €0,10/timme [[36]](https://www.pyannote.ai/pricing) (sida ej läst direkt, endast sökträff). |
| **NVIDIA Nemotron 3 Diarization** | end‑to‑end streaming Sortformer, 31‑lagers Transformer med RoPE, 100 M param., upp till 8 talare | **OpenMDW‑1.1** [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization) [[38]](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1) – permissiv, kommersiellt OK, kräver bara att licens och notiser följer med; patent‑terminering vid stämning | GPU (RTX PRO 5000, BF16): RTFx 1 340 (eager, bs 1) – 15 113 (compiled, bs 32) i 30,4 s‑offline‑läge; 12,5–292× i 0,32 s‑läge [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization). CPU: ONNX int8 RTF 0,011 (≈90×) vid 4 trådar på i9‑13900HX; PyTorch‑referens RTF 0,054 (≈18×) vid 8 trådar [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX) | NeMo‑kod Apache‑2.0 [[40]](https://github.com/NVIDIA-NeMo/NeMo/blob/main/LICENSE) [[41]](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE). VRAM: ~1,0–1,6 GiB uppmätt i C++/Python [[44]](https://github.com/0xShug0/audio.cpp/pull/660) [[43]](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF). DER (full): 12,73 % vs 19,09 % för Streaming Sortformer på NVIDIAs egen svit [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization). Portar: officiell NeMo‑Speech.cpp (ggml, CUDA/Metal/Vulkan/CPU, Apache‑2.0) [[42]](https://github.com/NVIDIA/NeMo-Speech.cpp); GGUF q8_0 i modellrepot och via audio.cpp [[43]](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF); ONNX [[45]](https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX) [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX); Transformers‑stöd [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization). Referensmätningens "~35× på CPU" ligger inom det spann andra rapporterar (18–90× beroende på runtime/precision) **[I]**. |
| NeMo Streaming Sortformer 4spk‑v2 | föregångare, 117 M param., 17‑lagers FastConformer, max 4 talare | CC‑BY‑4.0 [[39]](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | RTFx 874–3 204 (offline) på samma testbänk [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization); 214× i oberoende benchmark på A6000 [[13]](https://arxiv.org/abs/2509.26177) | Ersatt av Nemotron 3 enligt modellkortet. |
| diart | streaming‑diarisering ovanpå pyannote‑modeller | MIT [[49]](https://github.com/juanmc2005/diart) | Segmentering 8 ms, embedding 12 ms per steg på RTX 4060; latens 0,5–5 s | Rekommenderar `pyannote.audio<3.1`; inte för batch‑bearbetning av filer. |
| whisperX | faster‑whisper + pyannote community‑1 | BSD‑2‑Clause [[50]](https://github.com/m-bain/whisperX) | ASR "70× realtime" (large‑v2); diariseringen är samma pyannote‑pipeline | `DiarizationPipeline` laddar `pyannote/speaker-diarization-community-1`, kör `.to(device)` och skickar ljudet som in‑memory‑dict [[51]](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py); pins `pyannote-audio>=4.0.0`, `torch~=2.8.0`, `faster-whisper>=1.2.0` [[52]](https://github.com/m-bain/whisperX/blob/main/pyproject.toml). Historiska klagomål (1,5 h ljud >1 h på RTX 3090) [[54]](https://github.com/m-bain/whisperX/issues/274). Inga egna diariserings‑optimeringar. |
| DiariZen | WavLM‑Large (prunad 80 %) + Conformer EEND + pyannote‑klustring | Kod MIT; vikter **CC‑BY‑NC‑4.0** [[55]](https://github.com/BUTSpeechFIT/DiariZen) [[56]](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) | Ingen RTF publicerad; DER 13,3 % vs pyannoteAI 11,2 % [[13]](https://arxiv.org/abs/2509.26177) | Icke‑kommersiell licens på vikterna. |
| sherpa‑onnx | ONNX‑runtime‑pipeline: pyannote segmentation‑3.0 + 3D‑Speaker‑embedding + klustring | Apache‑2.0 [[57]](https://github.com/k2-fsa/sherpa-onnx); 3D‑Speaker Apache‑2.0 [[59]](https://github.com/modelscope/3D-Speaker) | Ej dokumenterad hastighet; en användare bytte från sherpa‑onnx till Nemotron och gick från 123 s till 35 s för 78 min på CPU [[47]](https://github.com/jankeesvw/omarchy-meeting-recorder/pull/1) | CPU/embedded‑fokus. |

---

## 4. Bänkplan för en RTX 3080 (10 GB)

### 4.1 Testklipp (fasta, 16 kHz mono WAV)

- **K1** = referensmätningens 2,5 min‑klipp, **K2** = 9,1 min‑klippet (jämförbarhet bakåt).
- **K3** = AMI‑mötet ES2004a (17,5 min) och **K4** = ett ~50 min AMI‑möte, med forced‑alignment‑referens (CC‑BY‑4.0) – samma protokoll som NVIDIA och ONNX‑porten använder, vilket ger jämförbar DER [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX) [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization).
- **K0** = 30 s tyst/brus‑klipp enbart för uppvärmning.

### 4.2 Regler

1. **Uppvärmning:** ladda modellen, kör K0 en gång, kasta resultatet. Rapportera K1 både *kall* (första riktiga anropet efter processtart) och *varm* (median av 3 efterföljande körningar). Skillnaden kall−varm är den "fasta overheaden" i §1.9.
2. **En variabel per körning.** Allt annat = baslinje. Fast seed.
3. **Loggning per körning:** väggtid totalt och per steg via `hook` (segmentation / embeddings / discrete_diarization) [[3]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py); **RTFx** = ljudlängd / väggtid (samma definition som NVIDIA [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization)); `torch.cuda.max_memory_allocated()` **och** `max_memory_reserved()` (allokatorns reservation är det som syns i `nvidia-smi`); CPU‑last (medel, `psutil`); antal detekterade talare; **DER** med `pyannote.metrics`, collar 0 och 0,25 s, overlap inräknat (som [[13]](https://arxiv.org/abs/2509.26177)); för K1/K2 utan referens: DER mot baslinjens egen utdata ("självkonsistens‑DER").
4. **Sanity‑check före allt:** `next(pipeline._segmentation.model.parameters()).device == cuda:0` och `torch.backends.cuda.matmul.allow_tf32` loggat *inuti* ett hook‑anrop (då har `fix_reproducibility` redan körts).

### 4.3 Miljö (pins där källor finns)

- Baslinje: `pyannote.audio==4.0.7` [[26]](https://pypi.org/project/pyannote-audio/), `torch>=2.8.0`, `torchaudio>=2.8.0`, `torchcodec>=0.7.0` [[25]](https://github.com/pyannote/pyannote-audio/blob/develop/pyproject.toml), CUDA‑build av torch, `ffmpeg` installerat [[27]](https://github.com/pyannote/pyannote-audio). Python 3.10–3.12.
- whisperX‑spår: `whisperx` med `torch~=2.8.0`, `pyannote-audio>=4.0.0`, `faster-whisper>=1.2.0`, `ctranslate2>=4.5.0` [[52]](https://github.com/m-bain/whisperX/blob/main/pyproject.toml).
- Nemotron GPU‑spår: Python ≥3.12, `Cython packaging`, `nemo-toolkit[asr]`, `libsndfile1 ffmpeg` [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization); alternativt `transformers` från git enligt modellkortet.
- Nemotron CPU‑spår: `numpy onnxruntime soundfile` + `model.int8.onnx` [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX), och/eller NeMo‑Speech.cpp / audio.cpp med `q8_0.gguf` [[42]](https://github.com/NVIDIA/NeMo-Speech.cpp) [[43]](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF).

### 4.4 Planerade körningar och utfallsband

Baslinje **B** = community‑1, `config.yaml`‑defaults, ljud via filsökväg, varm. Alla "speedup" nedan är RTFx(run)/RTFx(B) på K2 (och K4 där rimligt); DER‑Δ i absoluta procentenheter på K3/K4.

| # | Variabel som ändras | Grönt | Grått | Rött |
|---|---|---|---|---|
| B | – (baslinje, kall + varm) | varm RTFx ≥ 10× **eller** kall−varm ≤ 15 s | varm 3–10× | varm < 3× (bekräftar referensmätningen; gå direkt till sanity‑check i 4.2 p.4) |
| R1 | Ljud i minnet (`{"waveform","sample_rate"}`) [[1]](https://huggingface.co/pyannote/speaker-diarization-community-1) | ≥ +20 % | −10…+20 % | < −10 % (jfr [[20]](https://github.com/pyannote/pyannote-audio/issues/1692)) |
| R2 | TF32 på (patcha `fix_reproducibility` eller anropa `apply` direkt) [[6]](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/reproducibility.py) | ≥ +25 % och DER‑Δ ≤ 0,3 | +5…25 % | < +5 % eller DER‑Δ > 0,3 |
| R3 | `segmentation_batch_size`/`embedding_batch_size` 32 → 64 → 128 | ≥ +20 % utan OOM, `max_memory_reserved` < 9 GB | +5…20 % | OOM eller < +5 % (jfr [[17]](https://github.com/pyannote/pyannote-audio/issues/1566)) |
| R4a | `segmentation_step` 0,1 → 0,2 (2 s) | ≥ 1,6× och DER‑Δ ≤ 0,5 | 1,2–1,6× eller DER‑Δ 0,5–1,5 | < 1,2× eller DER‑Δ > 1,5 |
| R4b | `segmentation_step` 0,1 → 0,4 (4 s) | ≥ 2,5× och DER‑Δ ≤ 1,0 (SDBench: 0,01–0,06 på deras datamängder [[12]](https://arxiv.org/abs/2507.16136)) | 1,5–2,5× eller DER‑Δ 1,0–3,0 | < 1,5× eller DER‑Δ > 3,0 eller talarantal fel på K4 (jfr [[14]](https://arxiv.org/abs/2606.08505)) |
| R5 | fp16‑autocast på embedding (patch enligt PR #1996 eller lokal `torch.autocast` runt `forward_frames`) [[10]](https://github.com/pyannote/pyannote-audio/pull/1996) | embedding‑steget ≥ 2× snabbare och DER‑Δ ≤ 0,3 | 1,3–2× | < 1,3× eller NaN/DER‑Δ > 0,3 |
| R6 | `clustering=AgglomerativeClustering` i stället för VBx (3.1‑parametrar [[29]](https://huggingface.co/pyannote/speaker-diarization-3.1)) | klustringstid −50 % och DER‑Δ ≤ 0,5 | – | DER‑Δ > 1,0 (behåll VBx) |
| R7 | `num_speakers` känt | ≥ +15 % | 0–15 % | – (informativt) |
| R8 | **Kombination** av alla gröna (R1–R7) | **RTFx ≥ 10× varm på K2 och K4, DER‑Δ ≤ 1,0** (målet 5–10× realtid uppnått med marginal) | RTFx 5–10× | < 5× → byt motor (R9/R10) |
| R9 | Nemotron 3 Diarization, NeMo, GPU, bf16, offline (chunk 340/rc 40) [[37]](https://huggingface.co/nvidia/Nemotron-3-Diarization) | RTFx ≥ 100×, VRAM < 3 GB, DER inom +2,0 av B på K3/K4 | RTFx 20–100× | < 20× (misstänk CPU‑fallback) eller DER > B + 5 |
| R10 | Nemotron 3 ONNX int8, CPU, 4 och 8 trådar [[46]](https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX) | RTFx ≥ 20× (porten mäter 55–125× på i9) och DER‑Δ mot R9 ≤ 0,5 | 5–20× | < 5× |
| R11 | whisperX (`--diarize`, community‑1) – enbart diariseringssteget tidsatt [[51]](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py) | inom ±15 % av R1 (bekräftar att whisperX inte tillför overhead) | – | > 30 % långsammare än R1 |

**Tolkningsregel:** blir B rött *och* R1–R7 alla gråa, är GPU:n inte flaskhalsen – felsök miljön (torch‑build, `.to(cuda)`, Windows‑drivrutin; jfr [[21]](https://github.com/pyannote/pyannote-audio/issues/1702)) innan fler parametrar testas.

---

## 5. Källförteckning

1. pyannote/speaker-diarization-community-1 (modellkort) – https://huggingface.co/pyannote/speaker-diarization-community-1
2. community-1 `config.yaml` – https://huggingface.co/pyannote/speaker-diarization-community-1/blob/main/config.yaml
3. pyannote.audio `pipelines/speaker_diarization.py` (develop) – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py
4. pyannote.audio `core/inference.py` – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/inference.py
5. pyannote.audio `pipelines/clustering.py` – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/clustering.py
6. pyannote.audio `utils/reproducibility.py` – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/reproducibility.py
7. pyannote.audio `core/pipeline.py` – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/pipeline.py
8. Issue #1370 "output variability with different CUDA versions" – https://github.com/pyannote/pyannote-audio/issues/1370
9. PyTorch CUDA semantics (TF32) – https://docs.pytorch.org/docs/stable/notes/cuda.html
10. PR #1996 "optimize embedding extraction in SpeakerDiarization" – https://github.com/pyannote/pyannote-audio/pull/1996
11. Issue #1634 (ResNet‑backbone/maskpooling, 3× redundans) – https://github.com/pyannote/pyannote-audio/issues/1634
12. SDBench (arXiv 2507.16136) – https://arxiv.org/abs/2507.16136
13. Benchmarking Diarization Models (arXiv 2509.26177) – https://arxiv.org/abs/2509.26177
14. Fast and Robust On‑Device Speaker Diarization (arXiv 2606.08505) – https://arxiv.org/abs/2606.08505
15. Issue #1453 "~50x slower than stated benchmark" – https://github.com/pyannote/pyannote-audio/issues/1453
16. Issue #1403 "Very low GPU usage (5%)" – https://github.com/pyannote/pyannote-audio/issues/1403
17. Issue #1566 "Inconsistent inference speed vs GPU utilization with batch size" – https://github.com/pyannote/pyannote-audio/issues/1566
18. Issue #1195 "large embedding batch size makes things slower" – https://github.com/pyannote/pyannote-audio/issues/1195
19. Issue #1626 "3.1.1 slower than 3.0 on CPU" – https://github.com/pyannote/pyannote-audio/issues/1626
20. Issue #1692 "Audio input as tensor or BytesIO is unexpectedly slow" – https://github.com/pyannote/pyannote-audio/issues/1692
21. Issue #1702 "Why is pyannote not using my GPU" (2× RTX 3080) – https://github.com/pyannote/pyannote-audio/issues/1702
22. Discussion #778 "Increasing speed of Speaker Diarization pipeline with CPU" – https://github.com/pyannote/pyannote-audio/discussions/778
23. pyannote.audio CHANGELOG – https://github.com/pyannote/pyannote-audio/blob/develop/CHANGELOG.md
24. pyannote.audio Releases – https://github.com/pyannote/pyannote-audio/releases
25. pyannote.audio `pyproject.toml` – https://github.com/pyannote/pyannote-audio/blob/develop/pyproject.toml
26. pyannote-audio på PyPI – https://pypi.org/project/pyannote-audio/
27. pyannote.audio README – https://github.com/pyannote/pyannote-audio
28. pyannote.audio LICENSE (MIT) – https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE
29. pyannote/speaker-diarization-3.1 (modellkort + config) – https://huggingface.co/pyannote/speaker-diarization-3.1
30. pyannote/segmentation-3.0 – https://huggingface.co/pyannote/segmentation-3.0
31. pyannote/wespeaker-voxceleb-resnet34-LM – https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
32. WeSpeaker pretrained‑licens – https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
33. pyannote.audio WeSpeaker‑wrapper – https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/embedding/wespeaker/__init__.py
34. pyannote/speaker-diarization-precision-2 – https://huggingface.co/pyannote/speaker-diarization-precision-2
35. pyannoteAI blogg "Community‑1" (domän blockerad vid läsning; siffror via sökträff och README [27]) – https://www.pyannote.ai/blog/community-1
36. pyannoteAI pricing (ej läst direkt) – https://www.pyannote.ai/pricing
37. nvidia/Nemotron-3-Diarization (modellkort) – https://huggingface.co/nvidia/Nemotron-3-Diarization
38. OpenMDW License 1.1 (text) – https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1
39. nvidia/diar_streaming_sortformer_4spk-v2 – https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
40. NeMo LICENSE (Apache‑2.0) – https://github.com/NVIDIA-NeMo/NeMo/blob/main/LICENSE
41. NeMo Speech LICENSE (Apache‑2.0) – https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE
42. NeMo‑Speech.cpp – https://github.com/NVIDIA/NeMo-Speech.cpp
43. audio-cpp/Nemotron-3-Diarization-GGUF – https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF
44. audio.cpp PR #660 (Nemotron 3 stöd, mätningar) – https://github.com/0xShug0/audio.cpp/pull/660
45. onnx-community/Nemotron-3-Diarization-ONNX – https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX
46. joosthel/Nemotron-3-Diarization-ONNX (CPU‑RTF, DER‑paritet) – https://huggingface.co/joosthel/Nemotron-3-Diarization-ONNX
47. omarchy-meeting-recorder PR #1 (CPU‑mätning Nemotron vs sherpa‑onnx) – https://github.com/jankeesvw/omarchy-meeting-recorder/pull/1
48. humla issue #193 (Nemotron vs Sortformer på Apple Silicon) – https://github.com/michaelwilhelmsen/humla/issues/193
49. diart – https://github.com/juanmc2005/diart
50. whisperX – https://github.com/m-bain/whisperX
51. whisperX `diarize.py` – https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py
52. whisperX `pyproject.toml` – https://github.com/m-bain/whisperX/blob/main/pyproject.toml
53. whisperX issue #499 (onnxruntime CPU vs GPU) – https://github.com/m-bain/whisperX/issues/499
54. whisperX issue #274 "Diarization too slow" – https://github.com/m-bain/whisperX/issues/274
55. DiariZen – https://github.com/BUTSpeechFIT/DiariZen
56. BUT-FIT/diarizen-wavlm-large-s80-md (CC‑BY‑NC‑4.0) – https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md
57. sherpa-onnx – https://github.com/k2-fsa/sherpa-onnx
58. sherpa-onnx offline‑diariseringsexempel – https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py
59. 3D‑Speaker – https://github.com/modelscope/3D-Speaker
60. pyannote-onnx – https://github.com/pengzhendong/pyannote-onnx
61. Linux Foundation, pressmeddelande OpenMDW‑1.1 – https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families
62. An approach to optimize inference of the DIART pipeline (arXiv 2408.02341, endast abstract) – https://arxiv.org/abs/2408.02341
