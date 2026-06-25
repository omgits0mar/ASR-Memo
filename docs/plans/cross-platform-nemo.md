# Implementation Plan: Cross-Platform Unified App (macOS / Linux / Windows, NVIDIA GPU + CPU)

> **Status:** Approved, implementation-ready.
> **How to start implementation:** When the maintainer gives the go-ahead flag,
> Claude should execute the workstreams below **in order** (1 → 6). Workstream 2's
> "NeMo streaming word-level timestamps" spike (see Risks) must be validated before
> writing the rest of the NeMo ASR backend. Use TDD with the existing offline fakes;
> do not break the macOS CoreML/ONNX path. Confirm the two open items under
> "Confirm-before-coding" before downloading models.

## Context

`meeting_asr` today is macOS-Apple-Silicon-only: the ASR backend
(`NemotronOnnxTranscriber`) runs the FP16 **ONNX** export via ONNX Runtime CoreML
EP, and the diarizer (`SortformerCoreMLDiarizer`) loads a **CoreML** `.mlpackage`
via `coremltools` — both Apple-only at the model layer, plus a macOS-only Swift
Process-Tap for system audio. We want one app that also runs on Linux and Windows,
using **NVIDIA's native NeMo/PyTorch models** (not ONNX conversions) on machines
with a CUDA GPU, and **NeMo on torch-CPU** on machines without one.

The architecture is already protocol-first, so this is **additive**, not a rewrite.
The platform coupling is isolated to four seams: the backend selection point
(`__init__.py`), the device resolver (`backends/device.py`), the model registry
(`models/registry.py`), and audio capture (`audio/`). All streaming/decode/fusion
logic is pure numpy and stays untouched.

**Decisions (confirmed with maintainer):**
- macOS keeps the existing CoreML + ONNX path (no re-validation risk).
- Non-mac with NVIDIA → NeMo on CUDA; non-mac without NVIDIA → NeMo on torch-CPU.
- System-audio loopback (WASAPI + PipeWire) is **in scope now**.
- NeMo+torch ships as an **optional dependency extra**, base install stays light.

## Target backend-selection matrix

| Host | ASR | Diarization | System audio |
|---|---|---|---|
| macOS (Apple Silicon) | `NemotronOnnxTranscriber` (ONNX/CoreML EP) | `SortformerCoreMLDiarizer` | Core Audio Process Tap (existing) |
| Linux/Windows + NVIDIA | `NemotronNeMoTranscriber` (CUDA) | `SortformerNeMoDiarizer` (CUDA) | WASAPI / PipeWire loopback |
| Linux/Windows, no GPU | `NemotronNeMoTranscriber` (torch-CPU) | `SortformerNeMoDiarizer` (torch-CPU) | WASAPI / PipeWire loopback |

---

## Workstream 1 — Device resolution (`backends/device.py`, `types.py`)

- Extend `ComputeBackend` enum (`types.py:82`): add `CUDA = "cuda"` and
  `TORCH_CPU = "torch-cpu"`. Keep existing CoreML/CPU/MPS values.
- Extend `DeviceProbe` with `has_torch` + `has_cuda` callables; add real probes
  (`_has_torch`, `_has_cuda` → `torch.cuda.is_available()`), lazy-imported.
- Rewrite `resolve_backend()` ordering:
  1. Apple Silicon + CoreML EP → `COREML_GPU_CPU` (unchanged mac path)
  2. torch + CUDA available → `CUDA`
  3. torch available → `TORCH_CPU`
  4. onnxruntime available → `CPU` (ONNX CPU EP fallback, e.g. mac w/o CoreML)
- All probes injectable → fully unit-testable offline (existing pattern).

## Workstream 2 — NeMo inference backends (new modules)

Both conform to the **existing protocols** (`asr/transcriber.py:SpeechTranscriber`,
`diarization/diarizer.py:SpeakerDiarizer`) — same `load(backend, ...)`, `reset()`,
`push()`, `flush()`, `supported_languages()`/`max_speakers()`, plus the
whole-utterance `transcribe_array()`/`diarize_array()` used by validation. NeMo is
**lazy-imported inside `load()`** so the offline test suite (fakes) never imports
torch/NeMo.

- **`asr/nemotron_nemo.py` → `NemotronNeMoTranscriber`**
  - `load()`: `ASRModel.restore_from(<cached .nemo>)`, `.to(device)`, `.eval()`,
    half precision on CUDA. Device from `ComputeBackend` (cuda/cpu).
  - Streaming via NeMo cache-aware streaming inference (the
    `speech_to_text_cache_aware_streaming_infer` API: `conformer_stream_step` with
    carried encoder/decoder cache tensors) — the NeMo-native equivalent of the
    hand-rolled loop in `nemotron_onnx.py`.
  - **Key contract detail:** emit **word-level `AsrToken`s with `t_start`/`t_end`**
    (request NeMo word timestamps) so the existing `fusion/aligner.py` is unchanged.
    This timestamp extraction is the main integration risk — validate early.
  - Reuse `NEMOTRON_LANGUAGES` + the `language_hint`→prompt logic; NeMo exposes the
    multilingual model's own language selection.
- **`diarization/sortformer_nemo.py` → `SortformerNeMoDiarizer`**
  - `load()`: `SortformerEncLabelModel.restore_from(<cached .nemo>)`, `.to(device)`.
  - Streaming diarization step API → per-frame speaker probs → `DiarFrame`s (80 ms),
    reusing the arrival-order labelling + `ACTIVATION_THRESHOLD` decode logic
    already in `sortformer_coreml.py` (lift `_decode`/`_label_for` into a shared
    helper to avoid duplication).

## Workstream 3 — Model registry & download (`models/registry.py`, `types.py`)

- Add a `framework` field to `ModelAsset` (`types.py`): `COREML` | `ONNX` | `NEMO`.
- Add NeMo assets to the registry:
  - ASR: `nvidia/nemotron-3.5-asr-streaming-0.6b` (`.nemo`).
  - Diarization: NVIDIA streaming Sortformer `.nemo` repo — **confirm exact repo id**
    (model card links only the ASR repo).
- Parameterize `model_registry(backend)` (or `framework`) to return the asset set
  matching the resolved backend, instead of always returning ONNX+CoreML.
- Download stays in the existing `prepare()` flow via `snapshot_download` of the
  `.nemo` into the same cache (`~/.cache/meeting_asr/models`); the NeMo backends use
  `restore_from(<cached path>)` (not `from_pretrained`) so `prepare_models()` remains
  the single authoritative, offline-after-download step. `check_cached`/integrity
  gate work unchanged on the `.nemo` file.
- `models/readiness.py:build_readiness()` already calls `resolve_backend()`; switch
  its `model_registry()` call to the backend-aware selection so readiness reports the
  correct missing models per platform.

## Workstream 4 — Backend factory & wiring (`__init__.py`, new `backends/factory.py`)

- New `backends/factory.py:build_inference_backends(backend) -> (diarizer, transcriber)`:
  - CoreML/ONNX-CPU backends → `SortformerCoreMLDiarizer` + `NemotronOnnxTranscriber`
  - `CUDA`/`TORCH_CPU` → `SortformerNeMoDiarizer` + `NemotronNeMoTranscriber`
- Refactor the two hardcoded sites to call the factory:
  - `__init__.py:_build_default_backends()` (line ~287, currently hardcodes the two
    Apple backends) — resolve backend, then build via factory.
  - `__init__.py:transcribe_file()` (line ~237, same hardcoded pair).
- `load()` call sites in `start_session()` (`__init__.py:183`) are unchanged — the
  resolved `ComputeBackend` already flows into `diarizer.load(backend)` /
  `transcriber.load(backend, precision=...)`; NeMo backends honor the same signature
  (precision → fp16 on CUDA, fp32 on CPU).

## Workstream 5 — Cross-platform audio capture (`audio/`)

- **Microphone:** confirm `audio/microphone.py` (sounddevice/PortAudio) is already
  OS-agnostic; fix any mac-only assumptions.
- **Windows loopback:** new `audio/wasapi_loopback.py` implementing `AudioCapture`
  via WASAPI loopback (sounddevice WASAPI loopback flag / `soundcard`), emitting the
  canonical 16 kHz mono float32 frames.
- **Linux loopback:** new `audio/pipewire_loopback.py` implementing `AudioCapture`
  via a PulseAudio/PipeWire `.monitor` source through PortAudio.
- Generalize the SYSTEM branch in `_build_default_backends()` to pick the right
  loopback capture per OS (mac → existing `CoreAudioTapCapture`, Windows → WASAPI,
  Linux → PipeWire) inside `CompositeCapture`.
- `models/readiness.py`: generalize `os_supports_process_tap()` →
  `os_supports_system_audio()` (mac 14.4+ / Windows / Linux-with-monitor) and update
  `system_audio_permission()` + the readiness advisory string accordingly.

## Workstream 6 — Packaging & dependencies (`pyproject.toml`, `packaging/`)

- Keep base deps light (mac path unchanged). Add an extra:
  `[project.optional-dependencies] nemo = ["nemo_toolkit[asr]", "torch", "Cython", "packaging"]`.
- Document the Linux/Windows GPU setup: `apt-get install -y libsndfile1 ffmpeg`,
  `pip install meeting_asr[nemo]` (CUDA wheels of torch for GPU hosts).
- Per-OS PyInstaller specs under `packaging/` (mac signed as today; add Windows/Linux
  targets). NeMo extra opt-in keeps base installers small.
- CI matrix (GitHub Actions: macOS + Windows + Linux) running the **offline** suite —
  the existing fakes + lazy imports make this possible with no models/GPU.

---

## Files touched (representative)

- `src/meeting_asr/types.py` — `ComputeBackend` enum, `ModelAsset.framework`
- `src/meeting_asr/backends/device.py` — torch/CUDA probes + resolver order
- `src/meeting_asr/backends/factory.py` — **new** backend factory
- `src/meeting_asr/asr/nemotron_nemo.py` — **new** NeMo ASR backend
- `src/meeting_asr/diarization/sortformer_nemo.py` — **new** NeMo diarizer
- `src/meeting_asr/models/registry.py` + `models/readiness.py` — NeMo assets, backend-aware selection, system-audio generalization
- `src/meeting_asr/__init__.py` — route `_build_default_backends` + `transcribe_file` through the factory
- `src/meeting_asr/audio/wasapi_loopback.py`, `audio/pipewire_loopback.py` — **new** loopback captures
- `pyproject.toml`, `packaging/` — `nemo` extra, per-OS builds, CI matrix
- `tests/` — device-resolver branches, factory, NeMo backends (via fakes), loopback contract tests

## Reuse (do not re-implement)

- Protocols: `asr/transcriber.py`, `diarization/diarizer.py`, `audio/capture.py`
- `fusion/aligner.py`, `session.py`, `pipeline.py` — unchanged
- Decode/label logic in `sortformer_coreml.py:_decode`/`_label_for` → share with NeMo diarizer
- `NEMOTRON_LANGUAGES` + language→prompt resolution
- `models/registry.py:prepare()`/`check_cached()` download+integrity machinery
- Injectable `DeviceProbe` test pattern; existing `_fakes.py`

## Key risks / confirm-before-coding

1. **NeMo streaming word-level timestamps → aligner contract.** Highest risk; spike
   first to confirm NeMo streaming yields per-word `t_start`/`t_end`.
2. **Sortformer `.nemo` repo id + streaming API** on NeMo `main` (recent) — confirm.
3. **NeMo install** pins to git `main`; version-pin for reproducibility.
4. **WASAPI/PipeWire loopback** device discovery reliability across machines.

## Verification

- **Offline (all OSes):** `make test` / `make test-fast` — fakes, no models/GPU;
  must pass on the new mac/Windows/Linux CI matrix.
- **GPU (CUDA box):** `prepare_models()` downloads the `.nemo` files; run a labeled
  clip via `make validate` (WER/DER/language-ID) and compare against the mac baseline.
- **CPU (no-GPU Linux/Windows):** same `make validate` on torch-CPU; confirm
  correctness (speed lower, accuracy on par).
- **Loopback:** play known audio, start a SYSTEM-source session, confirm captured
  frames + transcript on Windows (WASAPI) and Linux (PipeWire).
- **Backend selection:** unit-test `resolve_backend()` + factory across probe combos
  (Apple/CUDA/torch-CPU/ONNX-CPU) so the right backend pair is chosen per host.
