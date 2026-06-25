# ASR-Memo Development Guidelines

On-device, real-time, **diarized + multilingual** meeting transcription for macOS
Apple Silicon. Two NVIDIA models run concurrently (speaker diarization ∥ ASR) and
are fused by timestamp; a `pywebview` desktop app sits on top of an in-process
Python library. Repo: `omgits0mar/ASR-Memo`. Last updated: 2026-06-25.

## Models (downloaded to `~/.cache/meeting_asr/models`)

- **ASR** — NVIDIA Nemotron 3.5 ASR Streaming Multilingual 0.6B, run as the FP16
  ONNX export [`soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16`](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16)
  via ONNX Runtime (CoreML EP). 3-graph cache-aware streaming RNNT
  (encoder/decoder/joint + `.data` sidecars). ~40 language-locales.
- **Diarization** — NVIDIA Streaming Sortformer 4spk-v2.1, run as the CoreML build
  [`FluidInference/diar-streaming-sortformer-coreml`](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml)
  via coremltools. Up to 4 speakers, overlap-aware.
- Repo IDs + pinned revisions live in `src/meeting_asr/models/registry.py`; fetched
  via `huggingface_hub`. Respect each upstream NVIDIA model card's license.

## Active Technologies
- Python 3.11 (app host, bridge, backend, validation); HTML/CSS/JS (webview UI, no Node build step); Swift 5.9+ (existing Process-Tap helper, reused) + `pywebview` (WKWebView host + `js_api` bridge) added to the existing stack (`onnxruntime` CoreML EP, `coremltools`, `huggingface_hub`, `sounddevice`, `numpy`, `soxr`, `soundfile`); `PyInstaller` (dev-only) for `.app` packaging; `jiwer` (dev-only) for WER, plus a diarization-error metric helper for DER (validation only). Reference-only: `nemo_toolkit[asr]`, `torch` (MPS) (002-macos-app-ui)
- Local filesystem only — HF model cache (`~/.cache/meeting_asr/models`, unchanged), user-chosen export files (`.md`/`.json`), a cached small validation-sample set under `tests/fixtures/validation/`; no database (002-macos-app-ui)

- Python 3.11 (library + pipeline); Swift 5.9+ (Core Audio Process-Tap capture helper) + `onnxruntime` (CoreML EP), `coremltools`, `huggingface_hub`, `sounddevice` (PortAudio), `numpy`, `soxr`/`samplerate`, `soundfile`; native Swift Process-Tap helper. Reference-only: `nemo_toolkit[asr]`, `torch` (MPS) (001-meeting-asr-backend)

## Project Structure

```text
src/meeting_asr/      # in-process Python library (the backend)
  __init__.py          # public facade: prepare_models / check_readiness / start_session
  types.py _logging.py session.py pipeline.py
  audio/ (capture, microphone, coreaudio_tap, file_capture, mixer)
  diarization/ (diarizer protocol, sortformer_coreml)
  asr/ (transcriber protocol, nemotron_onnx)
  fusion/ (aligner)
  models/ (registry, readiness)
  backends/ (device resolver)
  export/ (markdown + json transcript export)
app/                   # desktop app: main.py (pywebview host) + bridge.py (Api/js_api)
  dto.py + web/ (index.html, app.js, styles.css)   # static UI, no Node build step
native/AudioTap/       # Swift Core Audio Process-Tap helper (system audio, macOS 14.4+)
validation/            # accuracy harness — WER / DER / language-ID over labeled clips
packaging/             # PyInstaller .app bundling + ad-hoc codesign
tests/ (unit, contract, integration, fixtures/audio) + _fakes.py _synth.py _metrics.py
specs/ (001-meeting-asr-backend, 002-macos-app-ui)   # spec/plan/research/contracts/tasks
```

## Commands

```
make setup          # venv + deps (one-time, network allowed)
make build-native   # Swift Process-Tap helper (system audio, macOS 14.4+)
make run            # launch the desktop app (python -m app.main)
make validate       # accuracy harness (WER / DER / language-ID) — needs models
make app            # build the double-click .app (PyInstaller + ad-hoc sign)
make test           # full offline suite (network guard active)
make test-fast      # unit + contract only
```
Run a subset directly: `python3 -m pytest tests/unit tests/contract`.
This repo also uses a conda env `meeting_asr` (py3.11) in practice — launch via
`conda run -n meeting_asr python -m app.main` (set `PYTHONPATH` to the repo root
for scripts that import `app`).

## Code Style

Python 3.9+-clean via `from __future__ import annotations` (3.11 is the declared
reference runtime). Swift 5.9+ for the Process-Tap helper. Heavy ML deps
(onnxruntime/coremltools/sounddevice/huggingface_hub) are lazy-imported per module
so unit/contract/integration tests run offline with deterministic fakes. Follow
standard conventions; `ruff`/`black` configured in `pyproject.toml`.

## Recent Changes
- Repo published as `omgits0mar/ASR-Memo` (public); README rebranded to ASR-Memo
  with NVIDIA / Hugging Face model references.
- 002-macos-app-ui (implemented): pywebview WKWebView desktop app on top of the 001
  backend — `app/main.py` host + `app/bridge.py` (`Api` / `js_api`), static
  HTML/CSS/JS UI, `FileCapture` import path, Markdown/JSON export, `validation/`
  accuracy harness (WER / DER / language-ID), `packaging/` PyInstaller `.app`. Live
  emit decoupled from the blocking `evaluate_js` via a daemon pump thread.

- 001-meeting-asr-backend (implemented): full on-device pipeline — capture (mic +
  system via Core Audio Process Taps) → mix (16 kHz mono) → diarize (Sortformer
  CoreML) ∥ transcribe (Nemotron ONNX FP16) → timestamp fusion → session/facade.
  Protocol-driven; 152 offline tests + `needs_models`/`needs_hardware`/`slow` gates.
  Whisper prototype `transcribe_meeting.py` deprecated in favor of `meeting_asr`.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
