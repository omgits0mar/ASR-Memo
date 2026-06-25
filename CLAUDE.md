# ASR_MeetingMinutes Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-06-15

## Active Technologies
- Python 3.11 (app host, bridge, backend, validation); HTML/CSS/JS (webview UI, no Node build step); Swift 5.9+ (existing Process-Tap helper, reused) + `pywebview` (WKWebView host + `js_api` bridge) added to the existing stack (`onnxruntime` CoreML EP, `coremltools`, `huggingface_hub`, `sounddevice`, `numpy`, `soxr`, `soundfile`); `PyInstaller` (dev-only) for `.app` packaging; `jiwer` (dev-only) for WER, plus a diarization-error metric helper for DER (validation only). Reference-only: `nemo_toolkit[asr]`, `torch` (MPS) (002-macos-app-ui)
- Local filesystem only — HF model cache (`~/.cache/meeting_asr/models`, unchanged), user-chosen export files (`.md`/`.json`), a cached small validation-sample set under `tests/fixtures/validation/`; no database (002-macos-app-ui)

- Python 3.11 (library + pipeline); Swift 5.9+ (Core Audio Process-Tap capture helper) + `onnxruntime` (CoreML EP), `coremltools`, `huggingface_hub`, `sounddevice` (PortAudio), `numpy`, `soxr`/`samplerate`, `soundfile`; native Swift Process-Tap helper. Reference-only: `nemo_toolkit[asr]`, `torch` (MPS) (001-meeting-asr-backend)

## Project Structure

```text
src/meeting_asr/      # in-process Python library (the backend)
  __init__.py          # public facade: prepare_models / check_readiness / start_session
  types.py _logging.py session.py pipeline.py
  audio/ (capture, microphone, coreaudio_tap, screencapturekit, mixer)
  diarization/ (diarizer protocol, sortformer_coreml)
  asr/ (transcriber protocol, nemotron_onnx)
  fusion/ (aligner)
  models/ (registry, readiness)
  backends/ (device resolver)
native/AudioTap/       # Swift Core Audio Process-Tap helper (macOS 14.4+)
tests/ (unit, contract, integration, fixtures/audio) + _fakes.py _synth.py _metrics.py
specs/001-meeting-asr-backend/   # spec/plan/research/data-model/contracts/tasks
```

## Commands

```
make setup          # venv + deps (one-time, network allowed)
make build-native   # Swift Process-Tap helper (macOS 14.4+)
make test           # full offline suite (network guard active)
make test-fast      # unit + contract only
```
Run a subset directly: `python3 -m pytest tests/unit tests/contract`.

## Code Style

Python 3.9+-clean via `from __future__ import annotations` (3.11 is the declared
reference runtime). Swift 5.9+ for the Process-Tap helper. Heavy ML deps
(onnxruntime/coremltools/sounddevice/huggingface_hub) are lazy-imported per module
so unit/contract/integration tests run offline with deterministic fakes. Follow
standard conventions; `ruff`/`black` configured in `pyproject.toml`.

## Recent Changes
- 002-macos-app-ui: Added Python 3.11 (app host, bridge, backend, validation); HTML/CSS/JS (webview UI, no Node build step); Swift 5.9+ (existing Process-Tap helper, reused) + `pywebview` (WKWebView host + `js_api` bridge) added to the existing stack (`onnxruntime` CoreML EP, `coremltools`, `huggingface_hub`, `sounddevice`, `numpy`, `soxr`, `soundfile`); `PyInstaller` (dev-only) for `.app` packaging; `jiwer` (dev-only) for WER, plus a diarization-error metric helper for DER (validation only). Reference-only: `nemo_toolkit[asr]`, `torch` (MPS)

- 001-meeting-asr-backend (implemented): full on-device pipeline — capture (mic +
  system via Core Audio Process Taps) → mix (16 kHz mono) → diarize (Sortformer
  CoreML) ∥ transcribe (Nemotron ONNX FP16) → timestamp fusion → session/facade.
  Protocol-driven; 83 offline tests + 6 needs_models gates. Whisper prototype
  `transcribe_meeting.py` deprecated in favor of `meeting_asr`.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
