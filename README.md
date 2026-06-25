<div align="center">

# 🎙️ ASR-Memo

**On-device, real-time, diarized & multilingual meeting transcription for macOS Apple Silicon.**

Capture your mic *and* system audio, get a live transcript that knows **who said what**, in **~40 languages** — all running locally, nothing leaves your Mac.

<p>
  <img alt="Platform" src="https://img.shields.io/badge/platform-macOS%2014.4%2B%20(Apple%20Silicon)-black?logo=apple">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Swift" src="https://img.shields.io/badge/swift-5.9%2B-F05138?logo=swift&logoColor=white">
  <img alt="Runtime" src="https://img.shields.io/badge/runtime-ONNX%20Runtime%20%2B%20CoreML-005CED?logo=onnx&logoColor=white">
  <img alt="Privacy" src="https://img.shields.io/badge/privacy-100%25%20on--device-success">
  <img alt="Tests" src="https://img.shields.io/badge/tests-152%20offline-brightgreen">
</p>

</div>

---

## What it does

ASR-Memo turns a live meeting into an attributed, timestamped, multilingual transcript **as it happens** — with the lowest-latency path available on Apple Silicon. It runs **two NVIDIA models concurrently** (speaker diarization ∥ speech recognition) and fuses their outputs by timestamp, so each line is labelled with a speaker and a language.

- 🔒 **Fully on-device** — audio, models, and inference never leave your machine. No cloud, no API keys, no telemetry.
- 🗣️ **Speaker diarization** — stable arrival-order labels (*Speaker 1, Speaker 2, …*), overlapping speech, up to 4 speakers.
- 🌍 **Multilingual ASR** — cache-aware streaming recognition across ~40 language-locales (EN/AR/ES/FR/DE/ZH/JA/…), per-segment language tagging, auto-detect.
- 🎧 **Mic + system audio** — captures your microphone (PortAudio) **and** the audio from meeting apps (Core Audio Process Taps, macOS 14.4+), mixed on one 16 kHz clock.
- ⚡ **Real-time** — ~1.5–2.0 s turn-to-text; transcript streams in live during the session.
- 🖥️ **Native desktop app** — a clean `pywebview` (WKWebView) UI, plus a programmable Python library underneath.
- 📤 **Import & export** — transcribe an existing audio file, export to Markdown or JSON.

---

## Models

ASR-Memo is built on **NVIDIA's** state-of-the-art streaming speech models, run locally through Apple-Silicon-optimized exports (ONNX + CoreML). Weights are pulled from the Hugging Face Hub on first run and cached under `~/.cache/meeting_asr/models`.

| Role | Model | Architecture by | On-device build (Hugging Face) | Notes |
|------|-------|-----------------|-------------------------------|-------|
| **Speech recognition** | Nemotron 3.5 ASR Streaming Multilingual 0.6B | [NVIDIA](https://huggingface.co/nvidia) | [`soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16`](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) | FP16 ONNX (encoder/decoder/joint RNNT graphs); runs on the **ONNX Runtime CoreML EP**. ~40 languages. |
| **Speaker diarization** | Streaming Sortformer 4-speaker v2.1 | [NVIDIA](https://huggingface.co/nvidia) | [`FluidInference/diar-streaming-sortformer-coreml`](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) | CoreML `.mlpackage`; runs via **coremltools** on the Apple Neural Engine / GPU. Up to 4 concurrent speakers, overlap-aware. |

**Credits & licenses** — the underlying [Nemotron ASR](https://huggingface.co/nvidia) and [Streaming Sortformer](https://huggingface.co/nvidia) models are research artifacts from **NVIDIA**; the ONNX/CoreML re-exports are community builds by [soniqo](https://huggingface.co/soniqo) and [FluidInference](https://huggingface.co/FluidInference). Please review and respect each upstream model's license on its Hugging Face model card before redistribution or commercial use. Model fetching/caching is handled by [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub).

---

## Architecture

```text
 ┌─ Microphone (PortAudio) ─┐
 │                          ├─►  Mixer  ──►  16 kHz mono  ──┬─►  Diarize   (Sortformer · CoreML)
 └─ System audio (Core ─────┘     (one session clock)       │
    Audio Process Taps · Swift)                             └─►  Transcribe (Nemotron · ONNX FP16)
                                                                         │
                                                       ┌─────────────────┘  run concurrently
                                                       ▼
                                          fusion.aligner  (timestamp fuse)
                                                       ▼
                                  TranscriptSegment { speaker, start/end, text, language }
                                                       ▼
                                   session  (state machine · streaming + queryable)
                                                       ▼
                          facade:  prepare_models()  ·  check_readiness()  ·  start_session()
                                                       ▼
                                pywebview desktop app  (WKWebView UI ↔ js_api bridge)
```

Every stage is a **`Protocol`** (`AudioCapture`, `SpeakerDiarizer`, `SpeechTranscriber`), so deterministic fakes and recorded fixtures stand in for live capture/models — the entire pipeline is verified **offline**, no network or hardware required in CI.

### Project layout

```text
src/meeting_asr/        # the backend — in-process Python library
  __init__.py             # public facade: prepare_models / check_readiness / start_session
  pipeline.py session.py types.py _logging.py
  audio/                  # capture, microphone, coreaudio_tap, file_capture, mixer
  diarization/            # Sortformer CoreML diarizer (+ protocol)
  asr/                    # Nemotron ONNX transcriber (+ protocol)
  fusion/                 # timestamp aligner
  models/                 # HF registry, download lifecycle, readiness
  backends/               # compute-device resolver
  export/                 # Markdown + JSON transcript export
app/                    # desktop app — pywebview host + JS↔Python bridge
  main.py bridge.py dto.py  +  web/ (index.html, app.js, styles.css)
native/AudioTap/        # Swift Core Audio Process-Tap helper (system audio, macOS 14.4+)
validation/             # accuracy harness — WER / DER / language-ID over labeled clips
packaging/              # PyInstaller .app bundling + ad-hoc codesign
tests/                  # 152 offline tests (unit · contract · integration) + fakes/fixtures
specs/                  # feature specs, plans, contracts, tasks (001 backend, 002 app)
```

---

## Quick start

> **Requirements:** macOS 14.4+ on Apple Silicon (M-series), Python 3.11, Xcode command-line tools (for the Swift helper), and `portaudio` (`brew install portaudio`).

```bash
# 1. Install (one-time; network allowed)
make setup            # creates .venv + installs deps
make build-native     # builds the Swift Core Audio Process-Tap helper (system audio)

# 2. Launch the desktop app
make run              # python -m app.main
```

On first launch the app guides you through **downloading the models** (readiness screen → *Prepare*), then you're ready to record.

**Using the app:**
- **Live** — tick *Microphone* and/or *System audio* → **Start** → grant the macOS mic / audio-capture permissions when prompted → speak / play the meeting → **Stop** → **Export** to Markdown/JSON.
- **Import a file** — *Import audio file* → pick any speech `.wav` → watch the diarized transcript stream in.

### Use it as a library

```python
from meeting_asr import prepare_models, check_readiness, start_session

prepare_models(progress=lambda p: print(p.asset, p.state, p.downloaded, p.total))
print(check_readiness())                      # models + permissions + compute backend

session = start_session(
    on_segment=lambda s: print(f"[{s.start:6.2f}] {s.speaker_label} ({s.language}): {s.text}")
)
# ... run your meeting ...
final = session.stop()                         # full queryable transcript
```

---

## Commands

| Command | What it does |
|---------|--------------|
| `make setup` | Create the venv and install dependencies (one-time, network allowed). |
| `make build-native` | Build the Swift Core Audio Process-Tap helper (system-audio capture). |
| `make run` | Launch the desktop app (`python -m app.main`). |
| `make validate` | Run the accuracy harness (WER / diarization / language-ID) over labeled clips. *Needs models.* |
| `make app` | Build the double-click `.app` bundle (PyInstaller + ad-hoc codesign). |
| `make test` | Full **offline** test suite (network guard active). |
| `make test-fast` | Unit + contract tests only. |

Run a subset directly: `python3 -m pytest tests/unit tests/contract`.

---

## Tech stack

- **Backend / pipeline** — Python 3.11, [NumPy](https://numpy.org/), [soxr](https://github.com/dofuuz/python-soxr) (resampling), [soundfile](https://github.com/bastibe/python-soundfile) (WAV I/O), [sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio mic capture).
- **Inference** — [ONNX Runtime](https://onnxruntime.ai/) (CoreML Execution Provider) for ASR, [coremltools](https://coremltools.readme.io/) for diarization, [huggingface_hub](https://huggingface.co/docs/huggingface_hub) for model download/caching.
- **System audio** — native **Swift 5.9+** Core Audio Process-Tap helper (macOS 14.4+).
- **Desktop UI** — [pywebview](https://pywebview.flowrl.com/) (WKWebView) + a thin `js_api` bridge; static HTML/CSS/JS (no Node build step).
- **Dev / validation** — [pytest](https://pytest.org/), [jiwer](https://github.com/jitsi/jiwer) (WER) + a DER helper, [PyInstaller](https://pyinstaller.org/) (`.app` packaging).

> Heavy ML deps are **lazy-imported** per module, so unit/contract/integration tests run fully offline with deterministic fakes.

---

## Testing & status

```bash
make test        # full offline suite (network guard active)
make test-fast   # unit + contract only
```

**152 tests pass offline.** Tests requiring downloaded models (`needs_models`), live hardware (`needs_hardware`), or long runs (`slow`) are marked and deselected by default — run `prepare_models()` on an Apple-Silicon machine to exercise them.

| Feature | Scope | State |
|---------|-------|-------|
| **001** — backend | Full on-device pipeline: capture → mix → diarize ∥ transcribe → fuse → session/facade | ✅ implemented |
| **002** — macOS app | pywebview desktop UI, file import, Markdown/JSON export, accuracy validation, `.app` packaging | ✅ implemented |

See [`specs/`](specs/) for the per-feature spec, plan, contracts, and task breakdown.

---

## Privacy

Everything runs **locally**. Audio is captured, mixed, diarized, and transcribed entirely on-device; models are downloaded once from Hugging Face and cached on disk. No audio, transcript, or usage data is sent anywhere.

---

## License

See [`LICENSE`](LICENSE). Note that the **NVIDIA models** (Nemotron ASR, Streaming Sortformer) and their community re-exports carry their **own licenses** — review each model card on Hugging Face before redistribution or commercial use.
