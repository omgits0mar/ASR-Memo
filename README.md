# ASR_MeetingMinutes — Diarized Meeting Transcription Backend

On-device, realtime, **diarized + multilingual** meeting transcription as an
**in-process Python library** (no UI — backend/APIs only). Optimized for the
lowest-latency path on macOS Apple Silicon:

- **Capture**: Core Audio Process Taps (system/meeting-app audio, macOS 14.4+) +
  PortAudio microphone, both resampled to 16 kHz mono on one session clock.
- **Diarization**: NVIDIA Streaming Sortformer (CoreML) — stable arrival-order
  speaker labels, overlapping speech, up to 4 speakers.
- **ASR**: NVIDIA Nemotron 3.5 ASR Streaming 0.6B (FP16 ONNX via ONNX Runtime
  CoreML EP, M-series GPU + CPU) — cache-aware streaming, ~40 language-locales.
- **Topology**: diarization and ASR run **concurrently** and are **fused by
  timestamp** (~1.5–2.0s turn-to-text, within the ≤3s gate).

> The existing `transcribe_meeting.py` (Whisper batch prototype) is **superseded**
> by this library (see spec Assumptions; migration tracked as T046).

## Quick start

See [`specs/001-meeting-asr-backend/quickstart.md`](specs/001-meeting-asr-backend/quickstart.md)
for the full walkthrough. In short:

```bash
make setup                 # venv + deps (one-time, network allowed)
make build-native          # Core Audio Process-Tap helper (system audio)

python3
>>> from meeting_asr import prepare_models, check_readiness, start_session
>>> prepare_models(progress=lambda p: print(p.asset, p.state, p.downloaded, p.total))
>>> print(check_readiness())                       # models + perms + compute backend
>>> session = start_session(on_segment=lambda s: print(s.start, s.speaker_label, s.text))
>>> ...                                            # run your meeting
>>> final = session.stop()
```

## Architecture

```
capture (mic + system) → mix(16kHz mono) → ┌─ diarize (Sortformer CoreML)
                                            └─ transcribe (Nemotron ONNX, FP16)
                                                          ↓
                              fusion.aligner → TranscriptSegment (speaker, t, text, lang)
                                                          ↓
                                       session (state machine, streaming + queryable)
                                                          ↓
                                       public facade: prepare_models / check_readiness / start_session
```

Protocol-driven and fully testable offline: every stage (`AudioCapture`,
`SpeakerDiarizer`, `SpeechTranscriber`) is a `Protocol`, so a recorded fixture
or deterministic fake stands in for live capture/models — satisfying the
constitution's offline-CI gate.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Setup (skeleton, deps, tooling, Swift) | ✅ |
| 2 | Foundational (types, protocols, device, registry, mixer, harness) | ✅ |
| 3 | US1 MVP — live diarized transcription (single source) | ✅ |
| 4 | US2 — system/remote participants | ✅ |
| 5 | US3 — multilingual per-segment | ✅ |
| 6 | US4 — model lifecycle & readiness API | ✅ |
| 7 | Polish — perf/quant/WER/diar-accuracy gates, offline CI | ✅ |

**Tests**: 83 passing offline; 6 `needs_models`/`needs_hardware`/`slow` gates skip
cleanly without the downloaded Nemotron/Sortformer models (run `prepare_models()`
+ an Apple-Silicon machine to exercise them). The architecture is fully
protocol-driven, so the entire pipeline is verified offline with deterministic
fakes + synthetic fixtures.

See [`specs/001-meeting-asr-backend/tasks.md`](specs/001-meeting-asr-backend/tasks.md).

## Desktop app & validation (002)

Feature `002-macos-app-ui` puts a clean, native-feeling macOS desktop UI on the
backend and proves the whole product works end-to-end on real audio. It is a
**single-process Python desktop app**: a `pywebview` WKWebView window renders a
static HTML/CSS/JS UI and calls the in-process `meeting_asr` library through a thin
JS↔Python bridge — no separate server, no cross-language bridge.

**What it adds** (around the unchanged 001 pipeline):
- **`app/`** — pywebview host (`app/main.py`), the `Api` bridge (`app/bridge.py`),
  the `AppSession` view-model, and the static UI (`app/web/`).
- **`src/meeting_asr/audio/file_capture.py`** — production `FileCapture` (promotes
  the test fixture): import a file → same diarized transcript.
- **`src/meeting_asr/export/`** — Markdown + JSON export of the transcript.
- **`validation/`** — repeatable accuracy harness: WER / diarization / language-ID
  over small labeled public clips (LibriSpeech / AMI / FLEURS).
- **`packaging/`** — PyInstaller `.app` bundling + ad-hoc codesign.

**Run / validate / package:**

```bash
make run        # launch the desktop app (python -m app.main)
make validate   # accuracy harness over curated clips (needs models)
make app        # build the double-click .app (PyInstaller + ad-hoc sign)
```

**User stories** — US1 live diarized/multilingual capture · US2 import a clip ·
US3 guided first-run setup + double-click `.app` · US4 review + Markdown/JSON
export · US5 accuracy validation (WER ≤ 15%, diarization ≥ 90%, language-ID ≥ 95%).
See [`specs/002-macos-app-ui/quickstart.md`](specs/002-macos-app-ui/quickstart.md)
and [`specs/002-macos-app-ui/tasks.md`](specs/002-macos-app-ui/tasks.md).

The bridge + file-import + export + validation-harness logic is fully exercised
offline with fakes/synthetic fixtures; real-model inference and the packaged
double-click launch are gated behind `needs_models` / Apple-Silicon hardware.

## Testing

```bash
make test        # full offline suite (network guard active)
make test-fast   # unit + contract only
```

Tests requiring downloaded models (`needs_models`), live hardware
(`needs_hardware`), or long runs (`slow`) are marked and deselected by default.
