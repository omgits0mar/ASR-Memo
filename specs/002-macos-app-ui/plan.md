# Implementation Plan: macOS Meeting Assistant App — UI, End-to-End Validation & Packaging

**Branch**: `002-macos-app-ui` | **Date**: 2026-06-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-macos-app-ui/spec.md`

## Summary

Put a clean, native-feeling macOS desktop UI on top of the implemented `001` backend
and prove the whole product works end-to-end on real audio. The front-end is a
**single-process Python desktop app**: a `pywebview` host that renders an HTML/CSS/JS
UI in the macOS-native `WKWebView` and calls the existing in-process `meeting_asr`
library directly through a thin JS↔Python bridge — **no separate server, no
cross-language bridge** (FR-006). The app does live diarized/multilingual capture
(US1), file import of downloaded/dataset clips (US2), guided first-run model-download +
permission setup (US3), readable review + Markdown/JSON export (US4), and ships as a
double-click-launchable, ad-hoc-signed `.app` (FR-011).

To make "it is working correctly" verifiable, this feature also (a) **completes the
real-model inference kernels** that `001` left as `needs_models` placeholders
(Nemotron token/language decode; Sortformer probability→AOSC decode), (b) adds a
production **file-import capture** backend (promoting the test `FixtureCapture` to a
shipped `FileCapture`), and (c) adds a repeatable **accuracy-validation harness** over
small public ASR/diarization/multilingual sample clips that reports WER, diarization
accuracy, and language-ID accuracy against thresholds (US5; WER ≤ 15%).

## Technical Context

**Language/Version**: Python 3.11 (app host, bridge, backend, validation); HTML/CSS/JS (webview UI, no Node build step); Swift 5.9+ (existing Process-Tap helper, reused)
**Primary Dependencies**: `pywebview` (WKWebView host + `js_api` bridge) added to the existing stack (`onnxruntime` CoreML EP, `coremltools`, `huggingface_hub`, `sounddevice`, `numpy`, `soxr`, `soundfile`); `PyInstaller` (dev-only) for `.app` packaging; `jiwer` (dev-only) for WER, plus a diarization-error metric helper for DER (validation only). Reference-only: `nemo_toolkit[asr]`, `torch` (MPS)
**Storage**: Local filesystem only — HF model cache (`~/.cache/meeting_asr/models`, unchanged), user-chosen export files (`.md`/`.json`), a cached small validation-sample set under `tests/fixtures/validation/`; no database
**Testing**: `pytest` offline suite (network guard) with deterministic fakes + synthetic fixtures for the bridge/app-session/export logic; `needs_models` gates for real-model WER/DER/language-ID; a headless bridge test that drives the JS-API surface without launching a window
**Target Platform**: macOS 14.4+ (Process Taps) / 13.0–14.3 (ScreenCaptureKit fallback), Apple Silicon (M1+)
**Project Type**: Single-process desktop application (Python webview front-end + in-process Python backend library) with a bundled native capture helper
**Performance Goals**: Live transcript line visible ≤ ~2 s after speech (SC-003, within the ≤3s constitution gate); relaunch-to-Ready < 30 s from cache (SC-002); sustain a 60-min session without timeline loss or UI freeze (SC-011, FR-018); UI stays responsive during long file imports (progress, not frozen)
**Constraints**: Fully on-device — no audio/transcript egress (Principle I, SC-010); UI thread never blocked by inference (all backend work on worker threads, results marshalled to JS); ASR/diarizer memory envelope unchanged from 001 (~1.2 GB FP16 within 16 GB); ad-hoc signing only (Developer-ID/notarization deferred)
**Scale/Scope**: Single user, single active session at a time (FR-020 sequential sessions reused from 001); up to ~4 concurrent speakers; ~40 language-locales; a *small curated* validation subset (not full benchmark coverage)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Local-First Processing | ✅ PASS | UI calls the in-process backend directly; no server, no network during sessions. Only network use remains the one-time model download (existing `prepare_models`). Export writes local files; validation reads locally cached clips. (FR-007, SC-010) |
| II | Real-Time Pipeline Architecture | ✅ PASS | Reuses 001's concurrent diarize∥transcribe-fused-by-timestamp pipeline unchanged. The UI is a presentation layer fed by the existing `on_segment` streaming callback; rendering is marshalled to the webview thread without blocking capture/inference. Completing real-model kernels does not alter stage decoupling. |
| III | Platform-Native Audio Interception | ✅ PASS | Reuses 001 capture (Core Audio Process Taps + mic) behind `AudioCapture`. New `FileCapture` is another `AudioCapture` implementation (file → 16 kHz mono frames) and adds no OS-coupling to the pipeline. |
| IV | Speaker Diarization as a First-Class Stage | ✅ PASS | Diarization remains first-class and authoritative; completing the Sortformer decode kernel realizes the existing AOSC stable-label contract. The UI only displays the labels the diarizer assigns. Coupling option (b) unchanged. |
| V | Hardware-Aware Inference Backends | ✅ PASS | No backend change: ONNX Runtime CoreML EP (FP16, `.cpuAndGPU`) for ASR, CoreML for Sortformer, via the existing resolver. Completing the inference kernels uses the already-loaded accelerated sessions; CPU fallback path preserved. `pywebview`/`PyInstaller` are UI/packaging only, not inference. |
| VI | Automatic Language Detection Per Turn | ✅ PASS | Completing the Nemotron decode wires per-token detected language into `TranscriptSegment.language`; the UI shows a per-segment language tag and the validation harness measures per-segment language-ID accuracy (SC-005). `language=None` auto-detect preserved. |
| VII | Modular, Interface-Driven Architecture | ✅ PASS | New code sits behind/around existing interfaces: `FileCapture` implements `AudioCapture`; export and validation are separate modules consuming `TranscriptSegment`; the app/bridge depend only on the public facade (`prepare_models`/`check_readiness`/`start_session`) plus a small new file-transcription entry point. No module reaches into another's internals. The webview UI is fully decoupled from backend internals via the JS-API contract. |

**Tech-stack additions vs. the constitution's "Technology Stack" section** (capability-level
principles satisfied; the section lists are RECOMMENDED DEFAULTS under v2.0.0):
- **`pywebview` (WKWebView)** — UI host. Not an inference backend; satisfies the spec's
  clarified "Python-based desktop app (native-feeling webview UI), in-process backend,
  packaged as `.app`" with the lowest integration risk. Recorded in Complexity Tracking.
- **`PyInstaller`** — dev-only `.app` bundler (ad-hoc signed). Build tooling, not runtime.
- **`jiwer` + a DER helper** — dev/validation-only metrics. Not shipped in the `.app`.

**Gate result**: PASS (no unwaived deviations). The feature adds a presentation layer,
a file-import capture backend, export, packaging, and a validation harness — all behind
or alongside existing protocols. Cleared for Phase 0/1.

## Project Structure

### Documentation (this feature)

```text
specs/002-macos-app-ui/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── js_bridge_api.md      # JS↔Python webview bridge (methods + events)
│   ├── file_transcription.md # File-import capture + facade entry point
│   ├── transcript_export.md  # Markdown + JSON export contract
│   └── validation_report.md  # Accuracy-validation harness output schema
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/meeting_asr/                 # Existing backend (001) — extended, not restructured
├── __init__.py                  # + transcribe_file(...) facade entry point (file → session)
├── audio/
│   └── file_capture.py          # NEW: FileCapture(AudioCapture) — file → 16 kHz mono frames
│                                #      (production promotion of tests/_fakes.FixtureCapture)
├── asr/
│   └── nemotron_onnx.py         # COMPLETE real kernel: _build_inputs/_parse_tokens/_decode_token
├── diarization/
│   └── sortformer_coreml.py     # COMPLETE real kernel: _infer_frame/_parse_outputs (AOSC)
└── export/                      # NEW module
    ├── __init__.py              # export_markdown(...) / export_json(...)
    ├── markdown.py              # speaker-grouped, human-readable
    └── json_export.py           # structured: speaker, start, end, language, text, confidence

app/                             # NEW: desktop front-end (single process; imports meeting_asr)
├── __init__.py
├── main.py                      # pywebview entry: create window, mount Bridge, run()
├── bridge.py                    # Api class exposed as js_api; marshals backend ↔ webview thread
├── app_session.py              # AppSession view-model: wraps a backend session/file run + state
└── web/                         # Static UI assets (no Node build step)
    ├── index.html               # Setup → Ready → Session → Review screens
    ├── styles.css               # Clean native-feeling macOS look; speaker color-coding
    └── app.js                   # Calls window.pywebview.api.*; renders streamed events

validation/                      # NEW: repeatable accuracy harness (dev/QA, not shipped)
├── __init__.py
├── datasets.py                  # Curated public-clip manifest + local cache loader
├── metrics.py                   # WER (jiwer), diarization accuracy/DER, language-ID accuracy
├── runner.py                    # Feed clips through the integrated pipeline → ValidationReport
└── __main__.py                  # `python -m validation` CLI → JSON/Markdown report + pass/fail

packaging/                       # NEW: .app bundling (ad-hoc signed)
├── MeetingAssistant.spec        # PyInstaller spec (--windowed, bundles app/web + native helper)
├── Info.plist.in               # NSMicrophoneUsageDescription + audio-capture descriptions
└── build_app.sh                 # PyInstaller build + `codesign --force --deep --sign -` (ad-hoc)

tests/
├── _fakes.py                    # Existing fakes (FixtureCapture is the FileCapture reference)
├── unit/
│   ├── test_file_capture.py     # NEW: file → frames, resample, bad/short/unreadable files
│   ├── test_export.py           # NEW: Markdown + JSON round-trip preserves all fields
│   └── test_validation_metrics.py # NEW: WER/DER/language-ID metric correctness on known input
├── contract/
│   ├── test_bridge_api.py       # NEW: js_api surface (headless) — methods + event payloads
│   └── test_file_transcription.py # NEW: transcribe_file() facade contract
├── integration/
│   ├── test_app_session_live.py # NEW: bridge drives a fake live session end-to-end
│   ├── test_app_session_file.py # NEW: bridge drives a file-import run end-to-end (fakes)
│   └── test_validation_run.py   # NEW: harness over synthetic labeled clips → report (offline)
│   # needs_models: real-model WER/DER/language-ID acceptance gates (SC-004/005/006/007)
└── fixtures/
    └── validation/              # NEW: small cached labeled clips + ground-truth manifests

Makefile                         # + `make app` (bundle), `make validate` (accuracy harness), `make run`
requirements.txt / pyproject.toml # + pywebview (runtime); PyInstaller/jiwer (dev extras)
```

**Structure Decision**: Keep 001's single-project `src/meeting_asr/` library as the
on-device core and **extend it in place** (file-capture backend, completed inference
kernels, export module). Add three sibling top-level packages that *depend on* the
library but are not part of it: `app/` (webview front-end), `validation/` (QA harness),
and `packaging/` (build tooling). This preserves Constitution VII module boundaries —
the UI talks to the backend only through the public facade plus the new JS-bridge
contract — and keeps the shippable `.app` (`app/` + `meeting_asr`) cleanly separated
from dev-only QA/build code (`validation/`, `packaging/`, `tests/`).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| **New top-level `app/` package (a "front-end" beyond the library)** | The spec's headline deliverable is a graphical, double-click macOS app; 001 deliberately deferred all UI. A presentation layer cannot live inside the dependency-free `meeting_asr` core without violating its no-UI/no-server contract. | Embedding UI in the library was rejected: it would couple the offline-testable core to `pywebview`/WKWebView and break the in-process-only, headless-CI guarantee. A separate `app/` keeps the core pure. |
| **`pywebview` (WKWebView) UI toolkit + `PyInstaller` packager** (not in the constitution's stack list) | Spec clarification mandates a Python webview desktop app calling the in-process backend, packaged as a double-click `.app`. `pywebview` is the lowest-risk way to get a native WKWebView with a direct Python bridge; `PyInstaller --windowed` is the standard ad-hoc `.app` bundler for this dep set. | A separate local web server + browser was rejected (extra process, not a real `.app`, weaker "in-process" guarantee). Native AppKit/SwiftUI was rejected because it reintroduces the cross-language bridge the clarification explicitly excluded. |
| **New `validation/` harness package** | FR-016/017 + SC-006/007 require a *repeatable, objective* accuracy pass (WER/DER/language-ID) to demonstrate correctness — this is the user's explicit "fully test that it is working" requirement and cannot be met by the offline fakes alone. | Folding validation into `tests/` was rejected: it must run the *real* models over labeled public clips on demand (a `needs_models` capability/CLI), distinct from the offline unit/contract suite. |
