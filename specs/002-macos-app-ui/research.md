# Phase 0 Research — macOS App UI, Validation & Packaging

Feature: `002-macos-app-ui` · Date: 2026-06-15

All `NEEDS CLARIFICATION` items from the spec's clarification session are already
resolved (UI = Python webview app calling the in-process backend; packaging = ad-hoc
`.app`; WER ≤ 15%; export = Markdown + JSON; clean native look). This document records
the remaining *technical* decisions needed to start Phase 1.

---

## Decision 1 — UI toolkit: `pywebview` (native WKWebView) + vanilla HTML/CSS/JS

**Decision**: Build the front-end as a `pywebview` app. On macOS, `pywebview` renders
in the OS-native `WKWebView` (Cocoa backend) and exposes an `js_api` Python object whose
methods are callable from JS as `window.pywebview.api.<method>(...)` (returning promises).
Push backend events to the UI with `window.evaluate_js(...)`. The UI itself is plain
HTML/CSS/JS with **no Node/bundler build step**, served from bundled static files.

**Rationale**:
- Directly satisfies the spec clarification: "Python-based desktop app (native-feeling
  webview UI) calls the in-process Python backend directly with no cross-language bridge."
- `pywebview`'s `js_api` *is* the in-process bridge — JS calls run Python in the same
  process; no localhost server, no IPC, no second language.
- WKWebView gives a genuinely native macOS feel and modern CSS for the "clean, polished"
  v1 look without a heavyweight UI framework.
- No bundler keeps the `.app` build simple (PyInstaller just copies `app/web/`), avoiding
  a Node toolchain in the packaging path.

**Alternatives considered**:
- **Toga / BeeWare native widgets** — true native controls, but a richer/heavier widget
  model and weaker fit for the transcript-stream + color-coded layout; `briefcase`
  packaging is its own ecosystem. Rejected for v1 simplicity.
- **Flet (Flutter)** — polished, but bundles a Flutter runtime and is further from
  "native-feeling macOS" + larger `.app`. Rejected.
- **Local Flask/FastAPI server + browser** — reintroduces a separate process and weakens
  the in-process/local-first guarantee the clarification asked for. Rejected.
- **Native AppKit/SwiftUI** — best native feel but reintroduces the cross-language bridge
  explicitly excluded by the clarification. Deferred (could be a later hardening option).
- **JS framework + bundler (React/Vite)** — better DX for complex UIs, but adds a Node
  build step into packaging for a UI this small. Deferred enhancement.

---

## Decision 2 — JS↔Python bridge & threading model

**Decision**: A single `Api` class (in `app/bridge.py`) is mounted as `js_api`. It exposes
synchronous-style request methods (`check_readiness`, `prepare_models`, `start_live`,
`stop_session`, `transcribe_file`, `export_transcript`, `pick_audio_file`,
`pick_export_path`). Backend streaming results reach the UI as **events**: the backend
runs on its existing worker threads, and the `on_segment` / `on_error` / progress
callbacks marshal a JSON-serializable payload onto the webview via
`window.evaluate_js("window.onBackendEvent(%s)" % json)`. The UI never calls a blocking
backend method on the WebView UI thread; long operations (`prepare_models`,
`transcribe_file`) run on a Python worker thread and report progress via events.

**Rationale**: Mirrors 001's callback-based streaming (`start_session(on_segment=...)`)
and `PrepareProgress` exactly, so the bridge is a thin translation layer (dataclass →
dict → JSON). Keeps the UI responsive (SC-003, SC-011, FR-018) and preserves the
pipeline's non-blocking stage contract (Principle II). `evaluate_js` is the documented
`pywebview` push channel and is thread-safe to call from worker threads.

**Alternatives**: Polling the session from JS on a timer (simpler but laggy, fights
SC-003) — rejected. A WebSocket between UI and backend (needs a server) — rejected,
violates in-process design.

---

## Decision 3 — File import: promote `FixtureCapture` to a production `FileCapture`

**Decision**: Add `src/meeting_asr/audio/file_capture.py` with `FileCapture(AudioCapture)`:
read any `soundfile`-supported audio file, downmix to mono, resample to 16 kHz via `soxr`,
and emit `AudioFrame`s on the session clock (block-stream by default; optional `realtime`
pacing). Add a facade entry point `transcribe_file(path, *, language_hint=, on_segment=,
on_error=, progress=) -> TranscriptionSession` that wires `FileCapture` into the existing
`Pipeline`/`TranscriptionSession` and runs to completion (then `stop()`), emitting a
0..1 progress fraction. Unreadable/unsupported/empty files raise a structured `ErrorInfo`
(`code="audio.unreadable"`) instead of hanging (FR-014, US2 scenario 4).

**Rationale**: `tests/_fakes.FixtureCapture` already proves the exact mechanism
(`_load_wav_mono` + `AudioMixer.feed` + threaded frame emission). Promoting it to a real,
hardened backend gives US2 with minimal new surface and reuses the whole downstream
pipeline (diarize ∥ transcribe → fuse → session). Keeping it an `AudioCapture` honors
Principle III/VII (no pipeline change). The progress fraction is derived from
bytes/duration consumed vs. total.

**Alternatives**: A separate batch code path bypassing the streaming pipeline — rejected;
it would duplicate fusion logic and diverge from live behavior, undermining "same result
live or from file."

---

## Decision 4 — Completing the real-model inference kernels (the 001 `needs_models` gap)

**Decision**: Implement the placeholder methods against the actual published exports,
verified under `needs_models` on Apple Silicon:
- **Nemotron ONNX** (`nemotron_onnx.py`): resolve real input/output names from the
  ONNX export and the tokenizer/language head; implement `_build_inputs` (audio +
  carried encoder/conv cache + optional language conditioning), `_update_cache`,
  `_parse_tokens`, and `_decode_token` (token-id → text via the tokenizer, detected
  language from the language head, RNNT blank handling, score → confidence).
- **Sortformer CoreML** (`sortformer_coreml.py`): resolve the real `MLModel` input/output
  spec; implement `_infer_frame` (FIFO window → CoreML `predict`) and `_parse_outputs`
  (per-frame speaker-probability tensor → active raw ids → AOSC stable `Speaker N`),
  with the existing overlap/capacity → low-confidence handling.

**Rationale**: The spec's assumption is explicit — confirming correct transcription,
language detection, and speaker separation *requires running the real models*. 001 shipped
these as adaptive placeholders behind `# pragma: no cover` precisely so 002 could complete
them. The interfaces (`SpeechTranscriber`, `SpeakerDiarizer`) and the cache-aware streaming
structure are already in place; this fills in the export-specific I/O.

**Open item carried into implementation**: the exact tensor names/shapes of the pinned
`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx` and
`FluidInference/diar-streaming-sortformer-coreml` exports are confirmed at implementation
time by inspecting the downloaded model (`session.get_inputs()/get_outputs()`,
`MLModel.input_description`). The registry already pins repo + expected files.

---

## Decision 5 — Transcript export: Markdown + JSON

**Decision**: New `src/meeting_asr/export/` module. `export_markdown(segments, speakers)`
produces a human-readable, **speaker-grouped/color-legible** document (per-segment line:
`[hh:mm:ss] **Speaker N** (lang): text`, with a speaker legend header).
`export_json(segments, speakers, meta)` produces a structured document: a top-level object
with `session` metadata and a `segments` array, each segment carrying
`{segment_id, speaker_label, start, end, text, language, confidence, confidence_band,
source}` — a direct serialization of `TranscriptSegment`. Both consume the session's
chronological `transcript()` snapshot. UI exposes "Export…" → native save dialog (format
chosen by extension).

**Rationale**: Matches the clarified export formats (FR-013, SC-009). JSON is a lossless
mirror of the in-memory dataclasses (machine-readable, re-ingestable by the validation
harness); Markdown is the shareable human view. No new entities — pure projection.

**Alternatives**: SRT/VTT/CSV — out of scope for v1 (not requested); easy to add later
behind the same module.

---

## Decision 6 — Packaging: `PyInstaller --windowed` + ad-hoc `codesign`

**Decision**: Bundle with PyInstaller into `MeetingAssistant.app`. The spec
(`packaging/MeetingAssistant.spec`) collects `app/web/` static assets, the `meeting_asr`
package, the ONNX Runtime / coremltools native libs, and the built Swift `AudioTap` helper
(into `Contents/MacOS/`/`Resources`). `Info.plist` includes
`NSMicrophoneUsageDescription` and audio-capture descriptions with plain-language strings.
Post-build, `build_app.sh` runs `codesign --force --deep --sign - MeetingAssistant.app`
(ad-hoc). Models are **not** bundled — they download on first run via the existing
guided `prepare_models` flow (US3), keeping the `.app` small and the cache shared.
`make app` wraps the build.

**Rationale**: Directly meets FR-011 / the packaging clarification (self-contained,
double-click, ad-hoc signed, no developer tooling for the end user, Developer-ID +
notarization deferred). PyInstaller handles the heavy native-lib dep set (onnxruntime,
coremltools) more robustly than alias-mode py2app and is the common pairing with
`pywebview`.

**Alternatives**: `py2app` (macOS-native bundler, but more friction with the ONNX/CoreML
native libs) — documented fallback. `briefcase` — tied to the BeeWare/Toga path not
chosen in Decision 1. Both rejected for v1.

**Risk noted**: ad-hoc-signed apps with TCC-gated mic/system-audio may need the user to
approve permissions on first launch and (for an unsigned helper) possibly a Gatekeeper
right-click-open; the guided setup (US3) surfaces and explains this. Full notarization is
the deferred fix.

---

## Decision 7 — Validation datasets & metrics

**Decision**: Curate a **small** labeled subset (a handful of clips per axis), cached
locally under `tests/fixtures/validation/` with ground-truth manifests, drawn from
freely available public sources:
- **ASR / WER (clean)**: a few LibriSpeech `test-clean` utterances (public, transcript
  ground truth) — primary SC-006 (WER ≤ 15%) evidence.
- **Diarization**: 1–2 short multi-speaker clips with reference speaker turns
  (e.g. AMI/VoxConverse-style or a synthesized two/three-speaker mix with known turns) —
  SC-004 speaker-attribution accuracy.
- **Multilingual / language-ID**: a few short clips across supported languages
  (e.g. FLEURS/Common Voice samples) with known language labels — SC-005.

Metrics in `validation/metrics.py`: **WER** via `jiwer`; **diarization accuracy** as
fraction of speech-time attributed to the correct speaker label (+ a DER-style helper),
tolerant of label permutation via optimal label matching; **language-ID accuracy** as
fraction of segments with the correct detected language. `runner.py` feeds each clip
through `transcribe_file` (the real integrated pipeline), compares to ground truth, and
emits a per-clip + aggregate `ValidationReport` with pass/fail vs. thresholds; runs are
reproducible within tolerance (fixed clips, deterministic decode). `python -m validation`
prints JSON + a Markdown summary.

**Rationale**: Operationalizes FR-016/017 and SC-006/007 with objective, repeatable,
known-answer evidence, while keeping the curated set small (demonstrate correctness, not
full benchmark coverage — per the spec assumption). Exact clip selection + license check
is finalized during implementation; only a representative subset is needed. Network access
to fetch clips is a one-time setup step (mirrors model download), then fully offline.

**Alternatives**: Full benchmark suites (LibriSpeech full, DIHARD) — rejected as overkill
for v1 and too slow for a repeatable gate. Reusing only the synthetic 001 fixtures —
insufficient (they validate plumbing, not real-model accuracy on real speech).

---

## Cross-cutting notes

- **No backend topology change**: capture → mix → (diarize ∥ transcribe) → fuse → session
  is reused verbatim. 002 adds inputs (file), outputs (export), a presentation layer (app),
  and a verifier (validation) around it.
- **Offline CI preserved**: app/bridge/export/validation logic is unit/contract/integration
  tested with the existing fakes + synthetic fixtures under the network guard; real-model
  accuracy stays behind `needs_models`. The headless bridge test exercises the JS-API
  surface without opening a WKWebView window.
- **Permissions UX**: readiness already enumerates missing mic/system-audio/model items
  (`SystemReadinessReport.missing`); the setup screen renders that list with plain-language
  explanations and a retry for interrupted downloads (US3, FR-008/009/010).
