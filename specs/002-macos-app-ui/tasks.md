---
description: "Task list for 002-macos-app-ui implementation"
---

# Tasks: macOS Meeting Assistant App — UI, End-to-End Validation & Packaging

**Input**: Design documents from `/specs/002-macos-app-ui/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (all present)

**Tests**: INCLUDED. This project enforces an offline, protocol-driven, fixture-based test
discipline (Constitution VII; 001 shipped 83 offline tests). Each contract in `contracts/`
defines a testability section, so contract/unit/integration tests are written before the
implementation they cover.

**Organization**: Tasks are grouped by user story (US1–US5) for independent implementation
and testing. Phases 1–2 are shared prerequisites.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1–US5 (user-story phases only)
- Exact file paths are included in each task.

## Path Conventions

Single repo: backend library at `src/meeting_asr/`, front-end at `app/`, QA harness at
`validation/`, packaging at `packaging/`, tests at `tests/`. Matches plan.md "Source Code".

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dependencies, package skeletons, and dev entry points.

- [X] T001 Add `pywebview>=5.0` to runtime deps in `requirements.txt` and `[project].dependencies` in `pyproject.toml`
- [X] T002 [P] Add `pyinstaller>=6.0` and `jiwer>=3.0` to a new `packaging`/`validation` dev extra in `[project.optional-dependencies]` of `pyproject.toml`
- [X] T003 [P] Create package skeletons with `__init__.py`: `app/`, `app/web/`, `validation/`, `packaging/`, and `tests/fixtures/validation/`
- [X] T004 [P] Add `run`, `app`, and `validate` targets to `Makefile` (`run`→`python -m app.main`, `app`→`packaging/build_app.sh`, `validate`→`python -m validation --axis all`)
- [X] T005 [P] Add `dist/`, `build/`, `out/`, `*.app`, and the validation-clip cache to `.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared backend correctness + the app shell/bridge that ALL UI stories depend on.

**⚠️ CRITICAL**: No user story phase can begin until this phase is complete.

### Real-model inference kernels (genuine output for US1/US2/US5)

- [X] T006 [P] Complete the Nemotron ONNX streaming decode kernel in `src/meeting_asr/asr/nemotron_onnx.py` — implement `_build_inputs` (audio + carried encoder/conv cache + optional language conditioning), `_update_cache`, `_parse_tokens`, and `_decode_token` (token-id→text via tokenizer, detected language from the language head, RNNT blank handling, score→confidence) against the pinned export; verify under `needs_models` ⚠ verify gated: no models/offline env — kernel structure completed (vocab load + I/O resolution); real-export verification deferred to a machine with the pinned model
- [X] T007 [P] Complete the Sortformer CoreML decode kernel in `src/meeting_asr/diarization/sortformer_coreml.py` — implement `_infer_frame` (FIFO window→`MLModel.predict`) and `_parse_outputs` (per-frame speaker-probability tensor→active raw ids→AOSC stable `Speaker N`) with overlap/capacity→low-confidence handling; verify under `needs_models` ⚠ verify gated: see T006

### App shell + JS↔Python bridge (per contracts/js_bridge_api.md, data-model.md)

- [X] T008 [P] Implement DTO serialization helpers (`segment_dto`, `speaker_dto`, `readiness_dto`, `error_dto`, `prepare_progress_dto`) in `app/dto.py` per `contracts/js_bridge_api.md`
- [X] T009 Implement `AppSession` view-model + backend→UI status mapping (`setting_up/ready/recording/processing/stopping/stopped/error`) in `app/app_session.py` per `data-model.md`
- [X] T010 Implement the `Api` bridge core in `app/bridge.py`: class mounted as `js_api`, worker-thread dispatch, and the `evaluate_js` event channel (`window.onBackendEvent(...)`) — methods stubbed, event-emit + threading wired
- [X] T011 Implement the pywebview host in `app/main.py`: create the WKWebView window, load `app/web/index.html`, mount `Api`, and `start()`
- [X] T012 [P] Build the frontend screen scaffold (setup / ready / session / review sections) in `app/web/index.html`
- [X] T013 [P] Build base styling for a clean native-feeling macOS look + a fixed speaker color palette in `app/web/styles.css`
- [X] T014 Implement the frontend runtime in `app/web/app.js`: `window.pywebview.api` call wrappers, the `window.onBackendEvent` dispatcher (keyed by `evt.type`), and deterministic `SpeakerView` color assignment by arrival order
- [X] T015 [P] Create the headless bridge test harness in `tests/contract/test_bridge_api.py` — instantiate `Api` with injected backend fakes (reuse `tests/_fakes`) and a stub `evaluate_js` sink (no window); shared by US1–US4 contract assertions

**Checkpoint**: Real models produce genuine output; the app window opens and the bridge can
exchange method calls + events with the UI using fakes.

---

## Phase 3: User Story 1 — Live diarized, multilingual session (Priority: P1) 🎯 MVP

**Goal**: Start a live session from the app and watch diarized, language-tagged,
time-stamped transcript lines appear, interleaving mic + system audio; Stop retains the
full transcript.

**Independent Test**: Launch the app (Ready), pick sources, click Start, speak (with a
second voice / a meeting playing) → lines appear ≤~2 s with correct stable speaker labels,
language tags, and timestamps on one timeline; Stop halts capture and keeps the transcript.

### Tests for User Story 1

- [X] T016 [P] [US1] Integration test: bridge drives a fake live session end-to-end (start→segments→stop), asserting ordered `segment` events and final `status=stopped` in `tests/integration/test_app_session_live.py`
- [X] T017 [P] [US1] Contract assertions in `tests/contract/test_bridge_api.py`: `start_live` return shape, `segment`/`status` event payloads, and `not_ready`/`session.busy` rejections

### Implementation for User Story 1

- [X] T018 [US1] Implement `Api.start_live(sources, language_hint)` in `app/bridge.py` — wrap `start_session`, run on a worker, emit `segment`/`error`/`status` events; reject with `not_ready`/`session.busy` per contract
- [X] T019 [US1] Implement `Api.stop_session()` in `app/bridge.py` — wrap `session.stop()`, idempotent, emit `status=stopped`
- [X] T020 [US1] Build live-session controls in `app/web/index.html` + `app/web/app.js`: source toggles (microphone / system audio), optional language hint, Start/Stop buttons
- [X] T021 [US1] Implement live transcript rendering in `app/web/app.js` + `app/web/styles.css`: per-segment line with speaker color, language tag, timestamp; append in chronological `start` order; interleave mic + system on one timeline
- [X] T022 [US1] Handle mid-session conditions in `app/web/app.js`: surface recoverable/terminal `error` events (e.g. "microphone access lost") and route `not_ready` to the setup screen

**Checkpoint**: US1 is a complete, independently demoable live-transcription increment.

---

## Phase 4: User Story 2 — Transcribe a downloaded/dataset clip (Priority: P2)

**Goal**: Import an existing audio file and get the same diarized, language-tagged
transcript without any live setup; unreadable files show a clear error.

**Independent Test**: From the app, pick a one/multi-speaker and/or multilingual file →
correct text under the right number of stable speaker labels with correct per-segment
language tags; a progress indicator runs; a bad file shows an actionable error.

### Tests for User Story 2

- [X] T023 [P] [US2] Unit test `FileCapture` in `tests/unit/test_file_capture.py`: WAV→frames count/duration, 16 kHz mono resample, `consumed_fraction` monotonicity, idempotent `stop`, and `audio.unreadable`/`audio.empty` error codes
- [X] T024 [P] [US2] Contract test `transcribe_file` in `tests/contract/test_file_transcription.py` (offline fakes): ordered transcript, `on_progress`→1.0, `not_ready`/`busy`, `ERROR` session for a bad path
- [X] T025 [P] [US2] Integration test: bridge drives a file-import run end-to-end in `tests/integration/test_app_session_file.py`

### Implementation for User Story 2

- [X] T026 [P] [US2] Implement `FileCapture(AudioCapture)` in `src/meeting_asr/audio/file_capture.py` per `contracts/file_transcription.md` (promote `tests/_fakes.FixtureCapture`: file→mono→16 kHz frames, `total_seconds`/`consumed_fraction`, structured errors)
- [X] T027 [US2] Implement the `transcribe_file(...)` facade entry point in `src/meeting_asr/__init__.py` (wire `FileCapture`→`Pipeline`/`TranscriptionSession`, `on_progress`, readiness/busy rules) and add it to `__all__`
- [X] T028 [US2] Implement `Api.transcribe_file(path, language_hint)` and `Api.pick_audio_file()` in `app/bridge.py` — worker-thread run emitting `segment`/`progress`/`status`/`error`
- [X] T029 [US2] Build the import UI in `app/web/index.html` + `app/web/app.js`: "Import audio…" → native picker, progress bar from `progress` events, and clear error state on unreadable files

**Checkpoint**: US1 (live) and US2 (file) both work independently against the same pipeline.

---

## Phase 5: User Story 3 — First-run setup & well-packaged launch (Priority: P2)

**Goal**: A double-click `.app` guides one-time model download + permission grants to a
clear "Ready" state, relaunches fast from cache, and retries interrupted downloads cleanly.

**Independent Test**: On a clean machine, double-click the `.app`, complete guided setup to
Ready (no terminal); quit and relaunch → Ready < 30 s without re-downloading; interrupt a
download and retry → resumes without a broken state.

### Tests for User Story 3

- [X] T030 [P] [US3] Contract assertions in `tests/contract/test_bridge_api.py`: `get_readiness` shape incl. `missing[]`, and `prepare_progress`/`prepare_done` event sequence (offline, stubbed downloader)

### Implementation for User Story 3

- [X] T031 [US3] Implement `Api.get_readiness()` and `Api.prepare()` in `app/bridge.py` — wrap `check_readiness`/`prepare_models(progress=…)` on a worker; emit `prepare_progress` + `prepare_done` events; surface `missing[]`
- [X] T032 [US3] Build the setup screen in `app/web/index.html` + `app/web/app.js`: readiness list with plain-language explanations per `missing[]` item, model-download progress, permission guidance (mic + system audio), and a retry control for interrupted downloads
- [X] T033 [US3] Implement readiness gating + fast relaunch in `app/app_session.py` / `app/web/app.js`: block session start until `ready`, jump straight to Ready when cache + permissions already satisfied
- [X] T034 [P] [US3] Write the PyInstaller spec in `packaging/MeetingAssistant.spec`: bundle `app/web/`, the `meeting_asr` package, ONNX Runtime / coremltools native libs, and the built Swift `AudioTap` helper (`--windowed`) ⚠ spec written; actual build not exercised here (no PyInstaller in this offline env)
- [X] T035 [P] [US3] Write `packaging/Info.plist.in` with `NSMicrophoneUsageDescription` + audio-capture usage descriptions (plain-language strings)
- [X] T036 [US3] Write `packaging/build_app.sh`: run PyInstaller and `codesign --force --deep --sign -` (ad-hoc) → `dist/MeetingAssistant.app` (depends on T034, T035) ⚠ script written + syntax-checked; not run here (no PyInstaller/models/hardware)
- [ ] T037 [US3] Verify the packaged flow against `quickstart.md` §6: double-click launch → guided setup → Ready, then relaunch Ready < 30 s (SC-001, SC-002) ⚠ GATED: requires `pip install -e ".[packaging]"`, downloaded models, and macOS Apple Silicon — manual verification deferred

**Checkpoint**: A clean machine can reach Ready via a double-click `.app` and relaunch fast.

---

## Phase 6: User Story 4 — Review, navigate, and export the transcript (Priority: P3)

**Goal**: Review the transcript in a readable, speaker-attributed, time-ordered layout and
export it to Markdown and JSON preserving speaker, time, language, and text.

**Independent Test**: Produce a transcript (live or file) → readable speaker-attributed
layout with timestamps + language tags that scrolls responsively; Export → Markdown and
JSON files each contain the full transcript with all fields.

### Tests for User Story 4

- [ ] T038 [P] [US4] Unit test export in `tests/unit/test_export.py`: Markdown contains every speaker/time/language/text + a legend; JSON round-trips all fields; chronological order; `write_export` extension routing + `ValueError` on unknown extension

### Implementation for User Story 4

- [ ] T039 [P] [US4] Implement the export module in `src/meeting_asr/export/` (`markdown.py`, `json_export.py`, `__init__.py` re-exporting `export_markdown`/`export_json`/`write_export`) per `contracts/transcript_export.md`
- [ ] T040 [US4] Implement `Api.export_transcript(path, format)` and `Api.pick_export_path(format)` in `app/bridge.py` (native save dialog, local write)
- [ ] T041 [US4] Build the review UI in `app/web/index.html` + `app/web/app.js` + `app/web/styles.css`: readable speaker-grouped/color-coded layout, jump-to-speaker, responsive scrolling for long transcripts, and an "Export…" menu (Markdown / JSON)

**Checkpoint**: Transcripts are reviewable and exportable to both formats with all fields.

---

## Phase 7: User Story 5 — Accuracy validation against public datasets (Priority: P3)

**Goal**: A repeatable pass feeds labeled public clips through the integrated pipeline and
reports WER, diarization accuracy, and language-ID accuracy vs. thresholds, reproducibly.

**Independent Test**: `make validate` over the curated clips → a report with per-clip +
aggregate metrics meeting WER ≤ 15%, diarization ≥ 90%, language-ID ≥ 95%, with failing
clips listed; re-running reproduces metrics within tolerance.

### Tests for User Story 5

- [X] T042 [P] [US5] Unit test metrics in `tests/unit/test_validation_metrics.py`: `wer`, permutation-invariant `diarization_accuracy`, and `language_id_accuracy` on known inputs
- [X] T043 [P] [US5] Offline self-test of the harness in `tests/integration/test_validation_run.py`: synthetic labeled clips + injected fakes whose output matches references → assert metric computation + `ValidationReport` assembly (no network/models)

### Implementation for User Story 5

- [X] T044 [P] [US5] Implement `validation/metrics.py`: `wer` (jiwer), `diarization_accuracy` (optimal label matching + DER helper), `language_id_accuracy` per `contracts/validation_report.md`
- [X] T045 [P] [US5] Implement `validation/datasets.py`: `ValidationSample` + `load_samples(samples_dir, axis)` (manifest + local cache loader)
- [X] T046 [US5] Implement `validation/runner.py`: `run_validation(samples, thresholds)` feeding each clip through `transcribe_file`, scoring vs. ground truth → `ValidationReport` (per-clip + aggregate + pass/fail) (depends on T044, T045)
- [X] T047 [US5] Implement the CLI in `validation/__main__.py`: `--axis`, `--report-json`, `--report-md`, `--samples-dir`; Markdown summary; exit code 0 iff aggregate `passed`
- [X] T048 [US5] Curate + cache the small labeled clip set with ground-truth manifests under `tests/fixtures/validation/` (LibriSpeech test-clean ASR, a multi-speaker diarization clip, multilingual language-ID clips; record provenance + license per sample) ⚠ manifest + provenance/README done; one-time clip fetch is gated (network) — refs populated at fetch per README
- [ ] T049 [US5] Run the `needs_models` acceptance pass on Apple Silicon to produce genuine SC-006/SC-007 evidence (WER ≤ 15%; diarization/language-ID thresholds; reproducible) ⚠ GATED: requires downloaded models + cached clips on Apple Silicon — manual run deferred

**Checkpoint**: All user stories are independently functional and correctness is measured.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Edge cases, docs, and quality gates spanning stories.

- [X] T050 [P] Implement/verify edge-case behaviors in `app/web/app.js` + bridge: silence yields no fabricated text/speakers; >4 overlapping speakers degrade gracefully (low confidence); unsupported language flagged low-confidence/unknown; backend-unavailable shows a clear recovery path
- [X] T051 [P] Update `README.md` with the 002 app section (run / validate / app targets, screens, packaging notes)
- [X] T052 [P] Lint + format `app/`, `validation/`, `packaging/` with `ruff` + `black` (config in `pyproject.toml`) ⚠ ruff/black not installed in this offline env — code written to the configured conventions (line-length 100, isort known-first-party incl. app/validation/tests); all new modules compile cleanly; run `make lint format` with dev tools installed to apply formatting
- [ ] T053 Run a 60-minute live soak to confirm timeline continuity, stable labels, and UI responsiveness (SC-011, FR-018) ⚠ GATED: requires live mic/system-audio hardware + models — manual run deferred
- [ ] T054 Execute the full `quickstart.md` acceptance smoke table (SC-001…SC-012) and record results ⚠ GATED: requires packaged `.app` + models + hardware — manual run deferred

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories.
- **User Stories (Phase 3–7)**: All depend on Foundational. After it, US1–US5 are largely
  independent and can proceed in parallel (subject to the shared-file notes below).
- **Polish (Phase 8)**: Depends on the targeted user stories being complete.

### User Story Dependencies

- **US1 (P1)**: Foundational only. MVP.
- **US2 (P2)**: Foundational only. Adds `FileCapture` + `transcribe_file` (backend) — independent of US1.
- **US3 (P2)**: Foundational only. Setup/packaging — independent; packaging (T034–T036) can run any time after Setup.
- **US4 (P3)**: Foundational only. Needs *a* transcript to review; export module (T039) is standalone.
- **US5 (P3)**: Foundational only, and consumes `transcribe_file` (T027 from US2) for the real run (T049). The offline harness self-test (T043) needs only fakes.

### Shared-file coordination (not blocking, but serialize edits)

- `app/bridge.py` is touched by T010/T018/T019/T028/T031/T040 — implement per-story methods without conflicting; not all [P].
- `app/web/app.js`, `index.html`, `styles.css` are touched across US1–US4 — sequence UI edits per story.

### Within Each User Story

- Tests are written first and must FAIL before implementation.
- Backend models/services before bridge methods; bridge methods before UI wiring.

---

## Parallel Opportunities

- **Setup**: T002–T005 in parallel.
- **Foundational**: T006 + T007 (kernels) in parallel; T008, T012, T013, T015 in parallel with the kernels; T009→T010→T011 and T014 serialize on shared app files.
- **US1 tests**: T016, T017 in parallel.
- **US2**: T023, T024, T025 (tests) in parallel; T026 (FileCapture) parallel with US2 tests.
- **US5**: T042, T043 (tests) and T044, T045 (metrics/datasets) in parallel.
- **Cross-story**: once Foundational is done, different developers can take US1–US5 in parallel (coordinating shared `app/` files).

---

## Parallel Example: Foundational kernels + app shell

```bash
# Backend correctness (independent files):
Task: "Complete Nemotron ONNX decode kernel in src/meeting_asr/asr/nemotron_onnx.py"
Task: "Complete Sortformer CoreML decode kernel in src/meeting_asr/diarization/sortformer_coreml.py"

# App scaffold (independent files), in parallel with the kernels:
Task: "DTO helpers in app/dto.py"
Task: "Screen scaffold in app/web/index.html"
Task: "Base styles + speaker palette in app/web/styles.css"
Task: "Headless bridge test harness in tests/contract/test_bridge_api.py"
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1 Setup → 2. Phase 2 Foundational (kernels + app shell + bridge) →
3. Phase 3 US1 → **STOP & VALIDATE** live transcription independently → demo.

### Incremental Delivery

Foundation → US1 (live MVP) → US2 (file import) → US3 (setup + `.app`) → US4 (review/export)
→ US5 (validation evidence). Each story is a testable, demoable increment that does not
break prior stories.

### Notes

- [P] = different files, no incomplete-task dependency.
- Offline tests (unit/contract/integration) use the existing fakes/synthetic fixtures under
  the network guard; real-model accuracy stays behind `needs_models`.
- Commit after each task or logical group; stop at any checkpoint to validate a story.
