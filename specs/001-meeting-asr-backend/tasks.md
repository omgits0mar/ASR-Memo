---
description: "Task list for Realtime Diarized Meeting Transcription Backend"
---

# Tasks: Realtime Diarized Meeting Transcription Backend

**Input**: Design documents from `/specs/001-meeting-asr-backend/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: INCLUDED — the project constitution mandates per-stage testability with
pre-recorded audio fixtures and offline CI, so contract + integration tests are required.

**Organization**: Tasks are grouped by user story. US1 is the MVP.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1–US4 per spec.md; Setup/Foundational/Polish carry no story label

## Path Conventions

Single-project Python library: `src/meeting_asr/`, `tests/`, native helper in `native/AudioTap/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project skeleton, dependencies, tooling

- [X] T001 Create the package layout per plan.md: `src/meeting_asr/{audio,diarization,asr,fusion,models,backends}/__init__.py`, `tests/{unit,integration,contract,fixtures/audio}/`, and `native/AudioTap/`
- [X] T002 Expand `requirements.txt` and add `pyproject.toml` with deps: `onnxruntime`, `coremltools`, `huggingface_hub`, `sounddevice`, `numpy`, `soxr`, `soundfile`, `pytest`; reference extras `nemo_toolkit[asr]`, `torch`
- [X] T003 [P] Create the Swift package skeleton in `native/AudioTap/Package.swift` (executable target `AudioTap`, macOS 14.4 platform) and a stub `native/AudioTap/Sources/AudioTap/main.swift`
- [X] T004 [P] Configure tooling: `ruff`/`black` config, `pytest.ini`/`pyproject` test settings, and a Makefile (`setup`, `build-native`, `test`) per quickstart.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared types, protocols, hardware/model infra, and the test harness that ALL stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Define all enums + dataclasses in `src/meeting_asr/types.py` (AudioSourceKind, SessionStatus, CaptureState, ModelKind, ModelState; TranscriptSegment, Speaker, AudioSource, ModelAsset, SystemReadinessReport; AudioFrame, DiarFrame, AsrToken, ErrorInfo, PrepareProgress) per data-model.md
- [X] T006 [P] Implement the hardware-aware backend resolver in `src/meeting_asr/backends/device.py` (CoreML EP with `MLComputeUnits=.cpuAndGPU` → CPU EP / MPS fallback; expose `compute_backend` string) per research Decision 5
- [X] T007 [P] Define the `AudioCapture` protocol + `AudioFrame` contract in `src/meeting_asr/audio/capture.py` per contracts/audio_capture.md
- [X] T008 [P] Define the `SpeakerDiarizer` protocol in `src/meeting_asr/diarization/diarizer.py` per contracts/speaker_diarizer.md
- [X] T009 [P] Define the `SpeechTranscriber` protocol (with `load(backend, precision="fp16")`) in `src/meeting_asr/asr/transcriber.py` per contracts/speech_transcriber.md
- [X] T010 [P] Implement `ModelAsset` definitions + `prepare()` download/cache (resumable `huggingface_hub`, integrity check, no corrupt cache) in `src/meeting_asr/models/registry.py` (Nemotron FP16 ONNX + Sortformer CoreML repos/revisions)
- [X] T011 [P] Implement logging/error infrastructure (`ErrorInfo` factory, structured logger) in `src/meeting_asr/_logging.py`
- [X] T012 Implement `src/meeting_asr/audio/mixer.py`: resample any input to 16 kHz mono float32, single monotonic session clock, single-source passthrough (multi-source merge added in US2) — depends on T005, T007
- [X] T013 [P] Create the test harness in `tests/conftest.py`: fixture loader for `tests/fixtures/audio/`, and a network-isolation guard that fails any test making outbound calls (constitution offline-CI gate)
- [X] T014 [P] Add baseline single-/two-speaker English fixture clips to `tests/fixtures/audio/` with reference speaker turns + transcripts

**Checkpoint**: Types, protocols, device resolver, model registry, mixer, and offline test harness ready

---

## Phase 3: User Story 1 - Live diarized transcription of a single source (Priority: P1) 🎯 MVP

**Goal**: Capture the mic, diarize in real time, transcribe with Nemotron, and stream out speaker-labeled, timestamped segments via the in-process API; query and stop.

**Independent Test**: Run a session against a mic fixture; assert streamed `TranscriptSegment`s carry speaker label + start/end + text, a second voice gets "Speaker 2", `transcript()` returns ordered segments, and `stop()` returns the final transcript.

### Tests for User Story 1 ⚠️ (write first, ensure they fail)

- [X] T015 [P] [US1] Contract test for `SpeakerDiarizer` (Sortformer): stable arrival-order labels, `reset()`, 80ms `DiarFrame`s in `tests/contract/test_diarizer.py`
- [X] T016 [P] [US1] Contract test for `SpeechTranscriber` (Nemotron): streaming tokens, timestamps, `flush()`, FP16 default in `tests/contract/test_transcriber.py`
- [X] T017 [P] [US1] Contract test for the public API (`start_session`/`segments`/`transcript`/`stop`, structured results, `SessionBusyError`) in `tests/contract/test_public_api.py`
- [X] T018 [P] [US1] Integration test: mic fixture → streamed labeled segments + ordering + final transcript in `tests/integration/test_us1_single_source.py`

### Implementation for User Story 1

- [X] T019 [P] [US1] Implement `MicrophoneCapture` (PortAudio via `sounddevice`, resample→16 kHz mono, `CaptureState`, permission/device errors) in `src/meeting_asr/audio/microphone.py`
- [X] T020 [P] [US1] Implement `SortformerCoreMLDiarizer` (CoreML load via resolver, low-latency profile chunk6/rc7/FIFO188, AOSC stable labels, `push`/`reset`/`max_speakers`) in `src/meeting_asr/diarization/sortformer_coreml.py`
- [X] T021 [P] [US1] Implement `NemotronOnnxTranscriber` (ONNX Runtime CoreML EP FP16 `.cpuAndGPU`, 560ms cache-aware streaming, `push`/`flush`, `precision` knob, per-token language) in `src/meeting_asr/asr/nemotron_onnx.py`
- [X] T022 [US1] Implement `fusion/aligner.py`: align `DiarFrame` timeline with `AsrToken` stream → `TranscriptSegment`s (dominant-overlap speaker, carry language/score, alignment buffer ~diarization latency) — depends on T020, T021
- [X] T023 [US1] Implement `TranscriptionSession` state machine + segment delivery (callback + blocking iterator), `transcript()`, `speakers()`, `stop()` in `src/meeting_asr/session.py` — depends on T005
- [X] T024 [US1] Implement `pipeline.py` orchestration: capture → (diarize ∥ transcribe concurrently) → fuse, with buffering + backpressure handling (no dropped audio, coherent timestamps); on compute/memory pressure, degrade gracefully (buffer/signal lag) and surface the constraint to the consumer rather than dropping audio (FR-021) — depends on T012, T019–T023
- [X] T025 [US1] Implement the public facade in `src/meeting_asr/__init__.py`: `start_session(...)` wiring (mic source default for MVP), sequential-session `SessionBusyError` guard, mic `PermissionError` with actionable hint — depends on T024

**Checkpoint**: MVP — a mic-only live diarized, transcribed session works end-to-end and passes US1 tests independently

---

## Phase 4: User Story 2 - Capture remote meeting participants (Priority: P2)

**Goal**: Add macOS system-audio capture (Core Audio Process Taps) so meeting-app voices merge with the mic into one diarized timeline.

**Independent Test**: With mic + system-audio fixtures (and a recorded tap dump), a `sources=(MIC, SYSTEM)` session produces one merged time-ordered transcript with ≥1 remote speaker; revoking system-audio permission yields a clear error.

### Tests for User Story 2 ⚠️

- [X] T026 [P] [US2] Integration test: merged mic+system timeline, 3+ distinct speakers, overlap handling, and permission-denied error in `tests/integration/test_us2_system_audio.py`
- [X] T027 [P] [US2] Integration test: recorded tap dump → `CoreAudioTapCapture` subprocess → 16 kHz mono frames in `tests/integration/test_audiotap_helper.py`

### Implementation for User Story 2

- [X] T028 [US2] Implement the Swift Process-Tap helper in `native/AudioTap/Sources/AudioTap/main.swift` (`AudioHardwareCreateProcessTap` + `CATapDescription`, emit raw PCM to stdout, permission handling) — depends on T003
- [X] T029 [US2] Implement `CoreAudioTapCapture` in `src/meeting_asr/audio/coreaudio_tap.py`: spawn the helper, read PCM pipe, resample→16 kHz mono, tag `source=SYSTEM`, surface `CapturePermissionError` — depends on T028, T007
- [X] T030 [P] [US2] Implement `ScreenCaptureKitCapture` fallback (macOS 13.0–14.3) in `src/meeting_asr/audio/screencapturekit.py`
- [X] T031 [US2] Extend `audio/mixer.py` for multi-source merge on the shared clock with per-frame `source` tags (FR-009) — depends on T012
- [X] T032 [US2] Wire `sources=(MIC, SYSTEM)` default in the facade + system-audio permission detection/actionable error in `src/meeting_asr/__init__.py` and `models/readiness.py` — depends on T025, T029, T031

**Checkpoint**: Both local + remote participants captured and merged; US1 still passes

---

## Phase 5: User Story 3 - Multilingual transcription with per-segment language (Priority: P3)

**Goal**: Per-turn language handling — each segment transcribed in its spoken language with a detected language tag; optional session language hint.

**Independent Test**: Feed multilingual fixtures (two languages across speakers, and code-switching); assert correct per-segment `language` tags and that a `language_hint` biases without forcing one language.

### Tests for User Story 3 ⚠️

- [X] T033 [P] [US3] Integration test: per-segment language tags, two-language speakers, and `language_hint` behavior in `tests/integration/test_us3_multilingual.py`; assert per-segment language-ID accuracy ≥95% against the reference languages on the multilingual fixtures (SC-003)
- [X] T034 [P] [US3] Add multilingual + code-switching fixture clips (with reference languages) to `tests/fixtures/audio/`

### Implementation for User Story 3

- [X] T035 [US3] Plumb `language_hint` from `start_session` → `pipeline` → `SpeechTranscriber.push(..., language_hint=)` in `src/meeting_asr/session.py` and `src/meeting_asr/pipeline.py` — depends on T024
- [X] T036 [US3] Propagate per-token language → `TranscriptSegment.language` and flag unknown/unsupported language as LOW/UNKNOWN confidence band in `src/meeting_asr/fusion/aligner.py` (FR-018) — depends on T022

**Checkpoint**: Multilingual meetings transcribed per turn; US1–US2 still pass

---

## Phase 6: User Story 4 - Model lifecycle & readiness API (Priority: P3)

**Goal**: Clean programmatic surface to prepare/cache models, check readiness, and drive the full session lifecycle.

**Independent Test**: From a script: `prepare_models()` (progress + resumable), `check_readiness()` enumerates missing items, then start→segments→stop; second run loads from cache in <30s with no network.

### Tests for User Story 4 ⚠️

- [X] T037 [P] [US4] Contract test for `prepare_models`/`check_readiness` (progress callbacks, `missing[]`, cached fast-path, interrupted-download safety) in `tests/contract/test_lifecycle.py`
- [X] T038 [P] [US4] Integration test: full lifecycle prepare→readiness→start→segments→stop in `tests/integration/test_us4_lifecycle.py`

### Implementation for User Story 4

- [X] T039 [US4] Implement `prepare_models(progress=, force=)` with progress reporting + resumable/atomic download (no corrupt cache) in `src/meeting_asr/models/registry.py` and expose in `__init__.py` — depends on T010
- [X] T040 [US4] Implement `check_readiness()` full assembly (model states, mic + system-audio permissions, resolved compute backend, `os_supports_process_tap`, `missing[]`) in `src/meeting_asr/models/readiness.py` — depends on T006, T010
- [X] T041 [US4] Implement cached-load fast path (<30s, no re-download) + multi-session sequential management over backend lifetime in `src/meeting_asr/__init__.py` (FR-012, FR-020, SC-005) — depends on T039, T040

**Checkpoint**: Full lifecycle drivable via the documented API; all stories pass

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Performance gates, quantization gate, docs, prototype retirement

- [X] T042 [P] Implement the quantization gate harness in `tests/integration/test_quantization_gate.py`: measure WER per language band + turn-to-text latency for `precision` fp16/int8/int4 on fixtures; assert INT8/INT4 only "pass" at ≤1% absolute WER regression vs FP16 + latency win (research Decision 8)
- [X] T043 [P] Implement performance gate in `tests/integration/test_performance.py`: turn-to-text ≤3s (target ~1.5–2.0s) and a 60-min soak test for timeline/label/order stability (SC-001, SC-008); include a compute-pressure case asserting graceful degradation + lag signaling with no dropped audio/timeline corruption (FR-021)
- [X] T048 [P] Implement the diarization-accuracy gate in `tests/integration/test_diarization_accuracy.py`: measure speaker-attribution accuracy (DER / correct-attributed speech time) against labeled multi-speaker fixtures (up to 4 speakers); assert ≥90% of speech time attributed to the correct, stable speaker label (SC-002)
- [X] T044 [P] Add unit tests for `mixer`, `aligner`, `device` resolver, and `registry` in `tests/unit/`
- [X] T045 [P] Update `README`/`CLAUDE.md` usage and verify `quickstart.md` steps end-to-end
- [X] T046 Retire/migrate the legacy `transcribe_meeting.py` Whisper prototype (note supersession or remove)
- [X] T047 Verify offline CI: full `pytest` run with the network-isolation guard active after one-time setup

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **User Stories (Phase 3–6)**: All depend on Foundational
  - US1 (P1) is the MVP and should be completed first
  - US2/US3/US4 build on US1's session/pipeline/facade; sequential by priority is recommended, though US3 and US4 are largely independent of US2
- **Polish (Phase 7)**: Depends on the targeted stories being complete

### User Story Dependencies

- **US1 (P1)**: Foundational only — self-contained MVP
- **US2 (P2)**: Extends US1 (facade, mixer, pipeline) — independently testable via fixtures/tap dump
- **US3 (P3)**: Extends US1 (session/pipeline/aligner) — independent of US2
- **US4 (P3)**: Mostly Foundational + facade — independent of US2/US3

### Within Each User Story

- Tests written first and failing → backends (models) → fusion → session/pipeline → facade
- Story complete and passing before moving to the next priority

### Parallel Opportunities

- Setup: T003, T004 in parallel
- Foundational: T006–T011 and T013–T014 in parallel (after T005); T012 after T005+T007
- US1: tests T015–T018 in parallel; backends T019–T021 in parallel; then T022 → T023 → T024 → T025
- US2: T026/T027 in parallel; T030 parallel to T028/T029
- US3: T033/T034 in parallel
- US4: T037/T038 in parallel
- Polish: T042–T045 and T048 in parallel

---

## Parallel Example: User Story 1

```bash
# Tests first (parallel):
Task: "Contract test SpeakerDiarizer in tests/contract/test_diarizer.py"
Task: "Contract test SpeechTranscriber in tests/contract/test_transcriber.py"
Task: "Contract test public API in tests/contract/test_public_api.py"
Task: "Integration test single source in tests/integration/test_us1_single_source.py"

# Then backends (parallel, different files):
Task: "MicrophoneCapture in src/meeting_asr/audio/microphone.py"
Task: "SortformerCoreMLDiarizer in src/meeting_asr/diarization/sortformer_coreml.py"
Task: "NemotronOnnxTranscriber in src/meeting_asr/asr/nemotron_onnx.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1: Setup
2. Phase 2: Foundational (CRITICAL — blocks all stories)
3. Phase 3: US1 → **STOP and VALIDATE** (mic-only live diarized transcription)
4. Demo the MVP

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 → test → demo (MVP: mic transcription with speaker labels)
3. US2 → test → demo (meeting-app participants merged in)
4. US3 → test → demo (multilingual per turn)
5. US4 → test → demo (clean lifecycle/readiness API for the future UI)
6. Polish: enforce performance + quantization gates before merge

---

## Notes

- [P] = different files, no dependencies on incomplete tasks
- Constitution gates apply: offline CI after setup, ≤3s turn-to-text, 16 GB memory envelope, FP16 default with quantization gated on measured WER+latency
- The diarizer↔ASR parallel-fusion topology (not gate-by-segment) is intentional — see plan.md Complexity Tracking
- Commit after each task or logical group; validate at each story checkpoint
