<!--
SYNC IMPACT REPORT
==================
Version Change: 1.0.0 → 2.0.0 (MAJOR — backward-incompatible redefinition of Principle IV's
  stage-coupling contract; expanded accepted inference backends in Principle V)
Amendment Rationale (feature 001-meeting-asr-backend):
  - Principle IV redefined: diarization remains first-class, runs first, and is the
    authoritative source of speaker IDs, BUT the diarizer↔transcriber coupling MAY be either
    (a) gate-by-segment feeding OR (b) concurrent execution fused by a shared session
    timeline. Required because cache-aware streaming RNNT ASR (Nemotron) depends on an
    uninterrupted audio stream; per-segment re-feeding resets its streaming cache, raising
    latency and degrading accuracy in conflict with Principle II.
  - Principle V expanded: ONNX Runtime with the CoreML execution provider is now an accepted
    Apple-Silicon inference backend alongside PyTorch MPS and native CoreML-converted models.
  - Technology Stack section: lists are now explicitly RECOMMENDED DEFAULTS, not mandates;
    capability-level principles (I–VII) remain binding. ASR default updated to NVIDIA
    Nemotron streaming ASR; diarization default updated to NVIDIA Streaming Sortformer
    (pyannote.audio retained as documented fallback).
Principles Renamed/Redefined:
  IV.  Speaker Diarization as a First-Class Stage (coupling contract broadened)
  V.   Hardware-Aware Inference Backends (ONNX Runtime CoreML EP added)
Sections Modified:
  - Technology Stack & Platform Constraints (lists marked as recommended defaults; ASR +
    diarization defaults updated)
Sections Removed: None
Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md — references constitution dynamically; no change.
  ✅ .specify/templates/spec-template.md — Generic; no change.
  ✅ .specify/templates/tasks-template.md — Generic; no change.
  ✅ .specify/templates/agent-file-template.md — Generic; no change.
Deferred TODOs: None.
-->

# ASR Meeting Minutes Constitution

## Core Principles

### I. Local-First Processing

All audio capture, speaker diarization, speech-to-text transcription, and LLM-based
summarization MUST execute entirely on the user's device. No audio data, speaker
embeddings, transcripts, or meeting content MAY be transmitted to remote servers or
cloud APIs during normal operation. Privacy is non-negotiable: the meeting stays on
the machine.

**Rationale**: Meeting content is sensitive by nature. Users MUST be able to trust
that conversations are never exposed outside their local environment, regardless of
the conferencing software in use.

### II. Real-Time Pipeline Architecture

The processing pipeline (audio capture → diarization → transcription → summarization)
MUST operate in real-time or near real-time. Transcription output MUST appear within
≤3 seconds of a speaker finishing a turn on baseline Apple Silicon hardware (M1 or
equivalent). Each pipeline stage MUST be independently runnable and MUST accept
buffered input; no stage MAY block indefinitely waiting on another stage. The LLM
summarization stage runs post-meeting or on explicit user request — not inline during
live transcription.

**Rationale**: A transcript that arrives 30 seconds after speech is unusable in a
live meeting. Decoupled, buffered stages prevent one slow model from stalling the
entire pipeline.

### III. Platform-Native Audio Interception

Audio capture MUST use platform-native APIs to intercept system audio from third-party
meeting applications (Microsoft Teams, Google Meet, Zoom, and others) without
requiring per-app integrations or fragile OS-level accessibility hacks. On macOS,
Core Audio (HAL) or AVFoundation MUST be the capture mechanism. The AudioCapture
module MUST sit behind a platform-agnostic interface so that Windows (WASAPI) and
Linux (PipeWire/ALSA) backends can be added later without modifying pipeline code.

**Rationale**: Users switch between conferencing tools. The application MUST function
regardless of which meeting software is active — no integration contract with Teams
or Google Meet should be required.

### IV. Speaker Diarization as a First-Class Stage

Speaker diarization MUST be a first-class processing stage that begins on the captured
audio and is the **authoritative source of speaker identity** for the pipeline. Every
unit of recognized speech MUST be attributed to a stable speaker ID
(Speaker 1, Speaker 2, …) derived from voice characteristics, and speaker IDs MUST
remain consistent across the entire duration of a single session. The diarization
module MUST expose a defined interface that produces a speaker timeline of
`(speaker_id, start_time, end_time)` intervals.

The coupling between diarization and transcription MAY be realized either as
**(a) gate-by-segment feeding** — the diarizer cuts audio per speaker turn and feeds
`(speaker_id, audio_segment, start_time, end_time)` to the transcriber — or
**(b) concurrent execution fused on a shared session timeline** — diarization and
transcription run in parallel on the same audio stream and their outputs are aligned by
timestamp into speaker-attributed segments. In either case diarization remains
authoritative for speaker identity and stable session-scoped IDs.

**Rationale**: Diarization quality gates transcript quality. Mixing speakers in a
single transcription segment makes meeting notes incoherent. Stable IDs within a
session are required for the LLM summarizer to attribute statements correctly.
Timestamp fusion is permitted because cache-aware streaming ASR models depend on an
uninterrupted audio stream; chopping their input per turn resets the streaming cache and
violates the real-time latency guarantee of Principle II.

### V. Hardware-Aware Inference Backends

All ML inference (diarization, ASR, LLM) MUST use the hardware acceleration backend
appropriate to the target platform. On Apple Silicon (M-series), inference MUST use one
of the accepted GPU/ANE-accelerated backends: Metal Performance Shaders (MPS) via
PyTorch, native CoreML-converted models, or ONNX Runtime with the CoreML execution
provider (`MLComputeUnits = .cpuAndGPU`); CPU-only fallback is permitted only when no
accelerated backend is available at runtime. Future backends
(NVIDIA CUDA, AMD ROCm, CPU-only) MUST be pluggable via a backend abstraction layer
without modifying pipeline logic. Model size selection MUST fit within the available
unified memory envelope of the target device tier after accounting for all concurrently
loaded models.

**Rationale**: Running large transformer models in CPU mode on a 16 GB M2 cannot
meet the ≤3s latency requirement. Hardware-awareness is not an optimization — it is
a correctness requirement for real-time operation.

### VI. Automatic Language Detection Per Speaker Turn

Transcription MUST support automatic language detection applied per speaker turn,
not per session. A meeting where participants speak different languages, or switch
languages mid-meeting, MUST produce correctly transcribed output for each turn.
The ASR module MUST accept `language=None` to trigger per-turn auto-detection.
Detected language MUST be recorded in transcript metadata alongside speaker ID,
start time, and end time.

**Rationale**: Multilingual meetings are common in international teams. A single
forced session language produces incorrect transcripts for non-primary-language
speakers and is unacceptable.

### VII. Modular, Interface-Driven Architecture

Each pipeline stage — `AudioCapture`, `SpeakerDiarizer`, `SpeechTranscriber`,
`MeetingNotesSummarizer` — MUST be implemented as an independent module with a
documented Python protocol or abstract base class defining its interface. No module
MAY import from another module's internal implementation; all inter-module
communication MUST go through the defined interface. Each module MUST be
independently testable using pre-recorded audio fixtures, without requiring a live
meeting session or a running third-party conferencing app.

**Rationale**: Interface boundaries make model swaps (e.g., Whisper medium → large-v3,
or pyannote → a future diarizer) safe and local. Testability without live audio is
required for reliable automated validation in CI.

## Technology Stack & Platform Constraints

The specific libraries and models named below are **RECOMMENDED DEFAULTS**, not mandates.
A feature MAY substitute an alternative provided it still satisfies the capability-level
Core Principles (I–VII); any substitution MUST be recorded in the feature plan's
Complexity Tracking / deviations table.

**Current Target Platform**: macOS 13 Ventura or later, Apple Silicon (M1, M2, M3,
M4 families). Intel macOS is not a primary target. Features requiring Core Audio Process
Taps target macOS 14.4+, with a ScreenCaptureKit fallback for macOS 13.0–14.3.

**Audio Capture**: Core Audio HAL / AVFoundation on macOS (including Core Audio Process
Taps). System audio tap via a process tap, `AVAudioEngine` tap, or a virtual audio
device (e.g., BlackHole) if entitlement constraints prevent HAL-level interception.

**Speaker Diarization**: NVIDIA Streaming Sortformer (CoreML) as the default online
diarizer; `pyannote.audio` retained as a documented fallback. Inference uses an accepted
Apple-Silicon backend per Principle V.

**ASR**: NVIDIA Nemotron streaming ASR (multilingual, cache-aware streaming) via ONNX
Runtime with the CoreML execution provider (FP16 default) as the recommended default;
OpenAI Whisper (`whisper.cpp` Metal / `transformers` `device="mps"`) remains an accepted
alternative for non-streaming use. Model precision/tier MUST fit the unified-memory
envelope of the target device.

**LLM Summarization**: Local inference via `ollama` (recommended) or `llama.cpp`
/ `mlx-lm`. Models MUST fit within remaining unified memory after diarization and
ASR models are loaded.

**Core Python Dependencies**: `torch` (MPS), `torchaudio`, `transformers`,
`accelerate`, `soundfile`, `pyannote.audio`.

**Future Platforms**: Windows (WASAPI audio; CUDA / DirectML backends), Linux
(PipeWire/ALSA; CUDA backends). Platform-specific audio and inference code MUST be
isolated in backend implementations — pipeline modules MUST NOT assume macOS-only APIs.

## Development Workflow & Quality Gates

**Pipeline Build Order**: Stages MUST be built and validated sequentially:
`AudioCapture` → `SpeakerDiarizer` → `SpeechTranscriber` → `MeetingNotesSummarizer`.
Each stage MUST be validated in isolation with pre-recorded audio fixtures before
the next stage is integrated.

**Audio Test Fixtures**: A reference set of multi-speaker, multi-language audio files
MUST be maintained under `tests/fixtures/audio/`. All diarization and transcription
tests MUST be reproducible using these fixtures without a live system audio tap.

**Network Isolation in Tests**: Tests MUST NOT make outbound network calls. Model
downloads are permitted during initial environment setup (`make setup` / equivalent)
and MUST be cached locally thereafter. CI MUST run fully offline after setup.

**Real-Time Performance Gate**: The end-to-end transcription pipeline MUST demonstrate
≤3 s turn-to-text latency on a baseline M1 device before any feature branch is merged
to main.

**Memory Constraint Gate**: A new ML model dependency MUST be assessed for unified
memory footprint impact on a 16 GB Apple Silicon device — with all other pipeline
models concurrently loaded — before it is adopted.

**Constitution Check in Plans**: Every feature implementation plan (`plan.md`) MUST
include a Constitution Check section identifying which principles apply and confirming
no violations exist (or documenting and justifying any approved exceptions) before
Phase 0 research begins.

## Governance

This constitution supersedes all other development practices and architectural decisions
for the ASR Meeting Minutes project. All feature design, model selection, and platform
decisions MUST be consistent with the principles above.

**Amendment Procedure**:
1. Propose the change with explicit rationale in a pull request description.
2. Identify which principle(s) are affected; update the Sync Impact Report comment.
3. Increment version per semantic versioning rules:
   - MAJOR: Principle removal, redefinition, or backward-incompatible governance change.
   - MINOR: New principle, new mandatory section, or materially expanded guidance.
   - PATCH: Wording clarification, typo fix, or non-semantic refinement.
4. Update `LAST_AMENDED_DATE` to the merge date.

**Compliance**: All `plan.md` files MUST pass a Constitution Check. All feature
specifications MUST trace functional requirements to one or more principles in this
document. Complexity violations (e.g., adding a fourth pipeline module beyond the
four defined) MUST be justified in the plan's Complexity Tracking table.

**Versioning Policy**: `MAJOR.MINOR.PATCH`. Starting version: `1.0.0`.

**Version**: 2.0.0 | **Ratified**: 2026-06-14 | **Last Amended**: 2026-06-14
