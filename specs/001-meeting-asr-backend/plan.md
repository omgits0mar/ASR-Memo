# Implementation Plan: Realtime Diarized Meeting Transcription Backend

**Branch**: `001-meeting-asr-backend` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-meeting-asr-backend/spec.md`

## Summary

Build the on-device, backend-only pipeline that captures a meeting's audio (local
microphone + any meeting app's system output), separates speakers in real time, and
produces a multilingual, time-stamped, speaker-labeled transcript — exposed as an
**in-process Python library**. Optimized for the lowest-latency path on macOS Apple
Silicon: **Core Audio Process Taps** for capture, **Streaming Sortformer (CoreML)** for
diarization, and **Nemotron 3.5 ASR Streaming (FP16 ONNX via ONNX Runtime CoreML EP on the
M-series GPU + CPU)** for transcription, with diarization and ASR running **concurrently and
fused by timestamp** to keep turn-to-text latency at ~1.5–2.0s (within the constitution's
≤3s gate). ASR precision is FP16 by default; INT8/INT4 are opt-in only behind a measured
WER+latency gate.

## Technical Context

**Language/Version**: Python 3.11 (library + pipeline); Swift 5.9+ (Core Audio Process-Tap capture helper)  
**Primary Dependencies**: `onnxruntime` (CoreML EP), `coremltools`, `huggingface_hub`, `sounddevice` (PortAudio), `numpy`, `soxr`/`samplerate`, `soundfile`; native Swift Process-Tap helper. Reference-only: `nemo_toolkit[asr]`, `torch` (MPS)  
**Storage**: Local filesystem only — model cache (HF cache dir) and optional transcript/audio fixture files; no database  
**Testing**: `pytest` with pre-recorded multi-speaker/multi-language audio fixtures (`tests/fixtures/audio/`); offline (no network) after setup; Swift helper via integration test against a recorded tap dump  
**Target Platform**: macOS 14.4+ (Process Taps), Apple Silicon (M1+); ScreenCaptureKit fallback for macOS 13.0–14.3  
**Project Type**: Single-project Python library (in-process API) with a bundled native capture helper  
**Performance Goals**: Turn-to-text latency ≤3s (target ~1.5–2.0s); diarization low-latency profile ~1.04s; ASR 560ms chunk; sustain a continuous 60-min session without timeline loss  
**Constraints**: Fully on-device (no audio/transcript leaves the machine post-download); ASR runs **FP16 on the M-series GPU + CPU** (ONNX Runtime CoreML EP, `MLComputeUnits = .cpuAndGPU`); fit ASR + diarization models within a 16 GB unified-memory envelope (~1.2 GB FP16 ASR + diarizer); any quantization (INT8/INT4) must pass a WER+latency gate before adoption; 16 kHz mono internal audio; stable per-session speaker IDs  
**Scale/Scope**: Single concurrent meeting session at a time (sequential sessions over the backend lifetime); up to ~4 concurrent speakers; ~40 supported language-locales

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Local-First Processing | ✅ PASS | All capture/diarization/ASR on-device; only network use is one-time model download, then cached. No transcript/audio egress. (FR-014, SC-006) |
| II | Real-Time Pipeline Architecture | ✅ PASS | Cache-aware streaming ASR + streaming diarization; budget ~1.5–2.0s ≤ 3s. Stages buffered & non-blocking. Summarization is out of scope (deferred), consistent with "post-meeting" rule. |
| III | Platform-Native Audio Interception | ✅ PASS | Core Audio Process Taps (native, no per-app hooks, no accessibility hacks); `AudioCapture` protocol isolates native code for future Windows/Linux backends. (FR-002, FR-015) |
| IV | Speaker Diarization as a First-Class Stage | ✅ PASS | Diarizer is first-class, runs first on captured audio, and is authoritative for stable session-scoped speaker IDs. Constitution v2.0.0 §IV explicitly permits coupling option (b): concurrent execution fused on a shared session timeline. This plan uses that sanctioned coupling. |
| V | Hardware-Aware Inference Backends | ✅ PASS | CoreML (Sortformer) + ONNX Runtime CoreML EP **FP16 on M-series GPU + CPU** (`.cpuAndGPU`) for Nemotron — both are accepted Apple-Silicon backends under Constitution v2.0.0 §V; pluggable resolver with CPU/MPS fallback. FP16 models sized for 16 GB; quantization gated on measured WER+latency. |
| VI | Automatic Language Detection Per Turn | ✅ PASS | Nemotron language-ID conditioning; `language=None` auto; detected language recorded per segment with speaker/start/end. (FR-007, FR-008) |
| VII | Modular, Interface-Driven Architecture | ✅ PASS | `AudioCapture`, `SpeakerDiarizer`, `SpeechTranscriber` protocols; no cross-module internal imports; each testable with fixtures. Summarizer module deferred to a later feature. |

**Tech-stack choices vs. the constitution's "Technology Stack" section** (capability-level
principles satisfied; under Constitution v2.0.0 the section lists are explicitly RECOMMENDED
DEFAULTS, and Nemotron/Sortformer are now the named defaults):
- **ASR**: Nemotron 3.5 ASR (ONNX/CoreML) — the v2.0.0 default; superior for native streaming latency.
- **Diarization**: Streaming Sortformer (CoreML) — the v2.0.0 default; pyannote retained as documented fallback.

Both are recorded in Complexity Tracking and conform to the amended constitution.

**Gate result**: PASS (no unwaived deviations; Principle IV coupling option (b) and the
ONNX-Runtime-CoreML-EP backend are sanctioned by Constitution v2.0.0). Cleared for Phase 0/1.

## Project Structure

### Documentation (this feature)

```text
specs/001-meeting-asr-backend/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (Python protocol contracts)
│   ├── audio_capture.md
│   ├── speaker_diarizer.md
│   ├── speech_transcriber.md
│   └── public_api.md
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/meeting_asr/
├── __init__.py              # Public facade: prepare_models(), check_readiness(), TranscriptionSession
├── types.py                 # Dataclasses: TranscriptSegment, Speaker, SessionStatus, ReadinessReport, ...
├── pipeline.py              # Orchestrates capture → (diarize ∥ transcribe) → fuse
├── session.py               # TranscriptionSession state machine + segment stream (callbacks/iterator)
├── audio/
│   ├── __init__.py
│   ├── capture.py           # AudioCapture protocol + AudioFrame type
│   ├── coreaudio_tap.py     # System audio via Swift Process-Tap helper (subprocess/pipe)
│   ├── screencapturekit.py  # Fallback system-audio backend (macOS 13.0–14.3)
│   ├── microphone.py        # Mic capture via sounddevice (PortAudio)
│   └── mixer.py             # Resample → 16 kHz mono, mix sources on one session clock
├── diarization/
│   ├── __init__.py
│   ├── diarizer.py          # SpeakerDiarizer protocol
│   └── sortformer_coreml.py # Streaming Sortformer (CoreML), AOSC speaker timeline
├── asr/
│   ├── __init__.py
│   ├── transcriber.py       # SpeechTranscriber protocol
│   └── nemotron_onnx.py     # Nemotron 3.5 ASR (ONNX Runtime + CoreML EP, FP16, GPU+CPU), cache-aware streaming
├── fusion/
│   ├── __init__.py
│   └── aligner.py           # Align ASR tokens/timestamps to diarization timeline → segments
├── models/
│   ├── __init__.py
│   ├── registry.py          # ModelAsset definitions + prepare()/download/cache (huggingface_hub)
│   └── readiness.py         # SystemReadinessReport: models + permissions + compute
└── backends/
    ├── __init__.py
    └── device.py            # Hardware-aware resolver: CoreML EP GPU+CPU (.cpuAndGPU) + FP16 → CPU/MPS fallback

native/AudioTap/             # Swift Package: Core Audio Process Tap → raw PCM on stdout
├── Package.swift
└── Sources/AudioTap/main.swift

tests/
├── fixtures/audio/          # Multi-speaker, multi-language recorded clips + a tap dump
├── unit/                    # Per-module logic (mixer, aligner, registry, device resolver)
├── integration/             # End-to-end pipeline on fixtures; helper subprocess test
└── contract/                # Protocol-conformance tests for each module interface

requirements.txt             # Existing — to be expanded with new deps
transcribe_meeting.py        # Existing Whisper batch prototype — superseded; left until migration
```

**Structure Decision**: Single-project Python library (Constitution VII modular layout).
Each pipeline stage is its own package exposing a protocol in `*/{capture,diarizer,transcriber}.py`,
with concrete backends alongside. Platform-native capture is isolated under `audio/` and the
`native/AudioTap` Swift package. The public in-process API is the thin facade in
`src/meeting_asr/__init__.py`. No `frontend/` or web service — UI is a future feature.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| **Principle IV coupling option (b)** (now sanctioned by Constitution v2.0.0 §IV): diarization and ASR run in parallel and are fused by timestamp, instead of the diarizer feeding pre-cut audio segments into the transcriber | The Nemotron ASR is a *cache-aware streaming* RNNT: it depends on an uninterrupted audio stream to keep its encoder/conv cache valid. Cutting audio into per-speaker segments and re-feeding them resets that cache every turn — directly increasing latency and degrading accuracy, contradicting the user's explicit "lowest latency / optimal" requirement. Diarization is still first-class, runs first on the captured audio, and is authoritative for speaker IDs; only the *coupling mechanism* changes. | Strict gate-by-segment feeding was rejected because it breaks streaming-cache continuity (higher latency, lower accuracy) and would force fixed-size buffering that adds turn latency — the opposite of the goal. Timestamp fusion preserves both stages' streaming behavior. |
| **Tech-stack swap**: Nemotron 3.5 ASR (ONNX/CoreML) instead of constitution-listed Whisper | Spec mandates Nemotron; it is natively streaming + cache-aware + multilingual-per-turn, which Whisper is not. | Whisper is not natively streaming (worse live latency) and would not satisfy SC-001/Principle II as cleanly. |
| **Tech-stack swap**: Streaming Sortformer (CoreML) instead of constitution-listed pyannote.audio | Latest-research, lowest-latency online diarizer with stable AOSC IDs and ANE acceleration; pyannote's streaming path is higher-latency. | pyannote 3.1 retained as a documented fallback if Sortformer's 4-speaker cap/accuracy proves limiting; not chosen as primary because its online latency is worse. |
