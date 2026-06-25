# Phase 1 Data Model: Realtime Diarized Meeting Transcription Backend

**Feature**: `001-meeting-asr-backend` | **Date**: 2026-06-14

All entities are in-process Python dataclasses (see `src/meeting_asr/types.py`). Times are
seconds (float) relative to a single monotonic **session clock** that starts at session start.

---

## Enums

```text
AudioSourceKind   = MICROPHONE | SYSTEM            # local vs. meeting-app output
SessionStatus     = CREATED | PREPARING | ACTIVE | STOPPING | STOPPED | ERROR
CaptureState      = IDLE | CAPTURING | PERMISSION_DENIED | DEVICE_LOST | ERROR
ModelKind         = ASR | DIARIZER
ModelState        = ABSENT | DOWNLOADING | CACHED | LOADED | ERROR
Confidence band   = HIGH | MEDIUM | LOW | UNKNOWN  # also carries a 0..1 score
```

## Entity: TranscriptSegment

The core output unit. One attributed unit of recognized speech.

| Field | Type | Notes |
|-------|------|-------|
| `segment_id` | str (uuid) | Stable id for updates/dedup |
| `speaker_label` | str | "Speaker 1", "Speaker 2", … (session-scoped, stable) |
| `start` | float | Session-clock seconds; ≥ 0 |
| `end` | float | `end ≥ start` |
| `text` | str | Recognized text, with punctuation/capitalization |
| `language` | str \| None | Detected BCP-47-ish locale tag (e.g., "en", "es"); None if unknown |
| `confidence` | float | 0..1 |
| `confidence_band` | enum | Derived from score; LOW/UNKNOWN flags hard cases (FR-018) |
| `source` | AudioSourceKind \| None | Best-effort origin (mic vs system); None if mixed/ambiguous |
| `is_final` | bool | False for in-progress hypotheses, True once fused & finalized |

**Rules**: ordered by `start` then `end`; finalized segments are non-overlapping per speaker;
ordering/timestamps remain coherent under backpressure (FR-016). Empty-text finals are dropped
(silence edge case).

## Entity: Speaker

A distinct voice within one session.

| Field | Type | Notes |
|-------|------|-------|
| `label` | str | "Speaker N" |
| `first_seen` | float | Session-clock seconds of first attributed speech |
| `last_seen` | float | Updated as the speaker continues |
| `total_speech_seconds` | float | Aggregate attributed speech time |

**Rules**: labels assigned in arrival order (Sortformer AOSC); stable for session lifetime;
not persisted/recognized across sessions (Assumption: session-scoped, anonymous).

## Entity: AudioSource

A configured/observed capture input.

| Field | Type | Notes |
|-------|------|-------|
| `kind` | AudioSourceKind | MICROPHONE or SYSTEM |
| `enabled` | bool | Whether this source is part of the session |
| `state` | CaptureState | Live capture state |
| `device_name` | str \| None | Resolved device/process description |
| `sample_rate_in` | int \| None | Native rate before resample to 16 kHz mono |

## Entity: TranscriptionSession

A single live capture-and-transcribe run.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | str (uuid) | Identifier returned to the consumer |
| `status` | SessionStatus | State machine (below) |
| `sources` | list[AudioSource] | mic, system, or both |
| `language_hint` | str \| None | Optional bias; None = auto per-turn |
| `started_at` | float \| None | Wall-clock start |
| `stopped_at` | float \| None | Wall-clock stop |
| `speakers` | dict[str, Speaker] | By label |
| `segments` | list[TranscriptSegment] | Chronological; final transcript on stop |
| `error` | ErrorInfo \| None | Set when status == ERROR |

**State transitions**:
```text
CREATED → PREPARING → ACTIVE → STOPPING → STOPPED
   └──────────────────────────────────────→ ERROR   (from any state; carries ErrorInfo)
```
- `ACTIVE`: capture + diarize + transcribe running; segments stream out and the accumulating
  transcript is queryable at any time (FR-019).
- Permission/device failures move the session (or a source) to ERROR with actionable `ErrorInfo`.

## Entity: ModelAsset

A downloadable model required by the pipeline.

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Logical name (e.g., "nemotron-3.5-asr") |
| `kind` | ModelKind | ASR or DIARIZER |
| `repo_id` | str | HF repo (e.g., `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`) |
| `revision` | str | Pinned revision for reproducibility |
| `expected_files` | list[str] | Files that must exist for `CACHED` |
| `cache_path` | str \| None | Resolved local path once cached |
| `state` | ModelState | ABSENT → DOWNLOADING → CACHED → LOADED |
| `supported_languages` | list[str] \| None | ASR only (~40 locales); None for diarizer |

## Entity: SystemReadinessReport

Snapshot of whether the backend can run (FR-013).

| Field | Type | Notes |
|-------|------|-------|
| `ready` | bool | True only if all required items satisfied |
| `models` | list[ModelAsset] | Each with current `state` |
| `mic_permission` | bool | Microphone authorized |
| `system_audio_permission` | bool | Process-Tap / capture authorized |
| `compute_backend` | str | Resolved backend + compute units ("coreml-gpu+cpu", "coreml-ane", "mps", "cpu") |
| `os_supports_process_tap` | bool | macOS ≥ 14.4 |
| `missing` | list[str] | Human-readable list of what's blocking readiness |

## Supporting types

```text
AudioFrame   { pcm: float32[ ] (16 kHz mono), t_start: float, t_end: float, source: AudioSourceKind }
DiarFrame    { t_start: float, t_end: float, speaker_label: str, score: float }   # 80ms granularity
AsrToken     { text: str, t_start: float, t_end: float, language: str|None, score: float }
ErrorInfo    { code: str, message: str, recoverable: bool, hint: str|None }
PrepareProgress { asset: str, downloaded: int, total: int, state: ModelState }
```

**Fusion mapping**: `aligner` consumes the `DiarFrame` timeline + `AsrToken` stream and emits
`TranscriptSegment`s — grouping contiguous tokens that share the dominant overlapping
`speaker_label`, carrying token language/score up to the segment.
