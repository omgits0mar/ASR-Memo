# Phase 1 Data Model — macOS App UI, Validation & Packaging

Feature: `002-macos-app-ui` · Date: 2026-06-15

This feature adds **presentation, export, and validation** entities around the existing
`001` backend domain types (`TranscriptSegment`, `Speaker`, `SystemReadinessReport`,
`PrepareProgress`, `ErrorInfo`, `AudioSourceKind`, `SessionStatus`). Those are reused
unchanged; the new entities below are mostly *view models* and *projections*.

---

## Reused backend entities (from 001, unchanged)

- **TranscriptSegment** — `speaker_label, start, end, text, segment_id, language,
  confidence, confidence_band, source, is_final`. The atomic unit rendered and exported.
- **Speaker** — `label, first_seen, last_seen, total_speech_seconds`. Session roster.
- **SystemReadinessReport** — `models[], mic_permission, system_audio_permission,
  compute_backend, os_supports_process_tap, missing[]`, `.ready`. Drives the setup screen.
- **PrepareProgress** — `asset, downloaded, total, state`, `.fraction`. Drives download UI.
- **ErrorInfo** — `code, message, recoverable, hint`. Drives error surfaces.

---

## New entity: AppSession (view model)

The UI-facing wrapper around one transcription run (live or file). Lives in
`app/app_session.py`.

| Field | Type | Notes |
|-------|------|-------|
| `app_session_id` | str | UI-local id (distinct from backend `session_id`) |
| `input_mode` | enum `live` \| `file` | Selected source mode (FR-001, FR-005) |
| `source_kinds` | list[AudioSourceKind] | For `live`: mic and/or system |
| `file_path` | str \| None | For `file` mode |
| `status` | enum (see below) | UI status, derived from backend `SessionStatus` |
| `language_hint` | str \| None | Optional bias; `None` ⇒ auto per-turn (Principle VI) |
| `segments` | list[TranscriptSegment] | Accumulated, chronological |
| `speakers` | dict[label → Speaker] | Roster for color-coding |
| `progress` | float (0..1) \| None | File-import progress; `None` for live |
| `error` | ErrorInfo \| None | Terminal/non-terminal condition surfaced to UI |
| `backend_session` | TranscriptionSession | The wrapped 001 session |

**Status (UI)** — a presentation projection of backend `SessionStatus` + setup:
`setting_up → ready → recording (live) | processing (file) → stopping → stopped → error`.
Maps from backend states: `CREATED/PREPARING→setting_up`, `ACTIVE→recording|processing`,
`STOPPING→stopping`, `STOPPED→stopped`, `ERROR→error`; `ready` is pre-session when
`check_readiness().ready` is true.

**Validation rules**:
- `file` mode requires a readable `file_path`; otherwise → `ErrorInfo(code="audio.unreadable")`.
- A session may not start unless readiness is satisfied (else `ErrorInfo(code="not_ready")`,
  UI routes to setup) — enforces the "no start before setup" edge case.
- Exactly one `AppSession` is active at a time (mirrors backend FR-020 sequential guard).

**Transitions**: created → setting_up → ready → (recording|processing) → stopping →
stopped; any state → error. Stopped/error sessions are retained for review/export; a new
session resets segments/speakers/progress/error.

---

## New entity: SpeakerView (UI projection)

Display metadata for a `Speaker` within the rendered transcript (no backend change).

| Field | Type | Notes |
|-------|------|-------|
| `label` | str | `Speaker N` (stable, session-scoped) |
| `color` | str (CSS) | Assigned in arrival order from a fixed palette |
| `total_speech_seconds` | float | From `Speaker` |
| `segment_count` | int | Derived |

**Rule**: color is a deterministic function of arrival order (label index) so the same
speaker keeps the same color for the session (FR-003).

---

## New entity: ExportArtifact (projection, not persisted in-app)

The result of serializing a transcript. Produced by `meeting_asr.export`.

| Field | Type | Notes |
|-------|------|-------|
| `format` | enum `markdown` \| `json` | Chosen by file extension (FR-013) |
| `path` | str | User-chosen save location (local only) |
| `content` | str | Serialized document |

**JSON shape** (lossless mirror of dataclasses):
```json
{
  "session": { "app_session_id": "...", "input_mode": "file|live",
               "language_hint": null, "speakers": { "Speaker 1": {"total_speech_seconds": 12.3} } },
  "segments": [
    { "segment_id": "seg_...", "speaker_label": "Speaker 1",
      "start": 0.0, "end": 3.2, "text": "...", "language": "en",
      "confidence": 0.93, "confidence_band": "high", "source": "microphone" }
  ]
}
```
**Markdown shape**: a speaker legend header + one line per segment
`- [hh:mm:ss] **Speaker N** _(lang)_: text`, grouped/colored for readability.

**Rule** (SC-009): both formats MUST preserve speaker label, timestamps, language, and
text for every segment; round-trip JSON → segments MUST reproduce the input fields.

---

## New entity: ValidationSample

One labeled public clip used by the accuracy harness (`validation/datasets.py`).

| Field | Type | Notes |
|-------|------|-------|
| `sample_id` | str | Stable id |
| `path` | str | Local cached audio file |
| `axis` | enum `asr` \| `diarization` \| `language` | What it primarily validates |
| `ref_text` | str \| None | Ground-truth transcript (ASR/WER) |
| `ref_turns` | list[(speaker, start, end)] \| None | Reference diarization timeline |
| `ref_languages` | list[(start, end, lang)] \| None | Reference per-segment language |
| `source` | str | Dataset provenance + license note |

---

## New entity: ValidationReport

Output of one harness run (`validation/runner.py`).

| Field | Type | Notes |
|-------|------|-------|
| `generated_at` | str (ISO) | Run timestamp |
| `per_clip` | list[ClipResult] | One per `ValidationSample` |
| `aggregate` | AggregateMetrics | Means across clips |
| `thresholds` | dict | WER ≤ 0.15, diarization ≥ 0.90, language-ID ≥ 0.95 |
| `passed` | bool | All aggregate metrics meet thresholds |

**ClipResult**: `sample_id, axis, wer?, diarization_accuracy?, language_id_accuracy?,
passed, notes`.
**AggregateMetrics**: `mean_wer, mean_diarization_accuracy, mean_language_id_accuracy,
n_clips`.

**Rules** (FR-016/017, SC-006/007):
- WER computed only for clips with `ref_text`; diarization/language metrics only where
  references exist.
- Diarization accuracy is permutation-invariant (optimal label matching to references).
- Re-running the same samples/config reproduces metrics within a small tolerance.
- `passed` is per-clip vs. threshold and aggregate; failing clips are listed explicitly.

---

## Entity relationships

```
SystemReadinessReport ─drives→ Setup screen ─gates→ AppSession
AppSession ─wraps→ TranscriptionSession (001) ─emits→ TranscriptSegment*  ─render→ UI (SpeakerView color-coding)
AppSession.segments ─project→ ExportArtifact (markdown | json)
ValidationSample* ─fed through→ transcribe_file → TranscriptSegment* ─compare ref→ ValidationReport
```
