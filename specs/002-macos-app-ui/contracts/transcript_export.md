# Contract — Transcript Export (Markdown + JSON)

Module: `src/meeting_asr/export/` (`__init__.py` re-exports `export_markdown`,
`export_json`, `write_export`). Pure projection of session output (FR-013, SC-009).

## API

```python
def export_markdown(
    segments: Sequence[TranscriptSegment],
    speakers: Mapping[str, Speaker],
    *, session_meta: Optional[Mapping[str, object]] = None,
) -> str: ...

def export_json(
    segments: Sequence[TranscriptSegment],
    speakers: Mapping[str, Speaker],
    *, session_meta: Optional[Mapping[str, object]] = None,
) -> str: ...

def write_export(path: str, segments, speakers, *, session_meta=None) -> str:
    """Pick format from the path extension (.md/.markdown → markdown, .json → json),
       write the file locally, return the path. Raises ValueError on unknown extension."""
```

## Markdown format

- A header block: title, optional session metadata (mode, language hint, date), and a
  **speaker legend** (`Speaker N` → color swatch / count) for readability.
- One bullet per segment, chronological:
  `- [hh:mm:ss] **{speaker_label}** _({language})_: {text}`
  (language omitted/`?` if `None`; low-confidence segments flagged, e.g. trailing `⚠︎`).

## JSON format

A single object (lossless mirror of the dataclasses):

```json
{
  "session": {
    "app_session_id": "…", "input_mode": "live|file",
    "language_hint": null,
    "speakers": { "Speaker 1": { "total_speech_seconds": 12.3,
                                 "first_seen": 0.0, "last_seen": 41.2 } }
  },
  "segments": [
    { "segment_id": "seg_…", "speaker_label": "Speaker 1",
      "start": 0.0, "end": 3.2, "text": "…", "language": "en",
      "confidence": 0.93, "confidence_band": "high",
      "source": "microphone", "is_final": true }
  ]
}
```

## Guarantees

- **Completeness (SC-009)**: every segment's `speaker_label`, `start`, `end`, `language`,
  and `text` appear in **both** formats.
- **Order**: segments are sorted by `(start, end)` (chronological, FR-019).
- **JSON round-trip**: parsing the JSON back yields the same per-segment field values
  (numbers within float tolerance) — asserted in tests.
- **Local-only**: `write_export` writes to the user-chosen local path; no network
  (Principle I).
- **Encoding**: UTF-8; multilingual text preserved verbatim.

## Testability

`tests/unit/test_export.py`: build a known multi-speaker, multilingual segment list →
assert Markdown contains every speaker/time/language/text and a legend; assert JSON
round-trips all fields; assert chronological ordering; assert `write_export` extension
routing + `ValueError` on unknown extension.
