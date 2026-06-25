"""JSON export — lossless, structured transcript mirror (US4; SC-009).

A single object with a ``session`` metadata block and a ``segments`` array, each
segment carrying the full :class:`TranscriptSegment` field set. ``ensure_ascii=False``
preserves multilingual text verbatim; UTF-8 encoding. Pure projection — no state.
"""

from __future__ import annotations

import json
from typing import Mapping, Optional, Sequence

from ..types import AudioSourceKind, ConfidenceBand, Speaker, TranscriptSegment

__all__ = ["export_json"]


def _band(band) -> Optional[str]:
    return band.value if isinstance(band, ConfidenceBand) else band


def _source(source) -> Optional[str]:
    return source.value if isinstance(source, AudioSourceKind) else (
        source if source is None else str(source)
    )


def _segment_dict(seg: TranscriptSegment) -> dict:
    return {
        "segment_id": seg.segment_id,
        "speaker_label": seg.speaker_label,
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text,
        "language": seg.language,
        "confidence": float(seg.confidence),
        "confidence_band": _band(seg.confidence_band),
        "source": _source(seg.source),
        "is_final": bool(seg.is_final),
    }


def export_json(
    segments: Sequence[TranscriptSegment],
    speakers: Mapping[str, Speaker],
    *,
    session_meta: Optional[Mapping[str, object]] = None,
) -> str:
    """Render segments to a structured JSON document (chronological, lossless)."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    meta = dict(session_meta or {})
    speaker_block = {
        label: {
            "first_seen": float(sp.first_seen),
            "last_seen": float(sp.last_seen),
            "total_speech_seconds": float(sp.total_speech_seconds),
        }
        for label, sp in speakers.items()
    }
    doc = {
        "session": {
            "app_session_id": meta.get("app_session_id"),
            "input_mode": meta.get("input_mode"),
            "language_hint": meta.get("language_hint"),
            "speakers": speaker_block,
        },
        "segments": [_segment_dict(s) for s in ordered],
    }
    return json.dumps(doc, ensure_ascii=False, indent=2)
