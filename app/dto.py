"""JSON DTO serializers for the JS↔Python bridge (task T008).

Pure projection of backend dataclasses → the JSON-serializable plain objects defined
in ``specs/002-macos-app-ui/contracts/js_bridge_api.md`` (SegmentDTO / SpeakerDTO /
ReadinessDTO / ErrorInfo / PrepareProgress). Backend values live in-process; the
bridge sends only these dicts over the webview channel (Constitution I + VII).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from meeting_asr.export.palette import SPEAKER_COLORS, speaker_color
from meeting_asr.types import (
    AudioSourceKind,
    ConfidenceBand,
    ErrorInfo,
    ModelAsset,
    PrepareProgress,
    Speaker,
    SystemReadinessReport,
    TranscriptSegment,
)

__all__ = [
    "SPEAKER_COLORS",
    "speaker_color",
    "segment_dto",
    "speaker_dto",
    "readiness_dto",
    "error_dto",
    "prepare_progress_dto",
]

# NOTE: SPEAKER_COLORS / speaker_color are re-exported from
# meeting_asr.export.palette (single source). app/web/styles.css + app/web/app.js
# mirror the same hex values so the color is stable backend ↔ UI.


def _source_value(source: Optional[AudioSourceKind]) -> Optional[str]:
    """Contract: SegmentDTO.source ∈ {"microphone", "system", null}."""
    if isinstance(source, AudioSourceKind):
        return source.value
    if source is None:
        return None
    if isinstance(source, str) and source in ("microphone", "system"):
        return source
    return None  # unknown value → null (never an arbitrary string)


def _band_value(band: Optional[ConfidenceBand]) -> Optional[str]:
    return band.value if isinstance(band, ConfidenceBand) else (band if band is None else str(band))


def segment_dto(seg: TranscriptSegment) -> dict:
    """SegmentDTO — the atomic rendered/exported unit."""
    return {
        "segment_id": seg.segment_id,
        "speaker_label": seg.speaker_label,
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text,
        "language": seg.language,
        "confidence": float(seg.confidence),
        "confidence_band": _band_value(seg.confidence_band),
        "source": _source_value(seg.source),
        "is_final": bool(seg.is_final),
    }


def speaker_dto(speaker: Speaker, *, color: str, segment_count: int = 0) -> dict:
    """SpeakerDTO — display metadata for a roster speaker (color assigned by caller)."""
    return {
        "label": speaker.label,
        "color": color,
        "total_speech_seconds": float(speaker.total_speech_seconds),
        "segment_count": int(segment_count),
    }


def _model_dto(asset: ModelAsset) -> dict:
    return {
        "name": asset.name,
        "kind": asset.kind.value if hasattr(asset.kind, "value") else str(asset.kind),
        "state": asset.state.value if hasattr(asset.state, "value") else str(asset.state),
        "is_cached": bool(asset.is_cached()),
    }


def readiness_dto(report: SystemReadinessReport) -> dict:
    """ReadinessDTO — drives the setup screen + the ready gate."""
    return {
        "ready": bool(report.ready),
        "compute_backend": report.compute_backend,
        "os_supports_process_tap": bool(report.os_supports_process_tap),
        "mic_permission": bool(report.mic_permission),
        "system_audio_permission": bool(report.system_audio_permission),
        "models": [_model_dto(m) for m in report.models],
        "missing": list(report.missing),
    }


def error_dto(info: ErrorInfo) -> dict:
    """ErrorInfo — recoverable/terminal condition surfaced to the UI (FR-014/018)."""
    return {
        "code": info.code,
        "message": info.message,
        "recoverable": bool(info.recoverable),
        "hint": info.hint,
    }


def prepare_progress_dto(p: PrepareProgress) -> dict:
    """prepare_progress event payload (FR-008/010)."""
    return {
        "asset": p.asset,
        "downloaded": int(p.downloaded),
        "total": int(p.total),
        "fraction": float(p.fraction),
        "state": p.state.value if hasattr(p.state, "value") else str(p.state),
    }
