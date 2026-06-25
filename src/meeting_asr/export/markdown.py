"""Markdown export — speaker-grouped, human-readable transcript (US4; SC-009).

A header block (title + session metadata + a speaker legend) followed by one bullet
per segment, chronological::

    - [hh:mm:ss] **Speaker N** _(lang)_: text

Low/unknown-confidence segments are flagged with a trailing ⚠. Language is shown as
``?`` when unknown. Pure projection of ``TranscriptSegment`` — no backend state.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..types import ConfidenceBand, Speaker, TranscriptSegment
from .palette import speaker_color

__all__ = ["export_markdown"]


def _fmt_time(t: float) -> str:
    s = max(0, int(round(t)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _low(seg: TranscriptSegment) -> bool:
    return seg.confidence_band in (ConfidenceBand.LOW, ConfidenceBand.UNKNOWN)


def export_markdown(
    segments: Sequence[TranscriptSegment],
    speakers: Mapping[str, Speaker],
    *,
    session_meta: Optional[Mapping[str, object]] = None,
) -> str:
    """Render segments to a human-readable Markdown document (chronological)."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    lines: list[str] = ["# Meeting Transcript", ""]

    meta = dict(session_meta or {})
    if meta:
        mode = meta.get("input_mode")
        lang = meta.get("language_hint")
        bits = []
        if mode:
            bits.append(f"mode: {mode}")
        if lang:
            bits.append(f"language hint: {lang}")
        if bits:
            lines.append("> " + " · ".join(bits))
        lines.append("")

    # Speaker legend (arrival order: by first_seen) with the color swatch as a hex
    # code, so the Markdown carries the same color mapping as the UI (FR-003).
    if speakers:
        lines.append("**Speakers:**")
        for idx, label in enumerate(sorted(speakers, key=lambda l: speakers[l].first_seen)):
            spk = speakers[label]
            lines.append(
                f"- **{label}** `{speaker_color(idx)}` — {spk.total_speech_seconds:.1f}s of speech "
                f"(first {spk.first_seen:.1f}s, last {spk.last_seen:.1f}s)"
            )
        lines.append("")

    if not ordered:
        lines.append("_(no speech transcribed)_")
        return "\n".join(lines)

    lines.append("**Transcript:**")
    lines.append("")
    for seg in ordered:
        lang = seg.language or "?"
        flag = " ⚠" if _low(seg) else ""
        lines.append(
            f"- [{_fmt_time(seg.start)}] **{seg.speaker_label}** _({lang})_: "
            f"{seg.text}{flag}"
        )
    return "\n".join(lines) + "\n"
