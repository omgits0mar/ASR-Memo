"""Speaker color palette — single source of truth (US4 legend + app SpeakerView).

The Markdown export legend needs a color swatch (contracts/transcript_export.md) and
the desktop UI assigns the same colors by arrival order (app/dto.py). Defining it in
the backend ``export`` package keeps the dependency direction correct: ``app`` may
import from ``meeting_asr``, never the reverse.
"""

from __future__ import annotations

__all__ = ["SPEAKER_COLORS", "speaker_color"]

# Deterministic, accessible speaker palette assigned in arrival order (FR-003).
SPEAKER_COLORS = (
    "#1A7F64",  # green   — Speaker 1
    "#2D7FF9",  # blue    — Speaker 2
    "#B4515C",  # rose    — Speaker 3
    "#9B59B6",  # purple  — Speaker 4 (Sortformer 4spk capacity)
)


def speaker_color(index: int) -> str:
    """Color for the n-th speaker (0-based arrival order). Wraps if exceeded."""
    return SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
