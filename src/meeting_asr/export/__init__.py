"""Transcript export module — Markdown + JSON projections (US4; FR-013, SC-009).

Re-exports the format-specific writers plus :func:`write_export`, which routes by
file extension (``.md``/``.markdown`` → Markdown, ``.json`` → JSON) and writes the
file locally (no network — Principle I). Raises :class:`ValueError` on an unknown
extension so the UI can surface an actionable error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..types import Speaker, TranscriptSegment
from .json_export import export_json
from .markdown import export_markdown

__all__ = ["export_markdown", "export_json", "write_export"]


def write_export(
    path: str,
    segments: Sequence[TranscriptSegment],
    speakers: Mapping[str, Speaker],
    *,
    session_meta: Optional[Mapping[str, object]] = None,
) -> str:
    """Pick format from the path extension, write the file locally, return the path.

    ``.md``/``.markdown`` → Markdown; ``.json`` → JSON. Raises ``ValueError`` for an
    unknown extension.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("md", "markdown"):
        content = export_markdown(segments, speakers, session_meta=session_meta)
    elif ext == "json":
        content = export_json(segments, speakers, session_meta=session_meta)
    else:
        raise ValueError(f"unknown export extension '.{ext}' (use .md or .json)")
    Path(path).write_text(content, encoding="utf-8")
    return str(path)
