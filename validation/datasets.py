"""Validation datasets — labeled public-clip manifest + loader (T045 / US5).

A small curated set of labeled clips (ASR/diarization/language) cached under
``tests/fixtures/validation/`` with a per-sample ground-truth manifest. The audio is
fetched once (mirrors model download; not committed), the manifest is. ``load_samples``
reads the manifest and resolves local paths; filtering by axis (``asr`` /
``diarization`` / ``language`` / ``all``).

Manifest format (``<samples_dir>/manifest.json``)::

    { "samples": [ { "sample_id", "path", "axis", "ref_text", "ref_turns",
                     "ref_languages", "source" }, ... ] }

``ref_turns`` = ``[[speaker, start, end], ...]``; ``ref_languages`` =
``[[start, end, lang], ...]``; ``path`` is relative to the samples dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = ["ValidationSample", "AXES", "load_samples"]

AXES = ("asr", "diarization", "language", "all")


@dataclass
class ValidationSample:
    """One labeled public clip used by the accuracy harness."""

    sample_id: str
    path: str
    axis: str  # asr | diarization | language
    ref_text: Optional[str]
    ref_turns: Optional[List[tuple]]       # [(speaker, start, end), ...]
    ref_languages: Optional[List[tuple]]   # [(start, end, lang), ...]
    source: str  # dataset provenance + license note


def load_samples(samples_dir: str, axis: str = "all") -> List[ValidationSample]:
    """Load the curated labeled samples from ``samples_dir/manifest.json``.

    ``axis`` filters by ``all`` (default) or a single axis. Paths are resolved
    relative to the samples dir. Raises ``FileNotFoundError`` if no manifest exists.
    """
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; choose one of {AXES}")
    base = Path(samples_dir)
    manifest = base / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest.json under {samples_dir!r}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("samples", []) if isinstance(data, dict) else data

    out: List[ValidationSample] = []
    for row in rows:
        ax = row.get("axis", "asr")
        if axis != "all" and ax != axis:
            continue
        path = row["path"]
        resolved = str((base / path) if not Path(path).is_absolute() else Path(path))
        out.append(
            ValidationSample(
                sample_id=row["sample_id"],
                path=resolved,
                axis=ax,
                ref_text=row.get("ref_text"),
                ref_turns=[tuple(t) for t in row["ref_turns"]] if row.get("ref_turns") else None,
                ref_languages=[tuple(t) for t in row["ref_languages"]] if row.get("ref_languages") else None,
                source=row.get("source", ""),
            )
        )
    return out
