"""Validation runner — feed labeled clips through the pipeline → ValidationReport (T046 / US5).

Runs each :class:`~validation.datasets.ValidationSample` through the integrated
pipeline (default: :func:`meeting_asr.transcribe_file`; an injectable ``transcribe_fn``
seam backs the offline self-test), scores WER / diarization / language-ID against the
thresholds, and assembles a per-clip + aggregate :class:`ValidationReport` with
pass/fail. Reproducible within tolerance (fixed clips, deterministic decode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Callable, List, Optional, Sequence

from meeting_asr import transcribe_file as _default_transcribe
from meeting_asr.types import TranscriptSegment

from .datasets import ValidationSample
from .metrics import diarization_accuracy, language_id_accuracy, wer

__all__ = [
    "DEFAULT_THRESHOLDS",
    "ClipResult",
    "AggregateMetrics",
    "ValidationReport",
    "run_validation",
    "report_to_dict",
]

# SC-006/007 acceptance thresholds.
DEFAULT_THRESHOLDS = {"wer_max": 0.15, "diarization_min": 0.90, "language_id_min": 0.95}


@dataclass
class ClipResult:
    sample_id: str
    axis: str
    wer: Optional[float] = None
    diarization_accuracy: Optional[float] = None
    language_id_accuracy: Optional[float] = None
    passed: bool = False
    notes: str = ""


@dataclass
class AggregateMetrics:
    mean_wer: Optional[float] = None
    mean_diarization_accuracy: Optional[float] = None
    mean_language_id_accuracy: Optional[float] = None
    n_clips: int = 0


@dataclass
class ValidationReport:
    generated_at: str
    thresholds: dict
    per_clip: List[ClipResult] = field(default_factory=list)
    aggregate: AggregateMetrics = field(default_factory=AggregateMetrics)
    passed: bool = False


def run_validation(
    samples: Sequence[ValidationSample],
    *,
    thresholds: Optional[dict] = None,
    transcribe_fn: Optional[Callable[[ValidationSample], object]] = None,
) -> ValidationReport:
    """Run + score each sample; return a per-clip + aggregate report.

    ``transcribe_fn(sample) -> TranscriptionSession`` defaults to the real
    :func:`meeting_asr.transcribe_file` (``needs_models``); tests inject fakes.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    thr.update(thresholds or {})
    transcribe = transcribe_fn or (lambda s: _default_transcribe(s.path))

    per_clip: List[ClipResult] = []
    wer_vals: List[float] = []
    diar_vals: List[float] = []
    lang_vals: List[float] = []

    for sample in samples:
        clip = ClipResult(sample_id=sample.sample_id, axis=sample.axis)
        try:
            session = transcribe(sample)
            segments: List[TranscriptSegment] = session.transcript()
        except Exception as e:  # a failed clip must not abort the whole run
            clip.passed = False
            clip.notes = f"transcription failed: {e}"
            per_clip.append(clip)
            continue

        checks: List[bool] = []
        if sample.ref_text is not None:
            hyp_text = " ".join(s.text for s in segments)
            clip.wer = wer(sample.ref_text, hyp_text)
            wer_vals.append(clip.wer)
            checks.append(clip.wer <= thr["wer_max"])
        if sample.ref_turns is not None:
            clip.diarization_accuracy = diarization_accuracy(sample.ref_turns, segments)
            diar_vals.append(clip.diarization_accuracy)
            checks.append(clip.diarization_accuracy >= thr["diarization_min"])
        if sample.ref_languages is not None:
            clip.language_id_accuracy = language_id_accuracy(sample.ref_languages, segments)
            lang_vals.append(clip.language_id_accuracy)
            checks.append(clip.language_id_accuracy >= thr["language_id_min"])

        clip.passed = bool(checks) and all(checks)
        per_clip.append(clip)

    aggregate = AggregateMetrics(
        mean_wer=round(mean(wer_vals), 6) if wer_vals else None,
        mean_diarization_accuracy=round(mean(diar_vals), 6) if diar_vals else None,
        mean_language_id_accuracy=round(mean(lang_vals), 6) if lang_vals else None,
        n_clips=len(per_clip),
    )

    passed = True
    if aggregate.mean_wer is not None:
        passed &= aggregate.mean_wer <= thr["wer_max"]
    if aggregate.mean_diarization_accuracy is not None:
        passed &= aggregate.mean_diarization_accuracy >= thr["diarization_min"]
    if aggregate.mean_language_id_accuracy is not None:
        passed &= aggregate.mean_language_id_accuracy >= thr["language_id_min"]

    return ValidationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        thresholds=thr,
        per_clip=per_clip,
        aggregate=aggregate,
        passed=bool(passed),
    )


def report_to_dict(report: ValidationReport) -> dict:
    """Serialize a report to the JSON-serializable shape (contracts/validation_report.md)."""
    return {
        "generated_at": report.generated_at,
        "thresholds": report.thresholds,
        "per_clip": [
            {
                "sample_id": c.sample_id,
                "axis": c.axis,
                "wer": c.wer,
                "diarization_accuracy": c.diarization_accuracy,
                "language_id_accuracy": c.language_id_accuracy,
                "passed": c.passed,
                "notes": c.notes,
            }
            for c in report.per_clip
        ],
        "aggregate": {
            "mean_wer": report.aggregate.mean_wer,
            "mean_diarization_accuracy": report.aggregate.mean_diarization_accuracy,
            "mean_language_id_accuracy": report.aggregate.mean_language_id_accuracy,
            "n_clips": report.aggregate.n_clips,
        },
        "passed": report.passed,
    }
