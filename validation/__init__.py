"""validation — repeatable on-device accuracy harness (dev/QA only; not shipped).

Feeds small labeled public clips through the real integrated pipeline
(:func:`meeting_asr.transcribe_file`) and scores WER / diarization accuracy /
language-ID accuracy against thresholds. ``python -m validation`` is the CLI
(``make validate``). Real-model accuracy runs behind ``needs_models``; the harness
logic itself is exercised offline with fakes/synthetic clips.

Public surface::

    load_samples(samples_dir, axis="all") -> list[ValidationSample]
    run_validation(samples, thresholds=None) -> ValidationReport
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
