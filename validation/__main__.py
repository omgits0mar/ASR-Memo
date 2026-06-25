"""``python -m validation`` — accuracy-validation CLI (task T047 / US5).

    python -m validation [--axis asr|diarization|language|all]
                         [--report-json PATH] [--report-md PATH]
                         [--samples-dir tests/fixtures/validation]

Feeds each labeled clip through the real integrated pipeline, scores WER /
diarization / language-ID, prints a Markdown summary, optionally writes JSON/Markdown
reports, and exits ``0`` iff the aggregate ``passed`` is true (CI-usable gate).
Requires downloaded models (``needs_models``); the harness *logic* is exercised
offline by ``tests/integration/test_validation_run.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .datasets import AXES, load_samples
from .runner import report_to_dict, run_validation

DEFAULT_SAMPLES_DIR = "tests/fixtures/validation"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m validation", description="On-device accuracy validation harness")
    p.add_argument("--axis", default="all", choices=AXES, help="validate one axis (default: all)")
    p.add_argument("--report-json", metavar="PATH", help="write the full ValidationReport as JSON")
    p.add_argument("--report-md", metavar="PATH", help="write the Markdown summary to a file")
    p.add_argument("--samples-dir", default=DEFAULT_SAMPLES_DIR, help="directory containing manifest.json")
    return p


def markdown_summary(report) -> str:
    a = report.aggregate
    lines: List[str] = [
        "# Validation Report",
        "",
        f"- generated: {report.generated_at}",
        f"- result: **{'PASS' if report.passed else 'FAIL'}**",
        f"- clips: {a.n_clips}",
        f"- thresholds: WER ≤ {report.thresholds['wer_max']}, "
        f"diarization ≥ {report.thresholds['diarization_min']}, "
        f"language-ID ≥ {report.thresholds['language_id_min']}",
        "",
        "## Aggregate",
        f"- mean WER: {_fmt(a.mean_wer)}",
        f"- mean diarization accuracy: {_fmt(a.mean_diarization_accuracy)}",
        f"- mean language-ID accuracy: {_fmt(a.mean_language_id_accuracy)}",
        "",
        "## Per-clip",
        "| sample | axis | WER | diar | lang | result |",
        "|--------|------|-----|------|------|--------|",
    ]
    for c in report.per_clip:
        lines.append(
            f"| {c.sample_id} | {c.axis} | {_fmt(c.wer)} | "
            f"{_fmt(c.diarization_accuracy)} | {_fmt(c.language_id_accuracy)} | "
            f"{'✓' if c.passed else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        samples = load_samples(args.samples_dir, args.axis)
    except FileNotFoundError as e:
        sys.stderr.write(
            f"[validation] {e}\n"
            "Fetch the curated clips first (one-time); see tests/fixtures/validation/README.\n"
        )
        return 2
    if not samples:
        sys.stderr.write(f"[validation] no samples for axis '{args.axis}' under {args.samples_dir}\n")
        return 2

    report = run_validation(samples)  # real integrated pipeline (needs_models)

    summary = markdown_summary(report)
    print(summary)

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    if args.report_md:
        Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_md).write_text(summary, encoding="utf-8")

    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
