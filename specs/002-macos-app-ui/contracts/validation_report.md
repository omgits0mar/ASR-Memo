# Contract — Accuracy Validation Harness

Package: `validation/` (dev/QA only; not shipped in the `.app`). Realizes FR-016/017 and
the SC-006/007 acceptance evidence by running the **real integrated pipeline**
(`meeting_asr.transcribe_file`) over small labeled public clips and scoring against
ground truth.

## CLI

```
python -m validation [--axis asr|diarization|language|all]
                      [--report-json PATH] [--report-md PATH]
                      [--samples-dir tests/fixtures/validation]
```
Exit code `0` iff the aggregate `passed` is true; non-zero otherwise (CI-usable gate).
Prints a Markdown summary and, with `--report-json`, writes the full `ValidationReport`.

## Public functions

```python
def load_samples(samples_dir: str, axis: str = "all") -> list[ValidationSample]: ...
def run_validation(samples: Sequence[ValidationSample],
                   thresholds: Mapping[str, float] | None = None) -> ValidationReport: ...

# metrics (validation/metrics.py)
def wer(reference: str, hypothesis: str) -> float: ...                      # via jiwer
def diarization_accuracy(ref_turns, hyp_segments) -> float: ...             # permutation-invariant
def language_id_accuracy(ref_languages, hyp_segments) -> float: ...
```

## `ValidationReport` shape

```json
{
  "generated_at": "2026-06-15T12:00:00Z",
  "thresholds": { "wer_max": 0.15, "diarization_min": 0.90, "language_id_min": 0.95 },
  "per_clip": [
    { "sample_id": "libri-1089-0001", "axis": "asr",
      "wer": 0.07, "diarization_accuracy": null, "language_id_accuracy": null,
      "passed": true, "notes": "" }
  ],
  "aggregate": { "mean_wer": 0.09, "mean_diarization_accuracy": 0.93,
                 "mean_language_id_accuracy": 0.97, "n_clips": 6 },
  "passed": true
}
```

## Metric definitions & rules

- **WER** (SC-006): standard word error rate via `jiwer` over normalized text; computed
  only for clips with `ref_text`. Aggregate `mean_wer ≤ 0.15` to pass.
- **Diarization accuracy** (SC-004): fraction of reference speech-time assigned to the
  correct speaker, computed under the optimal reference↔hypothesis label permutation
  (labels are arbitrary `Speaker N`). A DER-style helper is reported alongside. Aggregate
  `≥ 0.90` to pass.
- **Language-ID accuracy** (SC-005): fraction of segments whose detected `language`
  matches the reference language for the overlapping time span. Aggregate `≥ 0.95`.
- **Pass/fail**: per-clip `passed` = all *applicable* metrics meet thresholds; report
  `passed` = all aggregate metrics meet thresholds. Failing clips are listed explicitly
  (FR-017).
- **Reproducibility** (SC-007): same samples + config reproduce metrics within a small
  tolerance (fixed clips, deterministic decode); the harness sets no nondeterministic
  seeds in the decode path.

## Execution modes

- **`needs_models` (real)**: default behavior with downloaded models on Apple Silicon —
  produces the genuine SC-006/007 evidence.
- **Offline self-test**: `tests/integration/test_validation_run.py` runs the harness over
  *synthetic* labeled clips with injected fakes whose output exactly matches references,
  asserting metric computation and report assembly (no network, no models) — validates the
  harness logic itself, not model accuracy.

## Datasets (curated, small — finalized at implementation)

Cached under `tests/fixtures/validation/` with per-sample ground-truth manifests:
- ASR/WER: a few LibriSpeech `test-clean` utterances.
- Diarization: 1–2 short multi-speaker clips with reference turns.
- Language-ID: a few short clips across supported languages (e.g. FLEURS/Common Voice).
Provenance + license recorded per `ValidationSample.source`. One-time fetch (mirrors model
download), offline thereafter.
