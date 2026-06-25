# Validation fixtures (US5 — 002)

A **small, curated** set of labeled public clips used by the accuracy-validation
harness (`make validate` / `python -m validation`) to produce objective, repeatable
WER / diarization / language-ID evidence (SC-006/007). Intentionally small —
*demonstrate correctness, not full benchmark coverage*.

## Provenance & license (per sample)

| Axis | Source | License | Use |
|------|--------|---------|-----|
| ASR (WER) | [LibriSpeech](https://openslr.org/12) `test-clean` | CC BY 4.0 | clean-speech WER (SC-006) |
| Diarization | [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) `Mix-Headset` | CC BY 4.0 | multi-speaker attribution (SC-004) |
| Language-ID | [FLEURS](https://github.com/google-research/datasets/tree/master/fleurs) | CC-BY-NC-4.0 | per-segment language (SC-005) |

Each `ValidationSample.source` records this provenance + license inline (see
`manifest.json`).

## One-time fetch (network), then fully offline

The audio is **not** committed (see `.gitignore`). Fetch it once, then `make validate`
runs offline against the local cache:

1. **LibriSpeech test-clean** — download a handful of utterances (e.g. from
   openslr.org/12), place the `.flac` under `librispeech/`, and copy each utterance's
   transcript line from the matching `<speaker>-<chapter>.trans.txt` into the
   `ref_text` field for that sample in `manifest.json`.
2. **AMI** — take a short `Mix-Headset` excerpt, place under `ami/`, and fill
   `ref_turns` (`[[speaker, start, end], ...]`) from the corpus RTTM.
3. **FLEURS** — take a few short clips across languages, place under `fleurs/`;
   `ref_languages` is already set per sample.

After fetching, audio files live under this directory and `manifest.json` holds the
ground truth. The gitignored extensions (`.wav/.flac/.mp3/.ogg/.m4a`) keep the repo
lean; the manifest is the committed source of truth.

## Running

```bash
make validate                                   # all axes (needs_models)
python -m validation --axis asr                 # WER ≤ 15% (SC-006)
python -m validation --report-json out/v.json --report-md out/v.md
```

Exit code is `0` iff the aggregate thresholds (WER ≤ 0.15, diarization ≥ 0.90,
language-ID ≥ 0.95) are met. The harness *logic* (metrics + report assembly) is
verified offline by `tests/integration/test_validation_run.py` over synthetic clips
with injected fakes — no network, no models.
