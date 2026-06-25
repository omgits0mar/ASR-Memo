"""Offline metrics for the constitution gates (WER, DER).

Pure-Python implementations used by the quantization / diarization-accuracy /
performance gate tests against the fixture manifests.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def wer(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    """Word error rate = Levenshtein(hyp, ref) / max(len(ref), 1)."""
    r, h = list(reference), list(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(r)


def der(attributed, reference) -> float:
    """Diarization error rate: misattributed speech time / total speech time.

    ``attributed``/``reference`` are iterables of (t_start, t_end, speaker_label).
    For each reference turn, find the attributed turn covering its midpoint; if
    the speakers differ, that turn's duration counts as error.
    """
    ref = sorted(reference, key=lambda t: t[0])
    att = sorted(attributed, key=lambda t: t[0])
    total = 0.0
    error = 0.0
    for t0, t1, spk in ref:
        mid = (t0 + t1) / 2
        total += t1 - t0
        match = next((a for a in att if a[0] <= mid <= a[1]), None)
        if match is None or match[2] != spk:
            error += t1 - t0
    return error / total if total > 0 else 0.0
