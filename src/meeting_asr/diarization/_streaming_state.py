"""Numpy port of NeMo's Streaming Sortformer state machine (002 / Decision 4).

The CoreML ``Sortformer.mlpackage`` does ALL neural compute (pre-encode +
frontend encoder + inference head → ``speaker_preds``). The streaming STATE —
how the speaker cache (``spkcache``) and FIFO roll frame-to-frame, and how the
cache is compressed back to ``spkcache_len`` — is pure buffer/top-k logic in
NeMo's ``SortformerModules`` (``nemo.collections.asr.modules.sortformer_modules``)
and is **parameter-free** (no model weights). This module ports that logic to
numpy so the diarizer runs on-device with no torch/NeMo at runtime.

Faithful 1:1 port of (NeMo main branch, ``sortformer_modules.py``):

* ``init_streaming_state``  — zero/growing buffers (sync path).
* ``streaming_update``      — FIFO append, overflow → spkcache, compress.
* ``_get_silence_profile``  — running mean silence embedding.
* ``_compress_spkcache``    — keep the ``spkcache_len`` most informative frames.
* ``_get_log_pred_scores`` / ``_disable_low_scores`` / ``_boost_topk_scores`` /
  ``_get_topk_indices`` / ``_gather_spkcache_and_preds`` — top-k selection.

Config is the FluidInference export profile for ``diar_streaming_sortformer_4spk-
v2.1`` (chunk_len=6, L=1, R=7, subsampling=8, fifo=40, spkcache=188,
update_period=31, fc_d_model=512, n_spk=4) — proven against the live model spec.
All other constants are NeMo ``SortformerModules.__init__`` defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- export profile (proven against the cached CoreML model spec) ----
N_SPK = 4
FC_D_MODEL = 512
SPKCACHE_LEN = 188
FIFO_LEN = 40
SPKCACHE_UPDATE_PERIOD = 31
# ---- NeMo SortformerModules defaults (unchanged for this export) ----
SPKCACHE_SIL_FRAMES_PER_SPK = 3
PRED_SCORE_THRESHOLD = 0.25
SIL_THRESHOLD = 0.2
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POS_SCORES_RATE = 0.5
SCORES_BOOST_LATEST = 0.05
MAX_INDEX = 99999


@dataclass
class StreamingState:
    """Mutable streaming buffers (batch=1). Lengths grow from 0; arrays are logical."""

    spkcache: np.ndarray = field(default_factory=lambda: np.zeros((1, 0, FC_D_MODEL), dtype=np.float32))
    fifo: np.ndarray = field(default_factory=lambda: np.zeros((1, 0, FC_D_MODEL), dtype=np.float32))
    spkcache_preds: np.ndarray | None = None
    fifo_preds: np.ndarray | None = None
    mean_sil_emb: np.ndarray = field(default_factory=lambda: np.zeros((1, FC_D_MODEL), dtype=np.float32))
    n_sil_frames: np.ndarray = field(default_factory=lambda: np.zeros((1,), dtype=np.int64))

    @property
    def spkcache_len(self) -> int:
        return int(self.spkcache.shape[1])

    @property
    def fifo_len(self) -> int:
        return int(self.fifo.shape[1])


def init_streaming_state() -> StreamingState:
    """NeMo ``init_streaming_state`` (sync path): empty/growing buffers."""
    return StreamingState()


def streaming_update(
    state: StreamingState,
    chunk: np.ndarray,
    preds: np.ndarray,
    lc: int = 0,
    rc: int = 0,
) -> np.ndarray:
    """NeMo ``streaming_update`` (sync). Returns ``chunk_preds`` [1, chunk_len, n_spk].

    ``chunk``  = ``chunk_pre_encoder_embs`` [1, lc+chunk_len+rc, 512] (CoreML output).
    ``preds``  = ``speaker_preds`` [1, spkcache_len+fifo_len+lc+chunk_len+rc, n_spk].
    """
    spkcache_len = state.spkcache_len
    fifo_len = state.fifo_len
    chunk_len = chunk.shape[1] - lc - rc  # 6 mid-stream

    # Predictions for the FIFO + chunk regions of the packed [spkcache|fifo|chunk] preds.
    state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
    chunk_core = chunk[:, lc : lc + chunk_len]  # [1, chunk_len, 512] NEW pre-enc frames
    chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

    # Append the new chunk to the FIFO tail.
    state.fifo = np.concatenate([state.fifo, chunk_core], axis=1)
    state.fifo_preds = np.concatenate([state.fifo_preds, chunk_preds], axis=1)

    if fifo_len + chunk_len > FIFO_LEN:
        # FIFO overflow: pop frames from the HEAD → append to the spkcache tail.
        pop_out_len = SPKCACHE_UPDATE_PERIOD
        pop_out_len = max(pop_out_len, chunk_len - FIFO_LEN + fifo_len)
        pop_out_len = min(pop_out_len, fifo_len + chunk_len)

        pop_out_embs = state.fifo[:, :pop_out_len]
        pop_out_preds = state.fifo_preds[:, :pop_out_len]
        state.mean_sil_emb, state.n_sil_frames = _get_silence_profile(
            state.mean_sil_emb, state.n_sil_frames, pop_out_embs, pop_out_preds
        )
        state.fifo = state.fifo[:, pop_out_len:]
        state.fifo_preds = state.fifo_preds[:, pop_out_len:]

        state.spkcache = np.concatenate([state.spkcache, pop_out_embs], axis=1)
        if state.spkcache_preds is not None:
            state.spkcache_preds = np.concatenate([state.spkcache_preds, pop_out_preds], axis=1)
        if state.spkcache.shape[1] > SPKCACHE_LEN:
            if state.spkcache_preds is None:  # first compression: seed preds for existing cache
                state.spkcache_preds = np.concatenate([preds[:, :spkcache_len], pop_out_preds], axis=1)
            state.spkcache, state.spkcache_preds = _compress_spkcache(
                state.spkcache, state.spkcache_preds, state.mean_sil_emb
            )

    return chunk_preds  # [1, chunk_len, n_spk] — the NEW frames to emit


def _get_silence_profile(mean_sil_emb, n_sil_frames, emb_seq, preds):
    is_sil = preds.sum(axis=2) < SIL_THRESHOLD
    sil_count = is_sil.sum(axis=1)
    if not (sil_count > 0).any():
        return mean_sil_emb, n_sil_frames
    sil_emb_sum = np.sum(emb_seq * is_sil[:, :, None], axis=1)
    upd_n_sil_frames = n_sil_frames + sil_count
    old_sil_emb_sum = mean_sil_emb * n_sil_frames[:, None]
    total = old_sil_emb_sum + sil_emb_sum
    upd_mean_sil_emb = total / np.clip(upd_n_sil_frames[:, None], 1, None)
    return upd_mean_sil_emb.astype(np.float32), upd_n_sil_frames.astype(np.int64)


def _get_log_pred_scores(preds: np.ndarray) -> np.ndarray:
    log_probs = np.log(np.clip(preds, PRED_SCORE_THRESHOLD, None))
    log_1_probs = np.log(np.clip(1.0 - preds, PRED_SCORE_THRESHOLD, None))
    log_1_probs_sum = log_1_probs.sum(axis=2)[:, :, None].repeat(N_SPK, axis=2)
    return log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)


def _disable_low_scores(preds: np.ndarray, scores: np.ndarray, min_pos_scores_per_spk: int) -> np.ndarray:
    is_speech = preds > 0.5
    scores = np.where(is_speech, scores, float("-inf"))
    is_pos = scores > 0
    is_nonpos_replace = (~is_pos) & is_speech & (is_pos.sum(axis=1)[:, None, :] >= min_pos_scores_per_spk)
    return np.where(is_nonpos_replace, float("-inf"), scores)


def _boost_topk_scores(scores: np.ndarray, n_boost_per_spk: int, scale_factor: float = 1.0, offset: float = 0.5) -> np.ndarray:
    """Boost the top-k scores per speaker (per column). Boost = subtract scale*log(offset)."""
    if n_boost_per_spk <= 0 or scores.shape[1] == 0:
        return scores
    n_frames = scores.shape[1]
    k = min(n_boost_per_spk, n_frames)
    # top-k indices per speaker (column); argpartition for the k largest.
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    rows = np.arange(scores.shape[0])[:, None]
    scores[rows, part, :] = scores[rows, part, :] - scale_factor * math.log(offset)
    return scores


def _get_topk_indices(scores: np.ndarray):
    batch_size, n_frames, _ = scores.shape
    n_frames_no_sil = n_frames - SPKCACHE_SIL_FRAMES_PER_SPK
    # Flatten (spk as the slow axis → permute to [batch, spk, frames] then ravel) to match NeMo.
    scores_flatten = np.transpose(scores, (0, 2, 1)).reshape(batch_size, -1)
    if scores_flatten.shape[1] < SPKCACHE_LEN:
        # Pad candidate pool so topk has enough slots.
        pad = np.full((batch_size, SPKCACHE_LEN - scores_flatten.shape[1]), float("-inf"), dtype=np.float64)
        scores_flatten = np.concatenate([scores_flatten.astype(np.float64), pad], axis=1)
    # top-SPKCACHE_LEN (unsorted); argpartition for largest.
    idx = np.argpartition(scores_flatten, -SPKCACHE_LEN, axis=1)[:, -SPKCACHE_LEN:]
    vals = np.take_along_axis(scores_flatten, idx, axis=1)
    valid = vals != float("-inf")
    topk_indices = np.where(valid, idx, MAX_INDEX)
    topk_indices_sorted = np.sort(topk_indices, axis=1)
    is_disabled = topk_indices_sorted == MAX_INDEX
    topk_indices_sorted = np.remainder(topk_indices_sorted, n_frames)
    is_disabled = is_disabled | (topk_indices_sorted >= n_frames_no_sil)
    topk_indices_sorted = np.where(is_disabled, 0, topk_indices_sorted)
    return topk_indices_sorted, is_disabled


def _gather_spkcache_and_preds(emb_seq, preds, topk_indices, is_disabled, mean_sil_emb):
    emb_dim = emb_seq.shape[2]
    idx_emb = topk_indices[:, :, None].repeat(emb_dim, axis=2)
    emb_gathered = np.take_along_axis(emb_seq, idx_emb, axis=1)
    sil_expanded = mean_sil_emb[:, None, :].repeat(SPKCACHE_LEN, axis=1)
    emb_gathered = np.where(is_disabled[:, :, None], sil_expanded, emb_gathered)
    idx_spk = topk_indices[:, :, None].repeat(N_SPK, axis=2)
    preds_gathered = np.take_along_axis(preds, idx_spk, axis=1)
    preds_gathered = np.where(is_disabled[:, :, None], 0.0, preds_gathered)
    return emb_gathered, preds_gathered


def _compress_spkcache(emb_seq: np.ndarray, preds: np.ndarray, mean_sil_emb: np.ndarray):
    """Keep the SPKCACHE_LEN most informative frames; return (spkcache, spkcache_preds)."""
    n_frames = preds.shape[1]
    spkcache_len_per_spk = SPKCACHE_LEN // N_SPK - SPKCACHE_SIL_FRAMES_PER_SPK
    strong_boost = math.floor(spkcache_len_per_spk * STRONG_BOOST_RATE)
    weak_boost = math.floor(spkcache_len_per_spk * WEAK_BOOST_RATE)
    min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * MIN_POS_SCORES_RATE)

    scores = _get_log_pred_scores(preds)
    scores = _disable_low_scores(preds, scores, min_pos_scores_per_spk)
    # permute_spk=False (inference) → no speaker permutation.
    scores = scores.astype(np.float64)  # stable -inf arithmetic
    if SCORES_BOOST_LATEST > 0:
        if n_frames > SPKCACHE_LEN:
            scores[:, SPKCACHE_LEN:, :] += SCORES_BOOST_LATEST
    scores = _boost_topk_scores(scores, strong_boost, scale_factor=2.0)
    scores = _boost_topk_scores(scores, weak_boost, scale_factor=1.0)
    if SPKCACHE_SIL_FRAMES_PER_SPK > 0:
        pad = np.full((1, SPKCACHE_SIL_FRAMES_PER_SPK, N_SPK), float("inf"))
        scores = np.concatenate([scores, pad], axis=1)
    topk_indices, is_disabled = _get_topk_indices(scores)
    spkcache, spkcache_preds = _gather_spkcache_and_preds(
        emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
    )
    return spkcache.astype(np.float32), spkcache_preds.astype(np.float32)


__all__ = [
    "StreamingState",
    "init_streaming_state",
    "streaming_update",
    "N_SPK",
    "FC_D_MODEL",
    "SPKCACHE_LEN",
    "FIFO_LEN",
]
