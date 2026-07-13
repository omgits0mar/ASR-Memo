"""SortformerCoreMLDiarizer — Streaming Sortformer via CoreML (task T020).

Drives the CoreML build of NVIDIA Streaming Sortformer 4spk-v2.1
(``FluidInference/diar-streaming-sortformer-coreml`` → ``Sortformer.mlpackage``)
for real-time speaker diarization on Apple Silicon.

The export is a cache-aware streaming FastConformer-Sortformer:

* **mel front end** (128 bins, 25 ms / 10 ms) — shared with the ASR path
  (:class:`meeting_asr.asr._features.LogMelFrontEnd`).
* **CoreML forward** — ``chunk`` [1,112,128] mel + carried ``spkcache`` [1,188,512]
  + ``fifo`` [1,40,512] (running lengths) → ``speaker_preds`` [1,242,4] (sigmoid
  probabilities over the packed [spkcache|fifo|chunk] window) +
  ``chunk_pre_encoder_embs`` [1,14,512] (the carried chunk).
* **streaming state** — FIFO append → overflow into the speaker cache → top-k
  compression back to 188, ported to numpy (:mod:`._streaming_state`; NeMo's
  ``SortformerModules`` logic, parameter-free — no torch/NeMo at runtime).

Each 480 ms chunk (48 mel frames) yields 6 diarization frames (80 ms each).
Per-frame active speakers = probs > 0.55 (the model applies sigmoid internally;
there is no dedicated silence channel — all-below-threshold ⇒ silence). Speaker
identities are stable across the session (Sortformer property) and labelled in
arrival order (FR-003, FR-004). ``coremltools`` is lazy-imported; the model opens
only when cached (``prepare_models()``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .._logging import ModelError, get_logger
from ..asr._features import LogMelFrontEnd
from ..models.registry import default_cache_dir, model_registry, refresh_state
from ..types import AudioFrame, ComputeBackend, DiarFrame, ModelKind
from . import _streaming_state as ss
from ._sortformer_decode import ACTIVATION_THRESHOLD, FRAME_SECONDS, SortformerFrameDecoder

_log = get_logger("diarization.sortformer")

_SAMPLE_RATE = 16000
# Export streaming profile (proven against the cached CoreML model spec).
CHUNK_LEN = 6  # diarization output frames per chunk
LEFT_CONTEXT = 1  # left context chunks
RIGHT_CONTEXT = 7  # right context chunks
SUBSAMPLING = 8
CORE_FRAMES = CHUNK_LEN * SUBSAMPLING  # 48 new mel frames per chunk (480 ms)
WINDOW_FRAMES = (LEFT_CONTEXT + CHUNK_LEN + RIGHT_CONTEXT) * SUBSAMPLING  # 112


class SortformerCoreMLDiarizer:
    """SpeakerDiarizer-conforming backend (Streaming Sortformer, CoreML)."""

    def __init__(self) -> None:
        self._model = None
        self._backend: Optional[ComputeBackend] = None
        self._model_path: Optional[Path] = None
        self._frontend: Optional[LogMelFrontEnd] = None
        # normalize='none' (the encoder takes raw log-mel; global per-feature stats are
        # incompatible with streaming — they'd differ per push — and 'none' matches the
        # whole-utterance path that scored DER 0).
        self._mel_normalize = "none"
        # streaming state
        self._state: Optional[ss.StreamingState] = None
        self._mel_bank: Optional[np.ndarray] = None  # [1, 128, T] accumulated mel
        self._mel_cursor = 0  # mel frames already chunked (stt_feat)
        self._pcm_tail = np.zeros(0, dtype=np.float32)  # un-mel'd PCM carry
        self._mel_seeded = False  # center=True left-reflect pad applied once
        self._decoder = SortformerFrameDecoder()

    # ---- SpeakerDiarizer protocol ----

    def load(self, backend: ComputeBackend) -> None:
        asset = next(a for a in model_registry() if a.kind is ModelKind.DIARIZER)
        asset = refresh_state(asset, default_cache_dir())
        if not asset.is_cached():
            raise ModelError(
                f"diarizer model '{asset.name}' not cached; run prepare_models() first"
            )
        self._model_path = Path(asset.cache_path) / asset.expected_files[0]
        try:
            import coremltools as ct  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ModelError("coremltools not installed (pip install coremltools)") from e
        try:
            # CPU_AND_GPU (not ALL): skips the slow ANE compilation pass that made a
            # cold .mlpackage load take ~77s. Matches the device resolver's
            # .cpuAndGPU choice; inference is unchanged (verified DER 0).
            self._model = ct.models.MLModel(str(self._model_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        except Exception as e:  # pragma: no cover - model dependent
            raise ModelError(f"failed to load CoreML model: {e}") from e
        self._frontend = LogMelFrontEnd(sample_rate=_SAMPLE_RATE)
        self._backend = backend
        self.reset()
        _log.info(
            "Sortformer CoreML loaded (chunk%d L%d R%d sub%d → %dms frames, threshold %.2f)",
            CHUNK_LEN, LEFT_CONTEXT, RIGHT_CONTEXT, SUBSAMPLING, FRAME_SECONDS * 1000, ACTIVATION_THRESHOLD,
        )

    def reset(self) -> None:
        self._state = ss.init_streaming_state()
        self._mel_bank = np.zeros((1, 128, 0), dtype=np.float32)
        self._mel_cursor = 0
        self._pcm_tail = np.zeros(0, dtype=np.float32)
        self._mel_seeded = False
        self._decoder.reset()

    def push(self, frame: AudioFrame) -> List[DiarFrame]:
        if self._model is None:
            raise ModelError("diarizer not loaded; call load() first")
        pcm = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        self._extend_mel(pcm)
        return self._drain_chunks()

    def flush(self) -> List[DiarFrame]:
        """Process any remaining buffered mel at end-of-stream."""
        if self._model is None:
            return []
        return self._drain_chunks()

    def max_speakers(self) -> int:
        return ss.N_SPK

    # ---- whole-utterance path (validation / file transcription) ----

    def diarize_array(self, audio: np.ndarray) -> List[DiarFrame]:
        """Diarize a complete 16 kHz mono signal in one pass (NeMo-faithful mel)."""
        if self._model is None or self._frontend is None:
            raise ModelError("diarizer not loaded; call load() first")
        self.reset()
        feats = self._frontend(np.asarray(audio, dtype=np.float32), normalize=self._mel_normalize)
        self._mel_bank = feats[None, :, :]  # [1, 128, T]
        self._mel_cursor = 0
        return self._drain_chunks()

    # ---- mel + streaming chunk drain ----

    def _extend_mel(self, pcm: np.ndarray) -> None:
        """Append PCM and extend the mel bank with center=True-aligned continuous framing.

        NeMo's preprocessor center-pads (reflect, ``n_fft//2``), so streaming frames must
        carry the same left alignment. We seed the left reflection once, then advance with
        an ``n_fft-hop`` carry so each push whose PCM is a multiple of ``hop`` contributes
        exactly ``len(pcm)//hop`` new frames with no gaps or duplicates.
        """
        fe = self._frontend
        if not self._mel_seeded and len(pcm) > 0:
            pad = fe.n_fft // 2
            head = pcm[:pad]
            refl = head[::-1] if len(head) == pad else np.pad(pcm, (pad - len(head), 0))[::-1]
            pcm = np.concatenate([refl, pcm])
            self._mel_seeded = True
        carry = fe.n_fft - fe.hop_length
        x = np.concatenate([self._pcm_tail, pcm])
        if len(x) < fe.n_fft:
            self._pcm_tail = x
            return
        feats = fe.frames(x, normalize=self._mel_normalize)  # [128, T_new]
        self._mel_bank = np.concatenate([self._mel_bank, feats[None, :, :]], axis=2)
        self._pcm_tail = x[-carry:]

    def _drain_chunks(self) -> List[DiarFrame]:
        """Process every complete CORE_FRAMES chunk currently available in the mel bank."""
        out: List[DiarFrame] = []
        total = self._mel_bank.shape[2]
        while self._mel_cursor + CORE_FRAMES <= total:
            out.extend(self._process_chunk(self._mel_cursor))
            self._mel_cursor += CORE_FRAMES
        return out

    def _process_chunk(self, stt_feat: int) -> List[DiarFrame]:
        end_feat = stt_feat + CORE_FRAMES
        total = self._mel_bank.shape[2]
        left_offset = min(LEFT_CONTEXT * SUBSAMPLING, stt_feat)
        right_offset = min(RIGHT_CONTEXT * SUBSAMPLING, total - end_feat)
        a, b = stt_feat - left_offset, end_feat + right_offset
        chunk = self._mel_bank[:, :, a:b].transpose(0, 2, 1)  # [1, ≤112, 128]
        actual_len = chunk.shape[1]
        if actual_len < WINDOW_FRAMES:  # right-pad to the fixed CoreML window
            chunk = np.pad(chunk, ((0, 0), (0, WINDOW_FRAMES - actual_len), (0, 0)))
        lc = round(left_offset / SUBSAMPLING)
        rc = int(np.ceil(right_offset / SUBSAMPLING))

        feed = {
            "chunk": np.ascontiguousarray(chunk, dtype=np.float32),
            "chunk_lengths": np.array([actual_len], dtype=np.int32),
            "spkcache": self._padded(self._state.spkcache, ss.SPKCACHE_LEN),
            "spkcache_lengths": np.array([self._state.spkcache_len], dtype=np.int32),
            "fifo": self._padded(self._state.fifo, ss.FIFO_LEN),
            "fifo_lengths": np.array([self._state.fifo_len], dtype=np.int32),
        }
        try:
            out = self._model.predict(feed)  # type: ignore[union-attr]
        except Exception as e:  # pragma: no cover - model dependent
            _log.warning("Sortformer predict failed: %s", e)
            return []
        preds = np.asarray(out["speaker_preds"], dtype=np.float32)  # [1, 242, 4]
        emb_len = int(np.asarray(out["chunk_pre_encoder_lengths"]).reshape(-1)[0])
        chunk_embs = np.asarray(out["chunk_pre_encoder_embs"], dtype=np.float32)[:, :emb_len, :]

        chunk_preds = ss.streaming_update(self._state, chunk_embs, preds, lc=lc, rc=rc)
        return self._decode(chunk_preds)

    @staticmethod
    def _padded(buf: np.ndarray, fixed_len: int) -> np.ndarray:
        """Right-pad a [1, L, D] logical buffer to [1, fixed_len, D] (zeros)."""
        cur = buf.shape[1]
        if cur == fixed_len:
            return np.ascontiguousarray(buf, dtype=np.float32)
        if cur == 0:
            return np.zeros((1, fixed_len, ss.FC_D_MODEL), dtype=np.float32)
        out = np.zeros((1, fixed_len, buf.shape[2]), dtype=np.float32)
        out[:, :cur, :] = buf
        return np.ascontiguousarray(out, dtype=np.float32)

    # ---- decode → DiarFrames ----

    def _decode(self, chunk_preds: np.ndarray) -> List[DiarFrame]:
        """[1, chunk_len, n_spk] probs → DiarFrames (80 ms each), arrival-order labels."""
        return self._decoder.decode(chunk_preds)

    def _label_for(self, raw_id: int) -> str:
        """Map a stable Sortformer speaker slot (0..3) to an arrival-order 'Speaker N'."""
        return self._decoder.label_for(raw_id)


__all__ = ["SortformerCoreMLDiarizer"]
