"""Shared Sortformer probability decoding."""

from __future__ import annotations

from typing import List

import numpy as np

from ..types import DiarFrame

FRAME_SECONDS = 0.08
ACTIVATION_THRESHOLD = 0.55


class SortformerFrameDecoder:
    """Decode Sortformer speaker probabilities to arrival-order ``DiarFrame``s."""

    def __init__(
        self,
        *,
        frame_seconds: float = FRAME_SECONDS,
        threshold: float = ACTIVATION_THRESHOLD,
    ) -> None:
        self.frame_seconds = frame_seconds
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.emit_idx = 0
        self._aosc_next = 1
        self._label_map: dict[int, str] = {}

    def decode(self, chunk_preds: np.ndarray) -> List[DiarFrame]:
        """``[..., n_spk]`` probabilities -> dominant-speaker frames."""
        arr = np.asarray(chunk_preds, dtype=np.float32)
        if arr.size == 0:
            return []
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)

        out: List[DiarFrame] = []
        for probs in arr:
            t0 = self.emit_idx * self.frame_seconds
            t1 = t0 + self.frame_seconds
            self.emit_idx += 1
            active = np.where(probs > self.threshold)[0]
            if active.size == 0:
                continue
            spk_raw = int(active[int(np.argmax(probs[active]))])
            score = float(probs[spk_raw])
            out.append(
                DiarFrame(t_start=t0, t_end=t1, speaker_label=self.label_for(spk_raw), score=score)
            )
        return out

    def label_for(self, raw_id: int) -> str:
        """Map a stable Sortformer speaker slot to arrival-order ``Speaker N``."""
        if raw_id not in self._label_map:
            self._label_map[raw_id] = f"Speaker {self._aosc_next}"
            self._aosc_next += 1
        return self._label_map[raw_id]


__all__ = ["ACTIVATION_THRESHOLD", "FRAME_SECONDS", "SortformerFrameDecoder"]
