"""SortformerNeMoDiarizer — native NeMo/PyTorch diarization backend."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .._logging import ModelError, get_logger
from ..models.registry import default_cache_dir, model_registry, refresh_state
from ..types import AudioFrame, ComputeBackend, DiarFrame, ModelKind
from ._sortformer_decode import FRAME_SECONDS, SortformerFrameDecoder

_log = get_logger("diarization.sortformer_nemo")

_SAMPLE_RATE = 16000
_MAX_SPEAKERS = 4


class SortformerNeMoDiarizer:
    """SpeakerDiarizer-conforming backend for native NeMo Sortformer."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._emitted_frames = 0
        self._decoder = SortformerFrameDecoder()

    def load(self, backend: ComputeBackend) -> None:
        if backend not in (ComputeBackend.CUDA, ComputeBackend.TORCH_CPU):
            raise ModelError(f"NeMo diarizer backend does not support {backend.value}")

        asset = next(a for a in model_registry(backend) if a.kind is ModelKind.DIARIZER)
        asset = refresh_state(asset, default_cache_dir())
        if not asset.is_cached():
            raise ModelError(
                f"diarizer model '{asset.name}' not cached; run prepare_models() first"
            )

        model_path = Path(asset.cache_path) / asset.expected_files[0]
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ModelError("NeMo diarization not installed (pip install meeting-asr[nemo])") from e

        self._device = "cuda" if backend is ComputeBackend.CUDA else "cpu"
        try:
            self._model = SortformerEncLabelModel.restore_from(
                restore_path=str(model_path),
                map_location=self._device,
                strict=False,
            )
        except TypeError:
            self._model = SortformerEncLabelModel.restore_from(str(model_path), map_location=self._device)
        except Exception as e:  # pragma: no cover - model dependent
            raise ModelError(f"failed to restore NeMo Sortformer model: {e}") from e

        self._move_model(self._model, self._device)
        self._configure_low_latency(self._model)
        self.reset()
        _log.info("Sortformer NeMo loaded (%s)", self._device)

    def reset(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._emitted_frames = 0
        self._decoder.reset()

    def push(self, frame: AudioFrame) -> List[DiarFrame]:
        if self._model is None:
            raise ModelError("diarizer not loaded; call load() first")
        pcm = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        self._audio_buf = np.concatenate([self._audio_buf, pcm])
        probs = self._probabilities_for_audio(self._audio_buf)
        return self._emit_new(probs)

    def flush(self) -> List[DiarFrame]:
        if self._model is None:
            return []
        probs = self._probabilities_for_audio(self._audio_buf)
        return self._emit_new(probs)

    def max_speakers(self) -> int:
        return _MAX_SPEAKERS

    def diarize_array(self, audio: np.ndarray) -> List[DiarFrame]:
        if self._model is None:
            raise ModelError("diarizer not loaded; call load() first")
        probs = self._probabilities_for_audio(np.asarray(audio, dtype=np.float32))
        decoder = SortformerFrameDecoder()
        return decoder.decode(probs)

    def _emit_new(self, probs: np.ndarray) -> List[DiarFrame]:
        if probs.size == 0:
            return []
        frames = np.asarray(probs, dtype=np.float32).reshape(-1, probs.shape[-1])
        new = frames[self._emitted_frames :]
        self._emitted_frames = len(frames)
        return self._decoder.decode(new)

    def _probabilities_for_audio(self, audio: np.ndarray) -> np.ndarray:
        result = self._diarize_audio(audio)
        probs = self._extract_probs(result)
        if probs is not None:
            return probs
        return self._segments_to_probs(self._extract_segments(result))

    def _diarize_audio(self, audio: np.ndarray) -> Any:
        model = self._model
        attempts = (
            lambda: model.diarize(
                audio=[audio],
                batch_size=1,
                sample_rate=_SAMPLE_RATE,
                include_tensor_outputs=True,
            ),
            lambda: model.diarize(audio=[audio], batch_size=1, sample_rate=_SAMPLE_RATE),
            lambda: model.diarize(audio=[audio], batch_size=1),
            lambda: model.diarize(audio),
        )
        last: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as e:
                last = e
                continue
        raise ModelError(f"NeMo diarize API not compatible: {last}")

    @staticmethod
    def _move_model(model: Any, device: str) -> None:
        to = getattr(model, "to", None)
        if callable(to):
            to(device)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @staticmethod
    def _configure_low_latency(model: Any) -> None:
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            return
        for name, value in (
            ("chunk_len", 6),
            ("chunk_right_context", 7),
            ("fifo_len", 188),
            ("spkcache_update_period", 144),
            ("spkcache_len", 188),
        ):
            try:
                setattr(modules, name, value)
            except Exception:
                _log.debug("could not set Sortformer module %s", name)
        check = getattr(modules, "_check_streaming_parameters", None)
        if callable(check):
            check()

    @staticmethod
    def _extract_probs(result: Any) -> Optional[np.ndarray]:
        cur = result
        if isinstance(cur, tuple) and len(cur) >= 2:
            cur = cur[1]
        elif isinstance(cur, dict):
            cur = cur.get("probs")
            if cur is None:
                cur = cur.get("predicted_probs")
            if cur is None:
                cur = cur.get("speaker_probs")
        else:
            return None

        if isinstance(cur, list):
            if not cur:
                return np.zeros((0, _MAX_SPEAKERS), dtype=np.float32)
            cur = cur[0]
        arr = np.asarray(cur, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _extract_segments(result: Any) -> list:
        cur = result[0] if isinstance(result, tuple) else result
        if isinstance(cur, dict):
            cur = cur.get("segments") or cur.get("predicted_segments") or []
        if isinstance(cur, list) and cur and isinstance(cur[0], list):
            return cur[0]
        return list(cur or []) if isinstance(cur, (list, tuple)) else []

    @staticmethod
    def _segments_to_probs(segments: list) -> np.ndarray:
        parsed: list[tuple[float, float, int]] = []
        max_end = 0.0
        for segment in segments:
            item = SortformerNeMoDiarizer._parse_segment(segment)
            if item is None:
                continue
            start, end, speaker = item
            parsed.append(item)
            max_end = max(max_end, end)
        if not parsed:
            return np.zeros((0, _MAX_SPEAKERS), dtype=np.float32)
        n = max(1, int(np.ceil(max_end / FRAME_SECONDS)))
        probs = np.zeros((n, _MAX_SPEAKERS), dtype=np.float32)
        for start, end, speaker in parsed:
            a = max(0, int(np.floor(start / FRAME_SECONDS)))
            b = max(a + 1, int(np.ceil(end / FRAME_SECONDS)))
            probs[a:b, min(max(speaker, 0), _MAX_SPEAKERS - 1)] = 1.0
        return probs

    @staticmethod
    def _parse_segment(segment: Any) -> Optional[tuple[float, float, int]]:
        if isinstance(segment, dict):
            start = segment.get("start", segment.get("begin", segment.get("t_start")))
            end = segment.get("end", segment.get("t_end"))
            speaker = segment.get("speaker", segment.get("speaker_index", segment.get("label", 0)))
        else:
            parts = re.split(r"[\s,]+", str(segment).strip())
            if len(parts) < 3:
                return None
            start, end, speaker = parts[0], parts[1], parts[2]
        try:
            speaker_s = str(speaker)
            m = re.search(r"(\d+)$", speaker_s)
            return float(start), float(end), int(m.group(1)) if m else int(speaker_s)
        except (TypeError, ValueError):
            return None


__all__ = ["SortformerNeMoDiarizer"]
