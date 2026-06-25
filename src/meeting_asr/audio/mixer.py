"""Audio mixer: resample → 16 kHz mono, single monotonic session clock (T012).

T012 implements **single-source passthrough**: normalize one source to 16 kHz
mono float32 and stamp each block onto a sample-count-driven session clock that
is continuous and monotonic (gaps preserved only when explicitly skipped).

Multi-source merge (FR-009) is added in T031 (US2) via :meth:`AudioMixer.merge`.

All resampling goes through :func:`soxr.resample` (high quality). The mixer never
silently drops audio under backpressure — the pipeline buffers and signals lag
(FR-016, FR-021); that policy lives in ``pipeline.py``, not here.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .._logging import get_logger
from ..types import AudioFrame, AudioSourceKind, CaptureState

_log = get_logger("audio.mixer")

SAMPLE_RATE = 16000  # internal pipeline rate (constitution: 16 kHz mono)

# Scaling from integer PCM → float32 [-1, 1].
_INT_SCALES = {np.dtype(np.int16): 32768.0, np.dtype(np.int32): 2147483648.0, np.dtype(np.uint8): 128.0}


def _to_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert any numeric PCM to float32 in [-1, 1]."""
    if pcm.dtype == np.float32 or pcm.dtype == np.float64:
        return np.asarray(pcm, dtype=np.float32)
    if pcm.dtype in _INT_SCALES:
        # center unsigned 8-bit at 0
        data = pcm.astype(np.float32)
        if pcm.dtype == np.uint8:
            data -= 128.0
        return data / _INT_SCALES[pcm.dtype]
    # last resort
    return np.asarray(pcm, dtype=np.float32)


def to_mono(pcm: np.ndarray) -> np.ndarray:
    """Collapse to mono float32. Accepts (n,) or (n, channels); averages channels."""
    f = _to_float32(np.asarray(pcm))
    if f.ndim == 1:
        return f
    if f.ndim == 2:
        # (n, channels) → average across channels
        return np.mean(f, axis=1).astype(np.float32)
    raise ValueError(f"unsupported pcm shape {f.shape}; expected 1-D or 2-D")


def resample(pcm: np.ndarray, from_rate: int, to_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono float32 PCM by ratio, returning float32."""
    mono = to_mono(pcm) if np.asarray(pcm).ndim > 1 else _to_float32(np.asarray(pcm))
    if from_rate == to_rate:
        return np.asarray(mono, dtype=np.float32)
    try:
        import soxr

        return soxr.resample(mono, from_rate, to_rate).astype(np.float32)
    except Exception as e:  # pragma: no cover - soxr missing in odd envs
        # Linear fallback so the pipeline never hard-fails on a missing lib.
        _log.warning("soxr unavailable (%s); using linear resample", e)
        ratio = to_rate / from_rate
        n_out = int(round(len(mono) * ratio))
        idx = np.linspace(0, len(mono) - 1, n_out)
        return np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)


def normalize_block(raw_pcm: np.ndarray, sample_rate: int, channels: int = 1) -> np.ndarray:
    """Full normalization: dtype → float32, downmix → mono, resample → 16 kHz."""
    block = np.asarray(raw_pcm)
    mono = to_mono(block) if block.ndim > 1 else _to_float32(block)
    _ = channels  # channels already encoded in block.shape; accepted for API clarity
    return resample(mono, sample_rate, SAMPLE_RATE)


class AudioMixer:
    """Drives the session clock and normalizes audio to 16 kHz mono float32.

    The clock is sample-count driven: each :meth:`feed` advances the source's
    emitted-sample counter, so timestamps are continuous and monotonic regardless
    of wall-clock jitter. Gaps (silence/dropout) are preserved, never collapsed.
    """

    def __init__(self, target_rate: int = SAMPLE_RATE) -> None:
        self.target_rate = target_rate
        self._samples_per_source: dict[AudioSourceKind, int] = {}

    # ----- single-source passthrough (T012) -----

    def feed(
        self,
        raw_pcm: np.ndarray,
        sample_rate: int,
        *,
        source: AudioSourceKind,
        channels: int = 1,
    ) -> AudioFrame:
        """Normalize one block and stamp it on the session clock; return a frame."""
        pcm = normalize_block(raw_pcm, sample_rate, channels=channels)
        emitted = self._samples_per_source.get(source, 0)
        t_start = emitted / self.target_rate
        t_end = (emitted + len(pcm)) / self.target_rate
        self._samples_per_source[source] = emitted + len(pcm)
        return AudioFrame(pcm=np.asarray(pcm, dtype=np.float32), t_start=t_start, t_end=t_end, source=source)

    def session_time(self, source: AudioSourceKind = AudioSourceKind.MICROPHONE) -> float:
        """Current session-clock time for a source (seconds)."""
        return self._samples_per_source.get(source, 0) / self.target_rate


# --------------------------------------------------------------------------- #
# Multi-source merge (T031; FR-009) — merges enabled sources onto one clock
# --------------------------------------------------------------------------- #


class MultiSourceMixer:
    """Merge several normalized sources onto one shared session clock (FR-009).

    Each feed is an :class:`AudioFrame` (16 kHz mono, session-clock t_start/t_end)
    from any source. Samples at the same session time are **summed** (overlap →
    louder, both voices preserved); each emitted mixed chunk carries the
    **dominant** (highest-energy) source tag for best-effort origin labeling.

    Streaming-safe: a chunk is emitted only once a later chunk has started (so the
    current tail can still accumulate), plus :meth:`flush` at end-of-stream.
    """

    def __init__(self, target_rate: int = SAMPLE_RATE, chunk_seconds: float = 0.1) -> None:
        self.rate = target_rate
        self.chunk_n = max(1, int(chunk_seconds * target_rate))
        self._pcm: dict = {}       # chunk_idx -> np.ndarray(chunk_n,) accumulator
        self._energy: dict = {}    # chunk_idx -> {AudioSourceKind: energy}
        self._next_emit = 0        # next contiguous chunk index to emit
        self._max_idx = -1

    def feed(self, frame: AudioFrame) -> List[AudioFrame]:
        off = round(frame.t_start * self.rate)
        n = len(frame.pcm)
        src = frame.source
        pcm = np.asarray(frame.pcm, dtype=np.float32)
        i = 0
        while i < n:
            abs_s = off + i
            cidx = abs_s // self.chunk_n
            local = abs_s - cidx * self.chunk_n
            take = min(self.chunk_n - local, n - i)
            seg = pcm[i : i + take]
            if cidx not in self._pcm:
                self._pcm[cidx] = np.zeros(self.chunk_n, dtype=np.float32)
                self._energy[cidx] = {}
            self._pcm[cidx][local : local + take] += seg
            e = self._energy[cidx]
            e[src] = e.get(src, 0.0) + float(np.sum(seg.astype(np.float32) ** 2))
            if cidx > self._max_idx:
                self._max_idx = cidx
            i += take
        return self._emit()

    def _emit(self) -> List[AudioFrame]:
        out: List[AudioFrame] = []
        # Emit contiguous chunks strictly below the highest seen index (tail held).
        while self._next_emit < self._max_idx and self._next_emit in self._pcm:
            out.append(self._finalize(self._next_emit))
            self._next_emit += 1
        return out

    def flush(self) -> List[AudioFrame]:
        out: List[AudioFrame] = []
        while self._next_emit in self._pcm:
            out.append(self._finalize(self._next_emit))
            self._next_emit += 1
        return out

    def _finalize(self, cidx: int) -> AudioFrame:
        pcm = np.clip(self._pcm.pop(cidx), -1.0, 1.0).astype(np.float32)
        energy = self._energy.pop(cidx)
        dom = max(energy, key=energy.get) if energy else AudioSourceKind.MICROPHONE
        t0 = cidx * self.chunk_n / self.rate
        return AudioFrame(pcm=pcm, t_start=t0, t_end=t0 + self.chunk_n / self.rate, source=dom)


class CompositeCapture:
    """AudioCapture-conforming merger of several captures on one clock (FR-009).

    Starts every sub-capture; each sub-frame feeds a :class:`MultiSourceMixer`,
    whose merged chunks are forwarded to the outer ``on_frame``. Per-frame source
    tags are preserved by the mixer (dominant source per chunk).
    """

    def __init__(self, captures: list, mixer: Optional[MultiSourceMixer] = None) -> None:
        from .capture import AudioCapture  # noqa: F401 (protocol reference)
        if not captures:
            raise ValueError("CompositeCapture needs at least one capture")
        self._captures = list(captures)
        self.kind = self._captures[0].kind  # nominal; per-frame tags carry real origin
        self._mixer = mixer or MultiSourceMixer()
        self._on_frame: Optional[Callable[[AudioFrame], None]] = None
        self._lock = __import__("threading").Lock()

    def permission_status(self) -> bool:
        return all(c.permission_status() for c in self._captures)

    def state(self) -> "CaptureState":  # type: ignore[name-defined]
        states = [c.state() for c in self._captures]
        if any(s == CaptureState.ERROR for s in states):
            return CaptureState.ERROR
        if any(s == CaptureState.PERMISSION_DENIED for s in states):
            return CaptureState.PERMISSION_DENIED
        if all(s == CaptureState.IDLE for s in states):
            return CaptureState.IDLE
        return CaptureState.CAPTURING

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        self._on_frame = on_frame
        # Start each sub-capture; permission/device failures propagate (FR-015).
        for cap in self._captures:
            cap.start(self._on_sub_frame)

    def _on_sub_frame(self, frame: AudioFrame) -> None:
        with self._lock:
            mixed = self._mixer.feed(frame)
        for m in mixed:
            if self._on_frame is not None:
                self._on_frame(m)

    def stop(self) -> None:
        for cap in self._captures:
            try:
                cap.stop()
            except Exception:
                _log.exception("error stopping sub-capture")
        with self._lock:
            tail = self._mixer.flush()
        for m in tail:
            if self._on_frame is not None:
                self._on_frame(m)


__all__ = [
    "SAMPLE_RATE",
    "AudioMixer",
    "MultiSourceMixer",
    "CompositeCapture",
    "normalize_block",
    "resample",
    "to_mono",
]
