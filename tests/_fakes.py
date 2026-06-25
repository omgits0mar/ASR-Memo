"""Offline fakes conforming to the meeting_asr protocols (used by US1–US4 tests).

These stand in for live capture / real Nemotron+Sortformer models so the entire
pipeline (capture → mix → diarize → transcribe → fuse → session) is exercised
deterministically and offline. They replay a *manifest* of known speaker turns
with configurable latency, mirroring the real backends' streaming contracts.

Real-model WER / language-ID / DER accuracy is gated separately (needs_models).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

from meeting_asr.audio.mixer import SAMPLE_RATE, AudioMixer
from meeting_asr.types import (
    AsrToken,
    AudioFrame,
    AudioSourceKind,
    CaptureState,
    ComputeBackend,
    DiarFrame,
)


def _load_wav_mono(path: str, rate: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(path, dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != rate:
        import soxr

        pcm = soxr.resample(pcm.astype(np.float32), sr, rate)
    return pcm.astype(np.float32)


# --------------------------------------------------------------------------- #
# Fake capture: reads a WAV and streams normalized AudioFrames
# --------------------------------------------------------------------------- #


class FixtureCapture:
    """AudioCapture-conforming fake: streams a WAV file as 16 kHz mono frames."""

    def __init__(self, wav_path: str, *, source: AudioSourceKind = AudioSourceKind.MICROPHONE,
                 block_seconds: float = 0.1, realtime: bool = False) -> None:
        self.kind = source
        self._wav = wav_path
        self._block = max(1, int(block_seconds * SAMPLE_RATE))
        self._realtime = realtime
        self._state = CaptureState.IDLE
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mixer = AudioMixer()

    def permission_status(self) -> bool:
        return True

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        pcm = _load_wav_mono(self._wav)
        self._state = CaptureState.CAPTURING
        self._stop.clear()

        def _run():
            n = len(pcm)
            i = 0
            while i < n and not self._stop.is_set():
                block = pcm[i : i + self._block]
                if len(block) == 0:
                    break
                frame = self._mixer.feed(block, SAMPLE_RATE, source=self.kind)
                on_frame(frame)
                i += self._block
                if self._realtime:
                    time.sleep(self._block / SAMPLE_RATE)
            self._state = CaptureState.IDLE if not self._stop.is_set() else self._state

        self._thread = threading.Thread(target=_run, name="FixtureCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._state = CaptureState.IDLE
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# Fake diarizer: replays manifest turns as 80ms DiarFrames (with latency)
# --------------------------------------------------------------------------- #


class ManifestDiarizer:
    """SpeakerDiarizer-conforming fake: emits frame-level decisions from a manifest."""

    def __init__(self, turns: List[dict], *, latency_s: float = 1.0,
                 frame_s: float = 0.08) -> None:
        self._turns = turns
        self._latency = latency_s
        self._frame_s = frame_s
        self._next_idx = 0  # index-based emission (no re-emit on float ties)
        self._backend: Optional[ComputeBackend] = None
        self._frames: List[DiarFrame] = self._materialize()

    def _materialize(self) -> List[DiarFrame]:
        frames: List[DiarFrame] = []
        for turn in self._turns:
            t = turn["t_start"]
            # Only emit full frame_s frames; drop the trailing sub-frame partial.
            while t + self._frame_s <= turn["t_end"] + 1e-9:
                frames.append(DiarFrame(t_start=t, t_end=t + self._frame_s, speaker_label=turn["speaker"], score=0.95))
                t += self._frame_s
        frames.sort(key=lambda f: f.t_start)
        return frames

    def load(self, backend: ComputeBackend) -> None:
        self._backend = backend

    def reset(self) -> None:
        self._next_idx = 0

    def push(self, frame: AudioFrame) -> List[DiarFrame]:
        # Emit decisions whose t_end is at least `latency_s` behind the audio frontier,
        # mimicking the streaming diarizer's look-back. Index-based → each emitted once.
        horizon = max(0.0, frame.t_end - self._latency)
        out: List[DiarFrame] = []
        while self._next_idx < len(self._frames) and self._frames[self._next_idx].t_end <= horizon + 1e-9:
            out.append(self._frames[self._next_idx])
            self._next_idx += 1
        return out

    def max_speakers(self) -> int:
        return 4


# --------------------------------------------------------------------------- #
# Fake transcriber: replays manifest text as word-level AsrTokens (with latency)
# --------------------------------------------------------------------------- #


class ManifestTranscriber:
    """SpeechTranscriber-conforming fake: emits word tokens from a manifest."""

    def __init__(self, turns: List[dict], *, latency_s: float = 0.56) -> None:
        self._latency = latency_s
        self._next_idx = 0  # index-based emission (no re-emit on float ties)
        self._backend: Optional[ComputeBackend] = None
        self._precision = "fp16"
        self._tokens: List[AsrToken] = self._materialize(turns)

    @staticmethod
    def _materialize(turns: List[dict]) -> List[AsrToken]:
        toks: List[AsrToken] = []
        for turn in turns:
            words = turn["text"].split()
            span = max(turn["t_end"] - turn["t_start"], 1e-3)
            step = span / max(len(words), 1)
            for i, w in enumerate(words):
                t0 = turn["t_start"] + i * step
                t1 = turn["t_start"] + (i + 1) * step
                toks.append(AsrToken(text=w, t_start=t0, t_end=t1, language=turn.get("language"), score=0.96))
        toks.sort(key=lambda t: t.t_start)
        return toks

    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        self._backend = backend
        self._precision = precision

    def reset(self) -> None:
        self._next_idx = 0

    def push(self, frame: AudioFrame, *, language_hint: str | None = None) -> List[AsrToken]:
        horizon = max(0.0, frame.t_end - self._latency)
        out: List[AsrToken] = []
        while self._next_idx < len(self._tokens) and self._tokens[self._next_idx].t_end <= horizon + 1e-9:
            out.append(self._tokens[self._next_idx])
            self._next_idx += 1
        return out

    def flush(self) -> List[AsrToken]:
        out = self._tokens[self._next_idx:]
        self._next_idx = len(self._tokens)
        return out

    def supported_languages(self) -> List[str]:
        return sorted({t.language for t in self._tokens if t.language})


class SilentDiarizer:
    """Emits no diarization (silence edge case)."""

    def load(self, backend: ComputeBackend) -> None:
        pass

    def reset(self) -> None:
        pass

    def push(self, frame: AudioFrame) -> List[DiarFrame]:
        return []

    def max_speakers(self) -> int:
        return 4


class SilentTranscriber:
    """Emits no tokens (silence edge case)."""

    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        pass

    def reset(self) -> None:
        pass

    def push(self, frame: AudioFrame, *, language_hint: str | None = None) -> List[AsrToken]:
        return []

    def flush(self) -> List[AsrToken]:
        return []

    def supported_languages(self) -> List[str]:
        return []
