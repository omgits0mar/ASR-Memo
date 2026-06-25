"""Log-mel front end for the Nemotron ONNX encoder (002 / Decision 4).

Reproduces NVIDIA NeMo's ``AudioToMelSpectrogramPreprocessor`` numerically so the
FP16 ONNX encoder sees the features it was traced with: pre-emphasis → STFT
(Hann, ``center=True``) → power spectrum → 128-bin slaney mel → log → per-feature
normalization. Parameters come from the export's ``config.json`` (n_fft 512, win
400, hop 160, pre-emph 0.97, 128 mel bins at 16 kHz).

The mel filterbank is a bundled ``mel_fb_128_512.npy`` (generated once with
``librosa.filters.mel`` — slaney norm, ``htk=False``) so the runtime stays
``numpy``-only with no ``librosa`` dependency. ``numpy.fft`` does the STFT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_FB_PATH = Path(__file__).with_name("mel_fb_128_512.npy")
_LOG_GUARD = 2.0**-24  # NeMo log_zero_guard_value (add)


class LogMelFrontEnd:
    """NeMo-faithful 128-bin log-mel extractor (numpy)."""

    def __init__(
        self,
        *,
        n_mels: int = 128,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemph: float = 0.97,
        sample_rate: int = 16000,
    ) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemph = preemph
        self.sample_rate = sample_rate
        # Hann window (symmetric, length win_length) zero-padded/centered to n_fft.
        hann = np.hanning(win_length + 1)[:-1].astype(np.float32)  # periodic=False
        pad = (n_fft - win_length) // 2
        self._window = np.zeros(n_fft, dtype=np.float32)
        self._window[pad : pad + win_length] = hann
        self._fb = self._load_filterbank(n_mels, n_fft)

    @staticmethod
    def _load_filterbank(n_mels: int, n_fft: int) -> np.ndarray:
        if not _FB_PATH.exists():  # pragma: no cover - packaging guard
            raise FileNotFoundError(
                f"mel filterbank not found at {_FB_PATH}; regenerate with scripts/gen_mel_fb.py"
            )
        fb = np.load(_FB_PATH).astype(np.float32)
        if fb.shape != (n_mels, n_fft // 2 + 1):  # pragma: no cover - packaging guard
            raise ValueError(f"mel filterbank shape {fb.shape} != {(n_mels, n_fft // 2 + 1)}")
        return fb

    def __call__(self, audio: np.ndarray, *, normalize: str = "per_feature") -> np.ndarray:
        """Whole-utterance log-mel (``center=True``) → ``[n_mels, T]`` (T = 1 + N//hop)."""
        return self._logmel(audio, center=True, normalize=normalize)

    def frames(self, audio: np.ndarray, *, normalize: str = "per_feature") -> np.ndarray:
        """Streaming log-mel (``center=False``) → ``[n_mels, 1 + (N-n_fft)//hop]``.

        Caller prepends ``n_fft - hop`` samples of carried context so successive
        chunks tile without gaps and a 320 ms (32-hop) advance yields 32 frames.
        """
        return self._logmel(audio, center=False, normalize=normalize)

    def _logmel(self, audio: np.ndarray, *, center: bool, normalize: str) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return np.zeros((self.n_mels, 0), dtype=np.float32)
        # Pre-emphasis: y[0]=x[0]; y[t]=x[t]-preemph*x[t-1].
        if self.preemph:
            x = np.concatenate([x[:1], x[1:] - self.preemph * x[:-1]]).astype(np.float32)
        spec = self._stft_power(x, center=center)  # [n_fft/2+1, T]
        mel = self._fb @ spec                      # [n_mels, T]
        feats = np.log(mel + _LOG_GUARD).astype(np.float32)
        if normalize == "per_feature":
            feats = self._normalize_per_feature(feats)
        return feats

    def _stft_power(self, x: np.ndarray, *, center: bool) -> np.ndarray:
        """STFT power spectrum ``|X|^2`` → ``[n_fft/2+1, T]``."""
        if center:
            pad = self.n_fft // 2
            x = np.pad(x, (pad, pad), mode="reflect")
        if len(x) < self.n_fft:
            x = np.pad(x, (0, self.n_fft - len(x)))
        n_frames = 1 + (len(x) - self.n_fft) // self.hop_length
        idx = np.arange(self.n_fft)[None, :] + self.hop_length * np.arange(n_frames)[:, None]
        frames = x[idx] * self._window[None, :]    # [T, n_fft]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)  # [T, n_fft/2+1]
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        return power.T                              # [n_fft/2+1, T]

    @staticmethod
    def _normalize_per_feature(feats: np.ndarray) -> np.ndarray:
        """Per-mel-bin mean/std normalization over time (NeMo 'per_feature')."""
        mean = feats.mean(axis=1, keepdims=True)
        std = feats.std(axis=1, ddof=1, keepdims=True) if feats.shape[1] > 1 else np.ones_like(mean)
        return ((feats - mean) / (std + 1e-5)).astype(np.float32)


__all__ = ["LogMelFrontEnd"]
