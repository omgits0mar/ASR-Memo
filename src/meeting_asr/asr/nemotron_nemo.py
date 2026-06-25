"""NemotronNeMoTranscriber — native NeMo/PyTorch ASR backend.

This adapter is intentionally lazy: torch/NeMo are imported only inside
``load()`` and model inference methods are called through small seams that are
easy to exercise with offline fakes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .._logging import ModelError, get_logger
from ..models.registry import NEMOTRON_LANGUAGES, default_cache_dir, model_registry, refresh_state
from ..types import AsrToken, AudioFrame, ComputeBackend, ModelKind

_log = get_logger("asr.nemotron_nemo")

_SAMPLE_RATE = 16000
_LANG_TAG = re.compile(r"\s*<([a-z]{2}(?:-[A-Z]{2})?)>\s*$")


class NemotronNeMoTranscriber:
    """SpeechTranscriber-conforming backend for native NeMo models."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._precision = "fp32"
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._emitted_count = 0

    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        if backend not in (ComputeBackend.CUDA, ComputeBackend.TORCH_CPU):
            raise ModelError(f"NeMo ASR backend does not support {backend.value}")
        if precision not in ("fp16", "fp32", "int8", "int4"):
            raise ModelError(f"unsupported precision '{precision}' (use fp16|fp32|int8|int4)")

        asset = next(a for a in model_registry(backend) if a.kind is ModelKind.ASR)
        asset = refresh_state(asset, default_cache_dir())
        if not asset.is_cached():
            raise ModelError(f"ASR model '{asset.name}' not cached; run prepare_models() first")

        model_path = Path(asset.cache_path) / asset.expected_files[0]
        try:
            from nemo.collections.asr.models import ASRModel  # type: ignore
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ModelError("NeMo ASR not installed (pip install meeting-asr[nemo])") from e

        self._device = "cuda" if backend is ComputeBackend.CUDA else "cpu"
        try:
            self._model = ASRModel.restore_from(
                restore_path=str(model_path),
                map_location=self._device,
            )
        except TypeError:
            self._model = ASRModel.restore_from(str(model_path), map_location=self._device)
        except Exception as e:  # pragma: no cover - model dependent
            raise ModelError(f"failed to restore NeMo ASR model: {e}") from e

        self._move_model(self._model, self._device)
        if self._device == "cuda" and precision == "fp16":
            self._try_half(self._model)
        self._precision = precision
        self.reset()
        _log.info("Nemotron NeMo loaded (%s, precision=%s)", self._device, precision)

    def reset(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._emitted_count = 0

    def push(self, frame: AudioFrame, *, language_hint: Optional[str] = None) -> List[AsrToken]:
        if self._model is None:
            raise ModelError("transcriber not loaded; call load() first")
        pcm = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        self._audio_buf = np.concatenate([self._audio_buf, pcm])
        tokens = self._tokens_for_audio(self._audio_buf, language_hint=language_hint)
        return self._emit_new(tokens)

    def flush(self) -> List[AsrToken]:
        if self._model is None:
            return []
        tokens = self._tokens_for_audio(self._audio_buf, language_hint=None)
        return self._emit_new(tokens)

    def supported_languages(self) -> List[str]:
        return list(NEMOTRON_LANGUAGES)

    def transcribe_array(self, audio: np.ndarray, *, language: Optional[str] = None) -> List[AsrToken]:
        if self._model is None:
            raise ModelError("transcriber not loaded; call load() first")
        return self._tokens_for_audio(np.asarray(audio, dtype=np.float32), language_hint=language)

    def _emit_new(self, tokens: List[AsrToken]) -> List[AsrToken]:
        out = tokens[self._emitted_count :]
        self._emitted_count = len(tokens)
        return out

    def _tokens_for_audio(
        self,
        audio: np.ndarray,
        *,
        language_hint: Optional[str],
    ) -> List[AsrToken]:
        if len(audio) == 0:
            return []
        result = self._transcribe_audio(audio, language_hint=language_hint)
        hyp = self._first_hypothesis(result)
        return self._extract_word_tokens(hyp, len(audio) / _SAMPLE_RATE, language_hint)

    def _transcribe_audio(self, audio: np.ndarray, *, language_hint: Optional[str]) -> Any:
        """Call NeMo ``transcribe`` while tolerating API differences across versions."""
        model = self._model
        target_lang = self._target_lang(language_hint)
        attempts = (
            lambda: model.transcribe(
                audio=[audio],
                batch_size=1,
                sample_rate=_SAMPLE_RATE,
                return_hypotheses=True,
                timestamps=True,
                target_lang=target_lang,
            ),
            lambda: model.transcribe(
                audio=[audio],
                batch_size=1,
                sample_rate=_SAMPLE_RATE,
                return_hypotheses=True,
                target_lang=target_lang,
            ),
            lambda: model.transcribe(
                audio=[audio],
                batch_size=1,
                sample_rate=_SAMPLE_RATE,
                return_hypotheses=True,
            ),
            lambda: model.transcribe([audio], batch_size=1, return_hypotheses=True),
            lambda: model.transcribe([audio]),
        )
        last: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as e:
                last = e
                continue
        raise ModelError(f"NeMo ASR transcribe API not compatible: {last}")

    @staticmethod
    def _move_model(model: Any, device: str) -> None:
        to = getattr(model, "to", None)
        if callable(to):
            to(device)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @staticmethod
    def _try_half(model: Any) -> None:
        half = getattr(model, "half", None)
        if callable(half):
            try:
                half()
            except Exception as e:  # pragma: no cover - model dependent
                _log.warning("NeMo ASR fp16 conversion failed; continuing in model default dtype: %s", e)

    @staticmethod
    def _target_lang(language_hint: Optional[str]) -> str:
        if not language_hint or language_hint.strip().lower() == "auto":
            return "auto"
        return language_hint

    @staticmethod
    def _first_hypothesis(result: Any) -> Any:
        cur = result
        if isinstance(cur, tuple):
            cur = cur[0]
        if isinstance(cur, list):
            return cur[0] if cur else ""
        return cur

    def _extract_word_tokens(
        self,
        hyp: Any,
        duration: float,
        language_hint: Optional[str],
    ) -> List[AsrToken]:
        text = self._hyp_text(hyp)
        text, detected_language = self._strip_language_tag(text)
        language = detected_language or (language_hint.split("-")[0] if language_hint else None)

        words = self._timestamp_words(hyp)
        if words:
            tokens = [self._word_item_to_token(w, language=language) for w in words]
            return [t for t in tokens if t.text and t.t_end >= t.t_start]
        return self._evenly_spaced_tokens(text, duration, language=language)

    @staticmethod
    def _hyp_text(hyp: Any) -> str:
        if isinstance(hyp, str):
            return hyp
        if isinstance(hyp, dict):
            return str(hyp.get("text", ""))
        return str(getattr(hyp, "text", ""))

    @staticmethod
    def _strip_language_tag(text: str) -> tuple[str, Optional[str]]:
        m = _LANG_TAG.search(text)
        if not m:
            return text.strip(), None
        return _LANG_TAG.sub("", text).strip(), m.group(1).split("-")[0]

    @staticmethod
    def _timestamp_words(hyp: Any) -> list:
        if isinstance(hyp, dict):
            ts = hyp.get("timestamp") or hyp.get("timestamps") or hyp.get("words")
        else:
            ts = (
                getattr(hyp, "timestamp", None)
                or getattr(hyp, "timestamps", None)
                or getattr(hyp, "words", None)
            )
        if isinstance(ts, dict):
            ts = ts.get("word") or ts.get("words") or ts.get("word_ts")
        return list(ts or []) if isinstance(ts, (list, tuple)) else []

    @staticmethod
    def _word_item_to_token(item: Any, *, language: Optional[str]) -> AsrToken:
        if isinstance(item, str):
            return AsrToken(text=item, t_start=0.0, t_end=0.0, language=language)
        getter = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        text = str(getter("word", getter("text", ""))).strip()
        start = _coerce_time(
            getter("start", getter("t_start", getter("start_time", getter("start_offset", 0.0))))
        )
        end = _coerce_time(
            getter("end", getter("t_end", getter("end_time", getter("end_offset", start))))
        )
        score = float(getter("score", getter("confidence", 1.0)) or 1.0)
        return AsrToken(text=text, t_start=start, t_end=end, language=language, score=score)

    @staticmethod
    def _evenly_spaced_tokens(text: str, duration: float, *, language: Optional[str]) -> List[AsrToken]:
        words = [w for w in text.split() if w]
        if not words:
            return []
        step = max(duration, 1e-3) / len(words)
        return [
            AsrToken(
                text=w,
                t_start=i * step,
                t_end=(i + 1) * step,
                language=language,
                score=1.0,
            )
            for i, w in enumerate(words)
        ]


def _coerce_time(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["NemotronNeMoTranscriber"]
