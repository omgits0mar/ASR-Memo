"""NemotronOnnxTranscriber — Nemotron 3.5 ASR Streaming via ONNX Runtime (T021).

Backend for the :class:`SpeechTranscriber` protocol. Drives the FP16 ONNX export of
NVIDIA Nemotron 3.5 ASR Streaming 0.6B through ONNX Runtime with the CoreML
Execution Provider (``MLComputeUnits=.cpuAndGPU`` → M-series GPU + CPU; CPU EP
fallback).

The export is a **cache-aware streaming FastConformer-RNNT** in three graphs:

* ``encoder.onnx`` — 32 log-mel frames (128 bins) + carried attention/conv/pre
  caches + a 128-slot ``language_mask`` prompt → 4 encoded frames + new caches.
* ``decoder.onnx`` — 2-layer LSTM prediction network (token + h/c → 640-d).
* ``joint.onnx`` — encoder ⊕ decoder → ``logits[13088]`` (vocab 13087 + blank 13087).

Each 320 ms chunk yields 4 encoded frames (8× subsampling, 80 ms/frame); RNN-T
greedy decoding runs the joint→argmax→prediction loop per frame, carrying the
LSTM state across frames and chunks so the stream is decoded exactly once
(FR-005, FR-016). Language is selected by a one-hot prompt slot from
``languages.json`` (FR-007/008); the front end is :mod:`._features`.

``onnxruntime`` is lazy-imported; the model opens only when cached
(``prepare_models()``). Real inference runs under ``needs_models`` on Apple
Silicon; the streaming contract is validated offline with fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .._logging import ModelError, get_logger
from ..backends.device import ort_providers
from ..models.registry import NEMOTRON_LANGUAGES, default_cache_dir, model_registry, refresh_state
from ..types import AsrToken, AudioFrame, ComputeBackend, ModelKind
from ._features import LogMelFrontEnd

_log = get_logger("asr.nemotron")

_SAMPLE_RATE = 16000
CHUNK_MS = 320  # export streaming chunk (config.json streaming.chunkMs)
_MAX_SYMBOLS = 10  # RNN-T greedy guard: max non-blank emissions per encoder frame


class NemotronOnnxTranscriber:
    """SpeechTranscriber-conforming backend (Nemotron ONNX, FP16 default)."""

    def __init__(self) -> None:
        self._enc = None
        self._dec = None
        self._joint = None
        self._backend: Optional[ComputeBackend] = None
        self._precision = "fp16"
        self._cache_path: Optional[Path] = None
        self._frontend: Optional[LogMelFrontEnd] = None

        # config-derived dims (resolved at load)
        self._n_mels = 128
        self._enc_layers = 24
        self._enc_hidden = 1024
        self._left_context = 56
        self._conv_cache = 8
        self._pre_cache_size = 9
        self._dec_layers = 2
        self._dec_hidden = 640
        self._subsampling = 8
        self._mel_frames = 32
        self._out_frames = 4
        self._blank_id = 13087
        self._num_prompts = 128

        # vocab / language prompt
        self._vocab: Dict[int, str] = {}
        self._prompt_dict: Dict[str, int] = {}
        self._auto_slot: Optional[int] = None  # multilingual auto-detect prompt (autoSlot)
        self._lang_slot = 0
        self._language: Optional[str] = "en"

        # streaming state
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._frame_buf: Optional[np.ndarray] = None  # carry for continuous mel framing
        self._frame_idx = 0  # global encoded-frame counter (80 ms each) → timestamps
        # encoder caches
        self._pre_cache: Optional[np.ndarray] = None
        self._cache_lc: Optional[np.ndarray] = None
        self._cache_lt: Optional[np.ndarray] = None
        self._cache_lc_len: Optional[np.ndarray] = None
        # RNN-T decoder (prediction net) state
        self._g: Optional[np.ndarray] = None  # decoder_output for the last emitted label
        self._h: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None
        # word-merge state: the model emits BPE subword pieces; the pipeline contract
        # (fake ManifestTranscriber, aligner's " ".join) is word-level tokens. We hold
        # the trailing partial word across push() calls and emit one AsrToken per word.
        self._held_pieces: List[AsrToken] = []

    # ---- SpeechTranscriber protocol ----

    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        if precision not in ("fp16", "int8", "int4"):
            raise ModelError(f"unsupported precision '{precision}' (use fp16|int8|int4)")
        asset = next(a for a in model_registry() if a.kind is ModelKind.ASR)
        asset = refresh_state(asset, default_cache_dir())
        if not asset.is_cached():
            raise ModelError(f"ASR model '{asset.name}' not cached; run prepare_models() first")
        self._cache_path = Path(asset.cache_path)
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ModelError("onnxruntime not installed (pip install onnxruntime)") from e

        providers = ort_providers(backend)
        so = ort.SessionOptions()
        self._enc = self._open_session(ort, so, "encoder.onnx", providers)
        self._dec = self._open_session(ort, so, "decoder.onnx", providers)
        self._joint = self._open_session(ort, so, "joint.onnx", providers)

        self._load_config()
        self._vocab = self._load_vocab()
        self._prompt_dict = self._load_prompts()
        self._frontend = LogMelFrontEnd(
            n_mels=self._n_mels, sample_rate=_SAMPLE_RATE,
        )
        self._backend = backend
        self._precision = precision
        self.reset()
        _log.info(
            "Nemotron ONNX loaded (precision=%s, chunk=%dms, vocab=%d, providers=%s)",
            precision, CHUNK_MS, len(self._vocab), providers,
        )

    def _open_session(self, ort, so, filename: str, providers: list):
        """Open one ONNX session, falling back to CPU EP if the accelerated EP fails.

        The encoder ships **external-data** weights (``*.onnx.data``), which the
        CoreML EP cannot initialize (it partitions subgraphs without the on-disk
        model path → "model_path must not be empty"). CPU EP loads them fine and,
        per the export card, still runs FP16 at real-time (RTF ~0.27). We therefore
        retry on CPU rather than failing the load.
        """
        path = str((self._cache_path or Path()) / filename)
        try:
            return ort.InferenceSession(path, so, providers=providers)
        except Exception as e:  # pragma: no cover - provider/model dependent
            if providers == ["CPUExecutionProvider"]:
                raise ModelError(f"failed to open ONNX session {filename}: {e}") from e
            _log.warning("%s: accelerated EP %s failed (%s); falling back to CPU EP",
                         filename, providers, e)
            try:
                return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            except Exception as e2:  # pragma: no cover - model dependent
                raise ModelError(f"failed to open ONNX session {filename}: {e2}") from e2

    def reset(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._frame_buf = None
        self._frame_idx = 0
        self._held_pieces = []
        self._reset_caches()
        if self._dec is not None:
            self._reset_decoder()

    def push(self, frame: AudioFrame, *, language_hint: Optional[str] = None) -> List[AsrToken]:
        if self._enc is None:
            raise ModelError("transcriber not loaded; call load() first")
        self._set_language(language_hint)
        pcm = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        self._audio_buf = np.concatenate([self._audio_buf, pcm])
        hop_samples = self._mel_frames * self._frontend.hop_length  # 32 * 160 = 5120 (320 ms)
        out: List[AsrToken] = []
        while len(self._audio_buf) >= hop_samples:
            chunk = self._audio_buf[:hop_samples]
            self._audio_buf = self._audio_buf[hop_samples:]
            feats = self._stream_features(chunk)  # exactly 32 frames
            out.extend(self._decode_features(feats))
        return self._merge_to_words(out)

    def flush(self) -> List[AsrToken]:
        """Decode any remaining buffered audio at end-of-stream."""
        if self._enc is None or len(self._audio_buf) == 0:
            return self._finalize_held()
        tail = self._audio_buf
        self._audio_buf = np.zeros(0, dtype=np.float32)
        feats = self._stream_features(tail)
        words = self._merge_to_words(self._decode_features(feats))
        words.extend(self._finalize_held())
        return words

    def supported_languages(self) -> List[str]:
        return list(NEMOTRON_LANGUAGES)

    # ---- whole-utterance path (file transcription + validation) ----

    def transcribe_array(self, audio: np.ndarray, *, language: Optional[str] = None) -> List[AsrToken]:
        """Decode a complete 16 kHz mono signal in one pass.

        Computes log-mel over the whole array (raw — the ONNX encoder applies its
        own normalization internally, so no external per-feature norm), then streams
        32-frame windows through the encoder/RNN-T with carried caches. This is the
        path the accuracy harness (``make validate``) exercises.
        """
        if self._enc is None or self._frontend is None:
            raise ModelError("transcriber not loaded; call load() first")
        self.reset()
        self._set_language(language)
        feats = self._frontend(np.asarray(audio, dtype=np.float32), normalize="none")
        words = self._merge_to_words(self._decode_features(feats))
        words.extend(self._finalize_held())
        return words

    # ---- BPE subword → word-level tokens (pipeline contract) ----

    def _merge_to_words(self, pieces: List[AsrToken]) -> List[AsrToken]:
        """Group BPE subword pieces into word-level ``AsrToken``s.

        The vocab's word-start marker (▁) is detokenized to a leading space, so a
        piece whose text begins with whitespace opens a new word; non-space-leading
        pieces continue the current word (so punctuation attaches, e.g. "incurred.").
        The trailing partial word is held until the next word-start (or
        :meth:`flush`/:meth:`_finalize_held`) so streaming stays correct across
        ``push()`` calls. Each emitted token's text is space-joinable (no leading
        space), matching the fake ``ManifestTranscriber``'s word granularity.
        """
        pieces = self._held_pieces + list(pieces)
        self._held_pieces = []
        out: List[AsrToken] = []
        current: List[AsrToken] = []
        for p in pieces:
            if p.text and p.text[:1].isspace() and current:
                out.append(self._build_word(current))
                current = []
            if p.text:
                current.append(p)
        self._held_pieces = current  # hold the last open word for the next call
        return out

    @staticmethod
    def _build_word(pieces: List[AsrToken]) -> AsrToken:
        return AsrToken(
            text="".join(p.text for p in pieces).strip(),
            t_start=pieces[0].t_start,
            t_end=pieces[-1].t_end,
            language=pieces[-1].language,
            score=sum(p.score for p in pieces) / len(pieces) if pieces else 0.0,
        )

    def _finalize_held(self) -> List[AsrToken]:
        out: List[AsrToken] = []
        if self._held_pieces:
            out.append(self._build_word(self._held_pieces))
            self._held_pieces = []
        return out

    # ---- features ----

    def _stream_features(self, chunk: np.ndarray) -> np.ndarray:
        """Per-chunk log-mel with carried framing context (continuous, exact frames).

        Prepends ``n_fft - hop`` samples of the previous chunk so ``center=False``
        framing yields exactly ``len(chunk) / hop`` frames with no inter-chunk gap.
        """
        fe = self._frontend
        carry = fe.n_fft - fe.hop_length
        if self._frame_buf is None:
            left = np.zeros(carry, dtype=np.float32)
        else:
            left = self._frame_buf
        x = np.concatenate([left, np.asarray(chunk, dtype=np.float32)])
        self._frame_buf = x[-carry:]
        return fe.frames(x, normalize="none")

    # ---- cache-aware encode + RNN-T greedy decode ----

    def _decode_features(self, feats: np.ndarray) -> List[AsrToken]:
        """Run the encoder over 32-frame windows and RNN-T greedy over each frame."""
        tokens: List[AsrToken] = []
        n_mels, total = feats.shape
        if total == 0:
            return tokens
        win = self._mel_frames
        lang_mask = self._language_mask()
        i = 0
        while i < total:
            block = feats[:, i : i + win]
            n_valid = block.shape[1]
            if n_valid < win:  # pad the final window
                block = np.pad(block, ((0, 0), (0, win - n_valid)))
            enc_out, enc_len = self._run_encoder(block, n_valid, lang_mask)
            frames = int(enc_len) if enc_len is not None else enc_out.shape[1]
            frames = min(frames, enc_out.shape[1])
            for f in range(frames):
                enc_f = enc_out[:, f : f + 1, :].astype(np.float32)  # [1,1,1024]
                tokens.extend(self._rnnt_step(enc_f))
                self._frame_idx += 1
            i += win
        return tokens

    def _run_encoder(self, block: np.ndarray, n_valid: int, lang_mask: np.ndarray):
        feed = {
            "audio_signal": block[None, :, :].astype(np.float32),       # [1,128,32]
            "audio_length": np.array([n_valid], dtype=np.int32),
            "language_mask": lang_mask,                                  # [1,128]
            "pre_cache": self._pre_cache,
            "cache_last_channel": self._cache_lc,
            "cache_last_time": self._cache_lt,
            "cache_last_channel_len": self._cache_lc_len,
        }
        (enc_out, enc_len, new_pre, new_lc, new_lt, new_lc_len) = self._enc.run(None, feed)
        self._pre_cache = np.asarray(new_pre, dtype=np.float32)
        self._cache_lc = np.asarray(new_lc, dtype=np.float32)
        self._cache_lt = np.asarray(new_lt, dtype=np.float32)
        self._cache_lc_len = np.asarray(new_lc_len, dtype=np.int32)
        return np.asarray(enc_out, dtype=np.float32), np.asarray(enc_len).reshape(-1)[0]

    def _rnnt_step(self, enc_f: np.ndarray) -> List[AsrToken]:
        """Greedy RNN-T inner loop for one encoder frame; returns emitted tokens."""
        out: List[AsrToken] = []
        t0 = self._frame_idx * self._frame_seconds()
        t1 = t0 + self._frame_seconds()
        for _ in range(_MAX_SYMBOLS):
            logits = self._joint.run(None, {"encoder_output": enc_f, "decoder_output": self._g})[0]
            vec = np.asarray(logits, dtype=np.float32).reshape(-1)
            k = int(vec.argmax())
            if k == self._blank_id:
                break
            text = self._detokenize(k)
            if text:
                out.append(
                    AsrToken(text=text, t_start=t0, t_end=t1, language=self._language,
                             score=self._softmax_score(vec, k))
                )
            self._advance_decoder(k)
        return out

    # ---- decoder (prediction network) ----

    def _reset_decoder(self) -> None:
        self._h = np.zeros((self._dec_layers, 1, self._dec_hidden), dtype=np.float32)
        self._c = np.zeros((self._dec_layers, 1, self._dec_hidden), dtype=np.float32)
        # Prime the prediction net with SOS = blank to get the initial decoder_output.
        self._g, self._h, self._c = self._run_decoder(self._blank_id, self._h, self._c)

    def _advance_decoder(self, token: int) -> None:
        self._g, self._h, self._c = self._run_decoder(token, self._h, self._c)

    def _run_decoder(self, token: int, h: np.ndarray, c: np.ndarray):
        feed = {"token": np.array([[token]], dtype=np.int64), "h": h, "c": c}
        g, h_out, c_out = self._dec.run(None, feed)
        return (np.asarray(g, dtype=np.float32),
                np.asarray(h_out, dtype=np.float32),
                np.asarray(c_out, dtype=np.float32))

    # ---- caches / config / vocab / language ----

    def _reset_caches(self) -> None:
        self._pre_cache = np.zeros((1, self._n_mels, self._pre_cache_size), dtype=np.float32)
        self._cache_lc = np.zeros((self._enc_layers, 1, self._left_context, self._enc_hidden), dtype=np.float32)
        self._cache_lt = np.zeros((self._enc_layers, 1, self._enc_hidden, self._conv_cache), dtype=np.float32)
        self._cache_lc_len = np.zeros((1,), dtype=np.int32)

    def _frame_seconds(self) -> float:
        return self._subsampling * self._frontend.hop_length / _SAMPLE_RATE  # 8*160/16000 = 0.08

    def _load_config(self) -> None:
        cfg_path = (self._cache_path or Path()) / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # pragma: no cover - packaging guard
            _log.warning("could not read config.json (%s); using defaults", e)
            return
        self._n_mels = int(cfg.get("numMelBins", self._n_mels))
        self._enc_layers = int(cfg.get("encoderLayers", self._enc_layers))
        self._enc_hidden = int(cfg.get("encoderHidden", self._enc_hidden))
        self._left_context = int(cfg.get("attentionLeftContext", self._left_context))
        self._conv_cache = int(cfg.get("convCacheSize", self._conv_cache))
        self._pre_cache_size = int(cfg.get("streaming", {}).get("preCacheSize", self._pre_cache_size))
        self._dec_layers = int(cfg.get("decoderLayers", self._dec_layers))
        self._dec_hidden = int(cfg.get("decoderHidden", self._dec_hidden))
        self._subsampling = int(cfg.get("subsamplingFactor", self._subsampling))
        self._mel_frames = int(cfg.get("streaming", {}).get("melFrames", self._mel_frames))
        self._out_frames = int(cfg.get("streaming", {}).get("outputFrames", self._out_frames))
        self._blank_id = int(cfg.get("blankTokenId", self._blank_id))
        self._num_prompts = int(cfg.get("numPrompts", self._num_prompts))

    def _load_vocab(self) -> Dict[int, str]:
        path = (self._cache_path or Path()) / "vocab.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # pragma: no cover - packaging guard
            _log.warning("could not read vocab.json: %s", e)
            return {}
        return {int(k): v for k, v in data.items()}

    def _load_prompts(self) -> Dict[str, int]:
        path = (self._cache_path or Path()) / "languages.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # pragma: no cover - packaging guard
            _log.warning("could not read languages.json: %s", e)
            return {}
        # autoSlot = the multilingual auto-detect prompt (used when no hint given).
        auto = data.get("autoSlot")
        self._auto_slot = int(auto) if auto is not None else None
        return dict(data.get("promptDictionary", {}))

    def _set_language(self, language_hint: Optional[str]) -> None:
        """Resolve the prompt slot from a language hint (locale or bare code).

        No hint (None/empty) or an explicit ``auto`` selects the multilingual
        auto-detect slot when the model exposes one — so a session that mixes
        English and Arabic is recognized without setting a code. ``_language`` is
        then left unknown (None) since detection is per-utterance by the model.
        """
        if not language_hint or language_hint.strip().lower() == "auto":
            if self._auto_slot is not None:
                self._lang_slot = self._auto_slot
                self._language = None  # auto-detect: don't claim a fixed language
            return
        slot = self._resolve_slot(language_hint)
        if slot is not None:
            self._lang_slot = slot
            self._language = language_hint.split("-")[0]

    def _resolve_slot(self, lang: str) -> Optional[int]:
        if lang in self._prompt_dict:
            return self._prompt_dict[lang]
        base = lang.split("-")[0]
        if base in self._prompt_dict:
            return self._prompt_dict[base]
        for key, slot in self._prompt_dict.items():  # any locale of this language
            if key.split("-")[0] == base:
                return slot
        return None

    def _language_mask(self) -> np.ndarray:
        mask = np.zeros((1, self._num_prompts), dtype=np.float32)
        if 0 <= self._lang_slot < self._num_prompts:
            mask[0, self._lang_slot] = 1.0
        return mask

    # ---- token → text ----

    def _detokenize(self, tid: int) -> str:
        piece = self._vocab.get(tid, "")
        if not piece or piece == "<unk>":
            return ""
        if piece.startswith("<") and piece.endswith(">"):  # <lang>/special markers
            return ""
        return piece.replace("▁", " ").replace("▂", " ")  # ▁ → space

    @staticmethod
    def _softmax_score(logits: np.ndarray, k: int) -> float:
        m = float(logits.max())
        ex = np.exp(logits - m)
        denom = float(ex.sum())
        return float(ex[k] / denom) if denom > 0 else 0.0


__all__ = ["NemotronOnnxTranscriber"]
