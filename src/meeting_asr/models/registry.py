"""Model registry + download/cache lifecycle (tasks T010, T039; Decision 6).

Declares each required :class:`ModelAsset` (HF repo id, pinned revision,
expected files) and implements :func:`prepare` — resumable download via
``huggingface_hub`` with integrity verification and **no corrupt cache** on
interruption (research Decision 6, edge case: interrupted download).

The downloader is an injectable seam (:func:`prepare`'s ``downloader`` kwarg) so
the whole lifecycle is unit-testable offline — the real download only happens
inside the default downloader, which lazy-imports ``huggingface_hub``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .._logging import ModelError, get_logger
from ..types import ModelAsset, ModelKind, ModelState, PrepareProgress

_log = get_logger("models.registry")

MODELS_ENV_VAR = "MEETING_ASR_MODELS_DIR"

# Nemotron 3.5 ASR Streaming supported language-locales (~40). Reference list;
# the loaded model's own ``supported_languages()`` is authoritative at runtime.
NEMOTRON_LANGUAGES = (
    "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "nl", "tr",
    "pl", "uk", "ar", "hi", "id", "vi", "th", "sv", "no", "da", "fi", "cs",
    "el", "he", "ro", "hu", "sk", "bg", "hr", "sr", "sl", "lt", "lv", "et",
    "ms", "fil", "ca", "mr",
)


def default_cache_dir() -> Path:
    """Resolve the model cache root (env override > XDG-style default)."""
    env = os.environ.get(MODELS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "meeting_asr" / "models"


def cache_dir_for(asset: ModelAsset, cache_root: Path) -> Path:
    """Per-asset cache directory under the cache root."""
    return Path(cache_root) / asset.name


def model_registry() -> List[ModelAsset]:
    """The canonical list of required models for the pipeline.

    Repos/revisions are configured here; adjust to the exact published artifacts
    once the FP16 ONNX / CoreML builds are pinned (research Decisions 1–2).
    """
    # FP16 ONNX export of nvidia/nemotron-3.5-asr-streaming-0.6b (.nemo ships no
    # ONNX; NeMo/torch target CUDA, so on Apple Silicon we run the ONNX/CoreML EP
    # path against this export). Cache-aware streaming RNNT is a 3-graph export:
    # encoder/decoder/joint (+ external-weight ``.data`` sidecars). ``expected_files``
    # lists every file the cache must hold; ``[0]`` (encoder.onnx) is the entry graph.
    asr = ModelAsset(
        name="nemotron-3.5-asr-streaming",
        kind=ModelKind.ASR,
        repo_id="soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16",
        revision="76daabfd0aaf5ec6ef1e6640eae3b364af6c9970",
        expected_files=(
            "encoder.onnx",
            "encoder.onnx.data",
            "decoder.onnx",
            "decoder.onnx.data",
            "joint.onnx",
            "joint.onnx.data",
            "config.json",
            "vocab.json",
            "languages.json",
        ),
        supported_languages=NEMOTRON_LANGUAGES,
    )
    # Streaming Sortformer 4spk-v2.1 CoreML. ``Sortformer.mlpackage`` is a package
    # *directory* (coremltools loads it by path), tracked here as a single cache
    # entry — ``[0]`` resolves to the package dir for the loader, and the downloader
    # expands ``*.mlpackage`` entries to fetch their contents recursively.
    diarizer = ModelAsset(
        name="diar-streaming-sortformer-4spk-v2.1",
        kind=ModelKind.DIARIZER,
        repo_id="FluidInference/diar-streaming-sortformer-coreml",
        revision="89f9a0d635c2f01b0202651432eb55d55939a07b",
        expected_files=(
            "Sortformer.mlpackage",
            "config.json",
        ),
    )
    return [asr, diarizer]


def check_cached(asset: ModelAsset, cache_root: Path) -> bool:
    """True iff every expected file exists and is non-empty under the cache dir."""
    base = cache_dir_for(asset, Path(cache_root))
    for rel in asset.expected_files:
        p = base / rel
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def refresh_state(asset: ModelAsset, cache_root: Path) -> ModelAsset:
    """Return a copy of ``asset`` with state set from the local cache (no network).

    A previously-ERROR asset keeps ERROR; otherwise CACHED if present else ABSENT.
    """
    if asset.state == ModelState.ERROR:
        return asset
    cached = check_cached(asset, Path(cache_root))
    new_state = ModelState.CACHED if cached else ModelState.ABSENT
    path = str(cache_dir_for(asset, Path(cache_root))) if cached else asset.cache_path
    return replace(asset, state=new_state, cache_path=path)


# Downloader seam: (asset, target_dir, report) -> target_dir after writing files.
Downloader = Callable[[ModelAsset, Path, Callable[[int, int], None]], Path]


def _download_patterns(expected_files: Sequence[str]) -> List[str]:
    """Build ``allow_patterns`` for ``snapshot_download``.

    Each expected file is fetched verbatim; package-*directory* entries
    (``*.mlpackage`` / ``*.mlmodelc``, tracked as one cache entry) are also
    expanded to ``<dir>/*`` so their contents come down recursively.
    """
    patterns: List[str] = []
    for f in expected_files:
        patterns.append(f)
        if f.endswith((".mlpackage", ".mlmodelc")):
            patterns.append(f + "/*")
    return patterns


def _default_downloader(asset: ModelAsset, target_dir: Path, report: Callable[[int, int], None]) -> Path:
    """Real download via ``huggingface_hub.snapshot_download`` (resumable)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        local_dir=str(target_dir),
        allow_patterns=_download_patterns(asset.expected_files),
        resume_download=True,
    )
    # Best-effort progress: count present files (hf_hub has richer callbacks we can wire later).
    present = sum(1 for f in asset.expected_files if (target_dir / f).exists())
    report(present, len(asset.expected_files))
    return target_dir


def prepare(
    assets: Sequence[ModelAsset],
    *,
    progress: Optional[Callable[[PrepareProgress], None]] = None,
    force: bool = False,
    cache_root: Optional[Path] = None,
    downloader: Optional[Downloader] = None,
) -> List[ModelAsset]:
    """Download/cache each asset; resumable + integrity-checked, no corrupt cache.

    Idempotent: a cached asset is skipped unless ``force``. Emits
    :class:`PrepareProgress` per asset (DOWNLOADING → CACHED). Raises
    :class:`ModelError` on interruption/missing-files — the cache is never left
    in a CACHED state for a partial download.

    ``downloader`` defaults to :func:`_default_downloader` resolved at call time,
    so tests can monkeypatch the module attribute to swap in an offline fake.
    """
    root = Path(cache_root or default_cache_dir())
    downloader = downloader or _default_downloader
    result: List[ModelAsset] = []

    def _emit(asset: str, downloaded: int, total: int, state: ModelState) -> None:
        if progress is not None:
            progress(PrepareProgress(asset=asset, downloaded=downloaded, total=total, state=state))

    for asset in assets:
        current = refresh_state(asset, root)
        if current.state == ModelState.CACHED and not force:
            _emit(current.name, 0, 0, ModelState.CACHED)
            result.append(current)
            continue

        target = cache_dir_for(current, root)
        target.mkdir(parents=True, exist_ok=True)
        _emit(current.name, 0, 1, ModelState.DOWNLOADING)
        try:
            downloader(current, target, lambda d, t: _emit(current.name, d, t or 1, ModelState.DOWNLOADING))
        except Exception as e:  # interrupted / network loss
            _log.error("download interrupted for %s: %s", current.name, e)
            result.append(replace(current, state=ModelState.ERROR))
            raise ModelError(f"download failed for '{current.name}': {e}") from e

        # Integrity gate: never mark CACHED unless every expected file is present & non-empty.
        probe = replace(current, cache_path=str(target))
        if not check_cached(probe, root):
            result.append(replace(current, state=ModelState.ERROR))
            raise ModelError(
                f"integrity check failed for '{current.name}': "
                f"missing one of {tuple(current.expected_files)}"
            )

        ready = replace(current, state=ModelState.CACHED, cache_path=str(target))
        _emit(ready.name, len(ready.expected_files), len(ready.expected_files), ModelState.CACHED)
        result.append(ready)

    return result


__all__ = [
    "MODELS_ENV_VAR",
    "NEMOTRON_LANGUAGES",
    "default_cache_dir",
    "cache_dir_for",
    "model_registry",
    "check_cached",
    "refresh_state",
    "prepare",
]
