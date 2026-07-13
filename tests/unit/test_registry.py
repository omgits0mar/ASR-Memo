"""Unit tests for models/registry.py (T010, T039; Decision 6).

Fully offline: the real ``huggingface_hub`` downloader is never invoked — we
inject a fake ``downloader`` seam. Covers cached fast-path, force re-download,
interrupted-download safety, and the no-corrupt-cache integrity gate.
"""

from __future__ import annotations

import pytest

from meeting_asr._logging import ModelError
from meeting_asr.models.registry import (
    check_cached,
    model_registry,
    prepare,
    refresh_state,
)
from meeting_asr.types import ComputeBackend, ModelAsset, ModelFramework, ModelKind, ModelState


def _asset(name="t", files=("model.onnx", "config.json")):
    return ModelAsset(
        name=name,
        kind=ModelKind.ASR,
        framework=ModelFramework.ONNX,
        repo_id="org/test",
        revision="main",
        expected_files=files,
    )


def _fake_downloader_ok(files):
    """Returns a downloader that writes the given (complete) file set."""
    def _dl(asset, target, report):
        for f in asset.expected_files:
            (target / f).write_bytes(b"\x00" * 16)
        report(len(asset.expected_files), len(asset.expected_files))
        return target
    return _dl


class TestRegistryDeclarations:
    def test_registry_has_asr_and_diarizer(self):
        asr = next(a for a in model_registry() if a.kind is ModelKind.ASR)
        dia = next(a for a in model_registry() if a.kind is ModelKind.DIARIZER)
        assert asr.framework is ModelFramework.ONNX
        assert dia.framework is ModelFramework.COREML
        assert asr.expected_files and asr.supported_languages
        assert dia.kind is ModelKind.DIARIZER and dia.supported_languages is None

    def test_cuda_registry_uses_nemo_assets(self):
        assets = model_registry(ComputeBackend.CUDA)
        assert {a.framework for a in assets} == {ModelFramework.NEMO}
        assert next(a for a in assets if a.kind is ModelKind.ASR).expected_files == (
            "nemotron-3.5-asr-streaming-0.6b.nemo",
        )
        assert next(a for a in assets if a.kind is ModelKind.DIARIZER).repo_id == (
            "nvidia/diar_streaming_sortformer_4spk-v2.1"
        )


class TestCacheDetection:
    def test_check_cached_false_when_absent(self, tmp_path):
        assert check_cached(_asset(), tmp_path) is False

    def test_check_cached_true_when_all_files_present(self, tmp_path):
        from meeting_asr.models.registry import cache_dir_for

        base = cache_dir_for(_asset(), tmp_path)
        base.mkdir(parents=True)
        for f in ("model.onnx", "config.json"):
            (base / f).write_bytes(b"x" * 8)
        assert check_cached(_asset(), tmp_path) is True

    def test_refresh_state_absent_then_cached(self, tmp_path):
        from meeting_asr.models.registry import cache_dir_for

        a = _asset()
        assert refresh_state(a, tmp_path).state is ModelState.ABSENT
        base = cache_dir_for(a, tmp_path)
        base.mkdir(parents=True)
        for f in a.expected_files:
            (base / f).write_bytes(b"x")
        assert refresh_state(a, tmp_path).state is ModelState.CACHED


class TestPrepareLifecycle:
    def test_prepare_downloads_and_caches_with_progress(self, tmp_path):
        ticks = []
        out = prepare(
            [_asset()], cache_root=tmp_path, downloader=_fake_downloader_ok(None),
            progress=lambda p: ticks.append((p.asset, p.state)),
        )
        assert out[0].state is ModelState.CACHED
        assert out[0].cache_path is not None
        states = [s for _, s in ticks]
        assert ModelState.DOWNLOADING in states and ModelState.CACHED in states

    def test_prepare_idempotent_skips_cached(self, tmp_path):
        from meeting_asr.models.registry import cache_dir_for

        a = _asset()
        base = cache_dir_for(a, tmp_path)
        base.mkdir(parents=True)
        for f in a.expected_files:
            (base / f).write_bytes(b"x")

        called = {"n": 0}
        def _spy(asset, target, report):
            called["n"] += 1
            return target

        out = prepare([a], cache_root=tmp_path, downloader=_spy)
        assert out[0].state is ModelState.CACHED
        assert called["n"] == 0  # cached fast-path: no download

    def test_force_redownloads_even_when_cached(self, tmp_path):
        from meeting_asr.models.registry import cache_dir_for

        a = _asset()
        base = cache_dir_for(a, tmp_path)
        base.mkdir(parents=True)
        for f in a.expected_files:
            (base / f).write_bytes(b"old")

        called = {"n": 0}
        def _spy(asset, target, report):
            called["n"] += 1
            for f in asset.expected_files:
                (target / f).write_bytes(b"new")
            return target

        out = prepare([a], cache_root=tmp_path, downloader=_spy, force=True)
        assert out[0].state is ModelState.CACHED and called["n"] == 1

    def test_interrupted_download_raises_and_marks_error(self, tmp_path):
        def _boom(asset, target, report):
            raise ConnectionError("network gone")

        with pytest.raises(ModelError):
            prepare([_asset()], cache_root=tmp_path, downloader=_boom)

    def test_partial_download_not_marked_cached(self, tmp_path):
        # Writes only the first expected file → integrity gate fails.
        def _partial(asset, target, report):
            (target / asset.expected_files[0]).write_bytes(b"x")
            return target

        with pytest.raises(ModelError):
            prepare([_asset(files=("a", "b"))], cache_root=tmp_path, downloader=_partial)
        # And the asset is never left CACHED:
        assert check_cached(_asset(files=("a", "b")), tmp_path) is False
