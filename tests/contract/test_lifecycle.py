"""Contract test: model lifecycle & readiness (task T037; FR-011, FR-012, FR-013).

Validates ``prepare_models``/``check_readiness`` against the registry seam with an
injected fake downloader (offline): progress callbacks, the ``missing[]`` list, the
cached fast-path (no re-download), and interrupted-download safety (no corrupt cache).
"""

from __future__ import annotations

import pytest

from meeting_asr._logging import ModelError
from meeting_asr.models.registry import prepare
from meeting_asr.types import ModelState


def _ok_downloader():
    def _dl(asset, target, report):
        for f in asset.expected_files:
            (target / f).write_bytes(b"\x00" * 16)
        report(len(asset.expected_files), len(asset.expected_files))
        return target
    return _dl


def test_prepare_emits_progress_and_caches(synthetic_fixture_unused, tmp_path):
    from meeting_asr.models.registry import model_registry

    ticks = []
    out = prepare(
        model_registry(), cache_root=tmp_path, downloader=_ok_downloader(),
        progress=lambda p: ticks.append((p.asset, p.state)),
    )
    assert all(a.state is ModelState.CACHED for a in out)
    assert ModelState.DOWNLOADING in [s for _, s in ticks]


def test_prepare_cached_fast_path_no_redownload(tmp_path):
    from meeting_asr.models.registry import model_registry

    calls = {"n": 0}

    def _spy(asset, target, report):
        calls["n"] += 1
        for f in asset.expected_files:
            (target / f).write_bytes(b"x")
        return target

    prepare(model_registry(), cache_root=tmp_path, downloader=_spy)
    first = calls["n"]
    prepare(model_registry(), cache_root=tmp_path, downloader=_spy)  # cached → no download
    assert calls["n"] == first


def test_prepare_interrupted_is_resumable_safe(tmp_path):
    from meeting_asr.models.registry import model_registry

    state = {"attempts": 0}

    def _flaky(asset, target, report):
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise ConnectionError("network dropped mid-download")
        for f in asset.expected_files:
            (target / f).write_bytes(b"x")
        return target

    with pytest.raises(ModelError):
        prepare(model_registry(), cache_root=tmp_path, downloader=_flaky)
    # second attempt succeeds (resumable); no corrupt cache left from the first
    out = prepare(model_registry(), cache_root=tmp_path, downloader=_flaky)
    assert all(a.state is ModelState.CACHED for a in out)


def test_check_readiness_enumerates_missing(tmp_path, monkeypatch):
    from meeting_asr.models import registry as reg
    from meeting_asr.models.readiness import build_readiness

    monkeypatch.setattr(reg, "default_cache_dir", lambda: tmp_path)
    report = build_readiness(cache_root=tmp_path)
    assert report.ready is False  # models not downloaded
    assert any("not downloaded" in m for m in report.missing)
    assert report.compute_backend  # resolved backend string present


@pytest.fixture
def synthetic_fixture_unused():
    return None
