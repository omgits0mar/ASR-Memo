"""Shared pytest harness (task T013).

Two responsibilities mandated by the constitution's offline-CI gate:

1. **Network-isolation guard** — an autouse fixture that blocks any outbound
   socket connection (except loopback) unless the test opts in via the
   ``allow_network`` marker or ``MEETING_ASR_ALLOW_NETWORK=1``. Any test that
   accidentally reaches the network fails loudly instead of silently egressing.

2. **Fixture loader + deterministic synthetic-meeting generator** — stands in for
   live capture/models so contract + integration tests run fully offline (the
   real Nemotron/Sortformer accuracy is gated separately under ``needs_models``).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# --------------------------------------------------------------------------- #
# pytest hooks / markers
# --------------------------------------------------------------------------- #


def pytest_configure(config):
    for marker in (
        "allow_network: opt OUT of the offline network guard (rare)",
        "needs_models: requires downloaded Nemotron/Sortformer models",
        "needs_hardware: requires live mic/system-audio/CoreML hardware",
        "slow: long-running soak / accuracy gates",
    ):
        config.addinivalue_line("markers", marker)


# --------------------------------------------------------------------------- #
# Network-isolation guard (offline-CI constitution gate)
# --------------------------------------------------------------------------- #


def _install_socket_block(monkeypatch) -> None:
    """Patch socket connect to raise on any non-loopback destination."""
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _blocked_connect(self, address):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if host not in _LOOPBACK:
            raise BlockingIOError(
                f"NETWORK BLOCKED by offline-CI guard → {address!r}. "
                "Mark the test `allow_network` only if egress is intentional."
            )
        return orig_connect(self, address)

    def _blocked_connect_ex(self, address):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if host not in _LOOPBACK:
            raise BlockingIOError(
                f"NETWORK BLOCKED by offline-CI guard → {address!r}."
            )
        return orig_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect_ex)


@pytest.fixture(autouse=True)
def _offline_network_guard(request, monkeypatch):
    """Block outbound network unless explicitly allowed (constitution offline-CI)."""
    allow = (
        request.node.get_closest_marker("allow_network") is not None
        or os.environ.get("MEETING_ASR_ALLOW_NETWORK") == "1"
    )
    if allow:
        yield
        return
    _install_socket_block(monkeypatch)
    yield


@pytest.fixture(autouse=True)
def _isolate_model_cache(request, tmp_path_factory, monkeypatch):
    """Point the model cache at an empty dir for offline tests.

    The offline suite is written assuming no models are downloaded (readiness/
    "not ready" paths must behave deterministically). A developer machine that has
    run ``prepare_models()`` populates the real cache (``~/.cache/meeting_asr``),
    which would otherwise flip those paths to "ready" and leak a live session.
    ``needs_models`` tests opt out so they use the real cached models.
    """
    if request.node.get_closest_marker("needs_models") is not None:
        yield
        return
    empty = tmp_path_factory.mktemp("empty_models")
    monkeypatch.setenv("MEETING_ASR_MODELS_DIR", str(empty))
    yield


# --------------------------------------------------------------------------- #
# Fixture loader + synthetic-meeting generator (offline stand-ins)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


def _write_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm.astype(np.float32), rate)


def ensure_synthetic_fixture(name: str):
    """Ensure a deterministic synthetic meeting fixture exists on disk.

    Returns (audio_path, manifest_path). Generates speech-like audio with known
    speaker turns so the *plumbing* (capture → mix → fuse → session) is tested
    deterministically. Real-model WER/accuracy is gated separately (needs_models).
    """
    from tests._synth import SCENARIOS, build_scenario

    wav = AUDIO_DIR / f"{name}.wav"
    manifest = AUDIO_DIR / f"{name}.json"
    if wav.exists() and manifest.exists():
        return wav, manifest
    scenario = SCENARIOS[name]
    audio, rate, turns = build_scenario(scenario)
    _write_wav(wav, audio, rate)
    manifest.write_text(
        json.dumps({"name": name, "sample_rate": rate, "turns": turns}, indent=2)
    )
    return wav, manifest


@pytest.fixture
def synthetic_fixture():
    """Return a helper that materializes a named synthetic fixture → manifest dict."""
    def _get(name: str):
        wav, manifest = ensure_synthetic_fixture(name)
        data = json.loads(manifest.read_text())
        data["wav"] = str(wav)
        return data

    return _get


@pytest.fixture
def load_audio():
    """Helper: load a fixture WAV as (float32 mono numpy, sample_rate)."""
    import soundfile as sf

    def _load(path):
        pcm, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        return pcm.astype(np.float32), rate

    return _load
