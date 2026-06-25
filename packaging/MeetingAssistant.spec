# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MeetingAssistant.app (task T034 / US3 packaging).

Builds a self-contained, ad-hoc-signed, double-click ``.app``:
  * entry: ``app/main.py`` (the pywebview host)
  * bundled data: the static UI (``app/web/``) + the built Swift AudioTap helper
  * runtime: the ``meeting_asr`` package + onnxruntime/coremltools native libs

Models are NOT bundled — the in-app guided setup downloads them on first run and
caches them, so relaunch reaches Ready fast (research Decision 6). Build with::

    bash packaging/build_app.sh        # or:  pyinstaller packaging/MeetingAssistant.spec --windowed --noconfirm

Verified end-to-end (double-click → guided setup → Ready) under T037 on a machine
with PyInstaller + the native deps installed; the spec itself is build tooling.
"""

from pathlib import Path

REPO_ROOT = Path(SPECPATH).resolve().parent  # SPECPATH = packaging/
APP_ENTRY = str(REPO_ROOT / "app" / "main.py")
WEB_DIR = REPO_ROOT / "app" / "web"
SRC_DIR = REPO_ROOT / "src"
# The Swift Process-Tap helper, if built (build_app.sh builds it first).
AUDIOTAP = REPO_ROOT / "native" / "AudioTap" / ".build" / "release" / "AudioTap"

datas = []
if WEB_DIR.is_dir():
    datas.append((str(WEB_DIR), "app/web"))
if AUDIOTAP.exists():
    datas.append((str(AUDIOTAP), "."))
# Bundled mel filterbank for the ASR log-mel front end (loaded via importlib path).
_MEL_FB = SRC_DIR / "meeting_asr" / "asr" / "mel_fb_128_512.npy"
if _MEL_FB.exists():
    datas.append((str(_MEL_FB), "meeting_asr/asr"))

hiddenimports = [
    # pywebview Cocoa backend (WKWebView) + its dependencies.
    "webview",
    "webview.platforms.cocoa",
    # Heavy ML / audio backends (lazy-imported; force-include their native libs).
    "onnxruntime",
    "coremltools",
    "soundfile",
    "sounddevice",
    "soxr",
    "numpy",
]

a = Analysis(
    [APP_ENTRY],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,        # --windowed: no Terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    a.binaries,
    a.datas,
    [],
    name="MeetingAssistant",
    icon=None,
    bundle_identifier="com.meetingassistant.app",
    info_plist={
        "CFBundleName": "Meeting Assistant",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": (
            "Meeting Assistant transcribes your meetings locally. "
            "Microphone access is needed to capture nearby speech."
        ),
        # System-audio capture via Core Audio Process Taps (macOS 14.4+) requires a
        # usage-description string so coreaudiod/TCC can present the audio-capture
        # prompt; without it the tap's IOProc RPC stalls (the helper's watchdog then
        # exits as a permission denial). The exact key has shifted across releases,
        # so declare both the system-audio-recording and audio-capture variants.
        "NSSystemAudioRecordingUsageDescription": (
            "Meeting Assistant can transcribe remote participants by capturing "
            "system audio. This stays on your device."
        ),
        "NSAudioCaptureUsageDescription": (
            "Meeting Assistant can transcribe remote participants by capturing "
            "system audio. This stays on your device."
        ),
        "NSAudioFileUsageDescription": (
            "Meeting Assistant reads audio files you choose to transcribe them "
            "locally on your Mac."
        ),
    },
)
