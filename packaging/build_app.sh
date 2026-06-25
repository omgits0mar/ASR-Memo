#!/usr/bin/env bash
# Build MeetingAssistant.app — PyInstaller bundle + ad-hoc codesign (task T036 / US3).
#
#   bash packaging/build_app.sh          # or:  make app
#
# Produces dist/MeetingAssistant.app (self-contained, ad-hoc signed, double-click).
# Models are NOT bundled — in-app guided setup downloads them on first run.
# Requires PyInstaller (pip install -e ".[packaging]") + the runtime native deps.
set -euo pipefail

# Resolve repo root (this script lives in packaging/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
APP="dist/MeetingAssistant.app"

# 0. Preflight: PyInstaller must be importable.
if ! "$PY" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[build_app] PyInstaller not installed. Run: pip install -e \".[packaging]\"" >&2
  exit 1
fi

# 1. (Optional) Build the Swift Process-Tap helper if a toolchain is present.
if [[ -d native/AudioTap ]] && command -v swift >/dev/null 2>&1; then
  echo "[build_app] building Swift AudioTap helper…"
  swift build -c release --package-path native/AudioTap \
    || echo "[build_app] Swift build failed — continuing with a mic-only build"
fi

# 2. PyInstaller bundle (--windowed via the spec's EXE console=False).
echo "[build_app] running PyInstaller…"
"$PY" -m PyInstaller packaging/MeetingAssistant.spec \
  --windowed --noconfirm --distpath dist --workpath build

if [[ ! -d "$APP" ]]; then
  echo "[build_app] ERROR: $APP was not produced" >&2
  exit 1
fi

# 3. Merge the canonical usage descriptions from Info.plist.in into the bundle.
PLIST="$APP/Contents/Info.plist"
if [[ -f packaging/Info.plist.in ]] && [[ -x /usr/libexec/PlistBuddy ]]; then
  echo "[build_app] merging usage descriptions into Info.plist…"
  merge_key() {
    local key="$1" val
    val="$(/usr/libexec/PlistBuddy -c "Print :$key" packaging/Info.plist.in 2>/dev/null || true)"
    [[ -n "$val" ]] || return 0
    /usr/libexec/PlistBuddy -c "Delete :$key" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :$key string $val" "$PLIST" 2>/dev/null || true
  }
  merge_key NSMicrophoneUsageDescription
  merge_key NSSystemAudioRecordingUsageDescription
  merge_key NSAudioFileUsageDescription
fi

# 4. Ad-hoc codesign (--deep signs bundled frameworks/binaries too).
echo "[build_app] ad-hoc codesign (--deep)…"
codesign --force --deep --sign - "$APP"

# 5. Verify the signature.
echo "[build_app] verifying signature…"
if codesign --verify --verbose=2 "$APP" 2>&1 | head -5; then
  echo "✓ built: $APP"
  echo "  open $APP   (first launch: guided model setup → Ready; <30s on relaunch)"
else
  echo "⚠ built $APP but codesign verification reported issues (ad-hoc signing)" >&2
fi
