"""pywebview host entry point (task T011; ``make run`` / ``python -m app.main``).

Creates the macOS-native ``WKWebView`` window, loads the static UI under
``app/web/``, mounts the :class:`app.bridge.Api` as ``js_api``, wires the
``evaluate_js`` event channel + native pickers, and starts the webview loop.
``pywebview`` is lazy-imported so the rest of ``app/`` stays importable headlessly.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["web_index_path", "main"]


def web_index_path() -> Path:
    """Resolve the bundled ``app/web/index.html`` (dev repo + PyInstaller _MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "app" / "web" / "index.html"


def main() -> int:
    """Launch the desktop app. Returns the webview exit code."""
    try:
        import webview  # type: ignore
    except ImportError:  # pragma: no cover - env dependent
        sys.stderr.write(
            "pywebview is not installed. Run `make setup` (or `pip install pywebview`) first.\n"
        )
        return 2

    from app.bridge import Api

    api = Api()
    index = web_index_path()
    if not index.exists():  # pragma: no cover - packaging dependent
        sys.stderr.write(f"UI not found: {index}\n")
        return 3

    window = webview.create_window(
        title="Meeting Assistant",
        url=str(index),
        js_api=api,
        width=1000,
        height=720,
        min_size=(760, 520),
        text_select=True,
    )
    api.attach(window)  # emit → window.evaluate_js; pickers → file dialogs
    webview.start()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual launch
    raise SystemExit(main())
