"""app — the macOS desktop front-end (pywebview host + JS↔Python bridge).

Single-process desktop application: a ``pywebview`` WKWebView window renders the
static UI under ``app/web/`` and calls the in-process ``meeting_asr`` library
through the :class:`app.bridge.Api` JS-API bridge (no separate server, no
cross-language bridge — research Decision 1). ``app.main`` is the entry point
(``make run`` / ``python -m app.main``).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
