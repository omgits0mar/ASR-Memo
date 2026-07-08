/* Fabricates window.pywebview.api over Tauri invoke so app.js runs unchanged
 * under both shells. Loaded before app.js; inert under real pywebview. */
(function () {
  if (!window.__TAURI__) return; // running under pywebview — do nothing
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  window.pywebview = {
    api: {
      get_readiness: () => invoke("get_readiness"),
      prepare: () => invoke("prepare"),
      start_live: (sources, language_hint = null) =>
        invoke("start_live", { sources, language_hint }),
      stop_session: () => invoke("stop_session"),
      transcribe_file: (path, language_hint = null) =>
        invoke("transcribe_file", { path, language_hint }),
      pick_audio_file: () => invoke("pick_audio_file"),
      pick_export_path: (format) => invoke("pick_export_path", { format }),
      export_transcript: (path, format = null) =>
        invoke("export_transcript", { path, format }),
      get_transcript: () => invoke("get_transcript"),
    },
  };

  listen("backend-event", (e) => {
    if (typeof window.onBackendEvent === "function") window.onBackendEvent(e.payload);
  });

  // app.js boots on this event (see its `pywebviewready` listener + ready-poll).
  window.dispatchEvent(new Event("pywebviewready"));
})();
