//! Tauri commands mirroring app/bridge.py:Api — same names, args, return
//! shapes, and events, so app/web/app.js runs unchanged. Phase 1 wires the
//! deterministic fakes; Phases 2–4 swap in real backends behind these commands.

use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use asr_memo_core::audio::health::{HealthFlag, HealthSnapshot};
use asr_memo_core::audio::mixer::MixedCapture;
use asr_memo_core::audio::record::WavWriter;
use asr_memo_core::fakes::{demo_script, FakeCapture, FakeDiarizer, FakeTranscriber};
use asr_memo_core::session::{Session, SessionEvent};
use asr_memo_core::traits::AudioCapture;
use asr_memo_core::types::{ErrorInfoDto, ReadinessDto, SegmentDto, SpeakerDto};

use crate::state::{AppState, CaptureSession};

// ---------- pure logic (unit-tested) ----------

pub fn fake_readiness() -> ReadinessDto {
    ReadinessDto {
        ready: true,
        compute_backend: "fake".into(),
        os_supports_process_tap: cfg!(target_os = "macos"),
        mic_permission: true,
        system_audio_permission: true,
        models: vec![],
        missing: vec![],
    }
}

/// Serialize a SessionEvent as the exact dict app/bridge.py:_emit sends.
pub fn event_payload(event: &SessionEvent) -> serde_json::Value {
    match event {
        SessionEvent::Status(s) => serde_json::json!({"type": "status", "status": s}),
        SessionEvent::Segment(seg) => serde_json::json!({"type": "segment", "segment": seg}),
        SessionEvent::Error(e) => serde_json::json!({"type": "error", "error": e}),
    }
}

pub fn render_markdown(segments: &[SegmentDto]) -> String {
    let mut out = String::from("# Transcript\n\n");
    for s in segments {
        out.push_str(&format!(
            "- [{:.1}s–{:.1}s] **{}**: {}\n",
            s.start, s.end, s.speaker_label, s.text
        ));
    }
    out
}

fn speakers_view(segments: &[SegmentDto]) -> Vec<SpeakerDto> {
    // Canonical palette from src/meeting_asr/export/palette.py::SPEAKER_COLORS.
    const COLORS: [&str; 4] = ["#1A7F64", "#2D7FF9", "#B4515C", "#9B59B6"];
    let mut labels: Vec<String> = Vec::new();
    for s in segments {
        if !labels.contains(&s.speaker_label) {
            labels.push(s.speaker_label.clone());
        }
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(i, label)| {
            let segs: Vec<&SegmentDto> = segments
                .iter()
                .filter(|s| s.speaker_label == label)
                .collect();
            SpeakerDto {
                color: COLORS[i % COLORS.len()].into(),
                total_speech_seconds: segs.iter().map(|s| s.end - s.start).sum(),
                segment_count: segs.len() as u32,
                label,
            }
        })
        .collect()
}

/// `audio_health` event payload — locked shape consumed by app.js:
/// `{type:"audio_health", mic:{rms,peak}, system:{rms,peak}, flags:[...]}`.
/// Pure (no `AppState`); unit-tested for shape stability.
pub fn audio_health_payload(snap: &HealthSnapshot) -> serde_json::Value {
    let flags: Vec<&str> = snap
        .flags
        .iter()
        .map(|f| match f {
            HealthFlag::SystemSilent => "system_silent",
            HealthFlag::RateChanged => "rate_changed",
        })
        .collect();
    serde_json::json!({
        "type": "audio_health",
        "mic": { "rms": snap.mic.rms, "peak": snap.mic.peak },
        "system": { "rms": snap.system.rms, "peak": snap.system.peak },
        "flags": flags,
    })
}

fn err(code: &str, message: &str, hint: &str) -> serde_json::Value {
    serde_json::json!({"error": ErrorInfoDto {
        code: code.into(), message: message.into(), recoverable: true, hint: Some(hint.into()),
    }})
}

// ---------- Tauri commands ----------

#[tauri::command(rename_all = "snake_case")]
pub fn get_readiness() -> ReadinessDto {
    fake_readiness()
}

#[tauri::command(rename_all = "snake_case")]
pub fn prepare<R: Runtime>(app: AppHandle<R>) -> serde_json::Value {
    // Fake mode: everything is already "ready"; complete immediately.
    let _ = app.emit(
        "backend-event",
        serde_json::json!({"type": "prepare_done", "readiness": fake_readiness()}),
    );
    serde_json::json!({"status": "prepared"})
}

#[tauri::command(rename_all = "snake_case")]
pub fn start_live<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    sources: Vec<String>,
    language_hint: Option<String>,
) -> serde_json::Value {
    let _ = sources; // fake mode ignores source selection
    if state.session.lock().unwrap().is_some() {
        return err(
            "busy",
            "a session is already running",
            "Stop the current session first.",
        );
    }
    state.transcript.lock().unwrap().clear();

    // Emit "starting" synchronously before the worker spawns (mirrors
    // app/bridge.py:252), so the UI badge transitions starting → recording
    // rather than the worker's "recording" being clobbered by the resolved
    // invoke handler running beginSession("starting") afterward.
    let _ = app.emit(
        "backend-event",
        event_payload(&SessionEvent::Status("starting".into())),
    );

    let (frames, tokens) = demo_script(8.0);
    let app_for_events = app.clone();
    let session = Session::run(
        Box::new(FakeCapture::from_script(frames)),
        Box::new(FakeTranscriber::with_tokens(tokens)),
        Box::new(FakeDiarizer::alternating(2.0)),
        language_hint,
        Box::new(move |event| {
            if let SessionEvent::Segment(seg) = &event {
                let st: State<'_, AppState> = app_for_events.state();
                st.transcript.lock().unwrap().push(seg.clone());
            }
            let _ = app_for_events.emit("backend-event", event_payload(&event));
        }),
    );
    match session {
        Ok(s) => {
            *state.session.lock().unwrap() = Some(s);
            serde_json::json!({"app_session_id": "live-1"})
        }
        Err(e) => err("start.failed", &e.to_string(), "Try again."),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn stop_session(state: State<'_, AppState>) -> serde_json::Value {
    if let Some(s) = state.session.lock().unwrap().take() {
        s.stop(); // emits final segments + status=stopped via on_event
    }
    serde_json::json!({"status": "stopped"})
}

#[tauri::command(rename_all = "snake_case")]
pub fn get_transcript(state: State<'_, AppState>) -> serde_json::Value {
    let segments = state.transcript.lock().unwrap().clone();
    serde_json::json!({"speakers": speakers_view(&segments), "segments": segments})
}

#[tauri::command(rename_all = "snake_case")]
pub fn export_transcript(
    state: State<'_, AppState>,
    path: String,
    format: Option<String>,
) -> serde_json::Value {
    let segments = state.transcript.lock().unwrap().clone();
    if segments.is_empty() {
        return err(
            "export.empty",
            "no transcript to export",
            "Produce a transcript first, then export.",
        );
    }
    let fmt = format.unwrap_or_else(|| {
        if path.ends_with(".json") {
            "json".into()
        } else {
            "markdown".into()
        }
    });
    let body = match fmt.as_str() {
        "json" => serde_json::to_string_pretty(
            &serde_json::json!({"speakers": speakers_view(&segments), "segments": segments}),
        )
        .unwrap(),
        "markdown" | "md" => render_markdown(&segments),
        other => {
            return err(
                "export.format",
                &format!("unknown format: {other}"),
                "Choose Markdown (.md) or JSON (.json).",
            )
        }
    };
    match std::fs::write(&path, body) {
        Ok(()) => serde_json::json!({"path": path}),
        Err(e) => err(
            "export.write_failed",
            &e.to_string(),
            "Could not write the file — check disk space and the chosen path.",
        ),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn transcribe_file(path: String, language_hint: Option<String>) -> serde_json::Value {
    let _ = (path, language_hint);
    err(
        "not_implemented",
        "file import lands in Phase 3 (ASR port)",
        "Use Start (live fake demo) in Phase 1.",
    )
}

#[tauri::command(rename_all = "snake_case")]
pub async fn pick_audio_file<R: Runtime>(app: AppHandle<R>) -> serde_json::Value {
    // blocking_pick_file() must NOT run on the main thread or an async-runtime
    // worker — it freezes the app (tauri-plugin-dialog docs; meetily hit the
    // same bug and wraps the call in spawn_blocking). Run it on the blocking pool.
    let path = tauri::async_runtime::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;
        app.dialog()
            .file()
            .add_filter("Audio", &["wav", "flac", "mp3", "m4a"])
            .blocking_pick_file()
            .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .ok()
    .flatten();
    serde_json::json!({"path": path})
}

#[tauri::command(rename_all = "snake_case")]
pub async fn pick_export_path<R: Runtime>(app: AppHandle<R>, format: String) -> serde_json::Value {
    let ext = if format == "json" { "json" } else { "md" };
    let path = tauri::async_runtime::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;
        app.dialog()
            .file()
            .add_filter(&format, &[ext])
            .set_file_name(format!("transcript.{ext}"))
            .blocking_save_file()
            .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .ok()
    .flatten();
    serde_json::json!({"path": path})
}

// ---------- capture/record mode (Phase 2a) ----------
//
// A distinct capture/record mode that runs the audio engine end-to-end with no
// transcript (ASR lands in Phase 3). start_capture wires mic + system sources
// into the mixer, ships mixed frames (upsampled 16k→48k) to a WAV writer, and
// emits `audio_health` events; stop_capture joins the mixer, closes the writer
// (by value), and returns the file path.

#[tauri::command(rename_all = "snake_case")]
pub fn start_capture<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    sources: Vec<String>,
    meeting_name: Option<String>,
) -> serde_json::Value {
    use asr_memo_core::audio::mic::CpalMicrophoneCapture;

    if state.capture.lock().unwrap().is_some() {
        return err(
            "capture.busy",
            "a capture is already running",
            "Stop the current recording first.",
        );
    }
    let want_mic = sources.iter().any(|s| s == "microphone");
    let want_sys = sources.iter().any(|s| s == "system");
    if !want_mic && !want_sys {
        return err(
            "capture.sources",
            "no sources selected",
            "Choose microphone and/or system audio.",
        );
    }

    let dir = recording_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return err(
            "capture.dir",
            &e.to_string(),
            "Could not create the recordings directory.",
        );
    }
    // Sanitize the meeting name: restrict to a safe filename charset so a
    // user-supplied value can't escape the recordings dir (no `/`, `\`, `..`).
    let name = {
        let cleaned: String = meeting_name
            .unwrap_or_default()
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
            .collect();
        if cleaned.trim().is_empty() {
            "meeting".to_string()
        } else {
            cleaned
        }
    };
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let path = dir.join(format!("{name}-{stamp}.wav"));

    let writer = match WavWriter::create(&path, 48000) {
        Ok(w) => w,
        Err(e) => {
            return err(
                "capture.file",
                &e.to_string(),
                "Could not open the recording file.",
            )
        }
    };
    // Shared between the mixer thread (writes frames) and stop_capture (closes
    // by value). `Option<>` lets stop_capture `take()` it once the mixer thread
    // has been joined and dropped its Arc clone.
    let writer: Arc<Mutex<Option<WavWriter>>> = Arc::new(Mutex::new(Some(writer)));

    // Health callback runs on the mixer thread; clone the AppHandle and emit
    // `audio_health` per snapshot.
    let app_for_health = app.clone();
    let on_health = Box::new(move |snap: HealthSnapshot| {
        let _ = app_for_health.emit("backend-event", audio_health_payload(&snap));
    });

    let mic: Option<Box<dyn AudioCapture>> = if want_mic {
        Some(Box::new(CpalMicrophoneCapture::new()))
    } else {
        None
    };
    let sys: Option<Box<dyn AudioCapture>> = if want_sys {
        #[cfg(target_os = "macos")]
        {
            Some(Box::new(
                asr_memo_core::audio::system::macos::CoreAudioTapCapture::new(),
            ))
        }
        // Phase 2b: Windows/Linux system capture not implemented.
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    } else {
        None
    };

    // Frame callback runs on the mixer thread: upsample 16k→48k for the
    // recording (zero-order hold ×3) and write, then emit a capture_frame tick.
    let app_for_frames = app.clone();
    let writer_for_cb = writer.clone();
    let mut mixer = MixedCapture::new(mic, sys, on_health);
    if let Err(e) = mixer.start(Box::new(move |frame| {
        if let Some(w) = writer_for_cb.lock().unwrap().as_mut() {
            let up = upsample_16_to_48(&frame.pcm);
            let _ = w.write_samples(&up);
        }
        let _ = app_for_frames.emit(
            "backend-event",
            serde_json::json!({
                "type": "capture_frame",
                "samples": frame.pcm.len(),
                "t": frame.t_end,
            }),
        );
    })) {
        return err(
            "capture.start",
            &e.to_string(),
            "Could not start audio capture.",
        );
    }

    let path_str = path.to_string_lossy().into_owned();
    *state.capture.lock().unwrap() = Some(CaptureSession {
        mixer,
        writer,
        path: path_str.clone(),
    });
    let _ = app.emit(
        "backend-event",
        serde_json::json!({"type": "status", "status": "recording"}),
    );
    serde_json::json!({"path": path_str})
}

#[tauri::command(rename_all = "snake_case")]
pub fn stop_capture(state: State<'_, AppState>) -> serde_json::Value {
    let session = state.capture.lock().unwrap().take();
    match session {
        Some(mut s) => {
            // Join the mixer thread FIRST — this drops its Arc<Mutex<...>>
            // clone, so once stop() returns we are the sole owner of the writer
            // and can `take()` + `close()` it by value.
            s.mixer.stop();
            let writer_opt = s.writer.lock().unwrap().take();
            if let Some(w) = writer_opt {
                if let Err(e) = w.close() {
                    return err(
                        "capture.close",
                        &e.to_string(),
                        "Recording may be incomplete — check the file.",
                    );
                }
            }
            serde_json::json!({"path": s.path})
        }
        None => serde_json::json!({ "path": serde_json::Value::Null }),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn reveal_recording(path: String) -> serde_json::Value {
    #[cfg(target_os = "macos")]
    {
        let p = std::path::Path::new(&path);
        if p.exists() {
            let _ = std::process::Command::new("open").arg("-R").arg(p).status();
            return serde_json::json!({"revealed": true});
        }
    }
    err(
        "capture.reveal",
        "recording not found or reveal unsupported on this OS",
        &format!("Record first, then reveal. Looked for: {path}"),
    )
}

/// Zero-order hold ×3: repeats each 16 kHz sample three times for 48 kHz. Good
/// enough for a speech recording (no anti-imaging filter, but the file is for
/// human review / re-decode, not model input).
fn upsample_16_to_48(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() * 3);
    for &s in input {
        out.extend_from_slice(&[s, s, s]);
    }
    out
}

/// `~/Library/Application Support/ASR-Memo/recordings` on macOS (via the `dirs`
/// crate, which handles the platform data dir correctly). Falls back to a local
/// `recordings/` directory if `dirs` can't resolve a data dir.
fn recording_dir() -> std::path::PathBuf {
    if let Some(proj) = dirs::data_dir() {
        return proj.join("ASR-Memo").join("recordings");
    }
    std::path::PathBuf::from("recordings")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_readiness_is_ready() {
        let r = fake_readiness();
        assert!(r.ready);
        assert_eq!(r.compute_backend, "fake");
        assert!(r.missing.is_empty());
    }

    #[test]
    fn audio_health_payload_shape() {
        let snap = HealthSnapshot {
            mic: asr_memo_core::audio::health::SourceHealth {
                rms: 0.1,
                peak: 0.5,
            },
            system: asr_memo_core::audio::health::SourceHealth {
                rms: 0.0,
                peak: 0.0,
            },
            flags: vec![HealthFlag::SystemSilent],
        };
        let v = audio_health_payload(&snap);
        assert_eq!(v["type"], "audio_health");
        assert_eq!(v["mic"]["peak"], 0.5);
        assert_eq!(v["system"]["rms"], 0.0);
        assert_eq!(v["flags"][0], "system_silent");
    }

    #[test]
    fn segment_event_payload_matches_bridge_contract() {
        let seg = asr_memo_core::types::SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.0,
            text: "hi".into(),
            language: None,
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: None,
            is_final: true,
        };
        let v = event_payload(&asr_memo_core::session::SessionEvent::Segment(seg));
        assert_eq!(v["type"], "segment");
        assert_eq!(v["segment"]["speaker_label"], "S1");
    }

    #[test]
    fn status_and_error_event_payloads_match_bridge_contract() {
        let v = event_payload(&asr_memo_core::session::SessionEvent::Status(
            "recording".into(),
        ));
        assert_eq!(
            v,
            serde_json::json!({"type": "status", "status": "recording"})
        );
        let e = asr_memo_core::types::ErrorInfoDto {
            code: "x".into(),
            message: "m".into(),
            recoverable: true,
            hint: None,
        };
        let v = event_payload(&asr_memo_core::session::SessionEvent::Error(e));
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["code"], "x");
    }

    #[test]
    fn export_markdown_writes_speaker_lines() {
        let seg = asr_memo_core::types::SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.5,
            text: "hello world".into(),
            language: Some("en".into()),
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: None,
            is_final: true,
        };
        let md = render_markdown(&[seg]);
        assert!(md.contains("**S1**"));
        assert!(md.contains("hello world"));
    }

    #[test]
    fn speakers_view_assigns_palette_and_sums_speech() {
        let mk = |label: &str, start: f64, end: f64| asr_memo_core::types::SegmentDto {
            segment_id: format!("s-{label}-{start}"),
            speaker_label: label.into(),
            start,
            end,
            text: "x".into(),
            language: None,
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: None,
            is_final: true,
        };
        // S1 appears twice (non-contiguous) → arrival order first, durations summed.
        let segs = vec![mk("S1", 0.0, 2.0), mk("S2", 2.0, 3.0), mk("S1", 5.0, 6.0)];
        let spk = speakers_view(&segs);
        assert_eq!(spk.len(), 2);
        assert_eq!(spk[0].label, "S1");
        assert_eq!(spk[0].color, "#1A7F64");
        assert_eq!(spk[0].total_speech_seconds, 3.0); // (2-0) + (6-5)
        assert_eq!(spk[0].segment_count, 2);
        assert_eq!(spk[1].label, "S2");
        assert_eq!(spk[1].color, "#2D7FF9");
        assert_eq!(spk[1].total_speech_seconds, 1.0); // (3-2)
        assert_eq!(spk[1].segment_count, 1);
    }
}
