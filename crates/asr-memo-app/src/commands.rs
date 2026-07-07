//! Tauri commands mirroring app/bridge.py:Api — same names, args, return
//! shapes, and events, so app/web/app.js runs unchanged. Phase 1 wires the
//! deterministic fakes; Phases 2–4 swap in real backends behind these commands.

use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use asr_memo_core::fakes::{demo_script, FakeCapture, FakeDiarizer, FakeTranscriber};
use asr_memo_core::session::{Session, SessionEvent};
use asr_memo_core::types::{ErrorInfoDto, ReadinessDto, SegmentDto, SpeakerDto};

use crate::state::AppState;

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
pub fn pick_audio_file<R: Runtime>(app: AppHandle<R>) -> serde_json::Value {
    use tauri_plugin_dialog::DialogExt;
    let path = app
        .dialog()
        .file()
        .add_filter("Audio", &["wav", "flac", "mp3", "m4a"])
        .blocking_pick_file()
        .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()));
    serde_json::json!({"path": path})
}

#[tauri::command(rename_all = "snake_case")]
pub fn pick_export_path<R: Runtime>(app: AppHandle<R>, format: String) -> serde_json::Value {
    use tauri_plugin_dialog::DialogExt;
    let ext = if format == "json" { "json" } else { "md" };
    let path = app
        .dialog()
        .file()
        .add_filter(&format, &[ext])
        .set_file_name(format!("transcript.{ext}"))
        .blocking_save_file()
        .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()));
    serde_json::json!({"path": path})
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
}
