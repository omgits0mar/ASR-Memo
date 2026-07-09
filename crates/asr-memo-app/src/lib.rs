mod commands;
mod state;

use state::AppState;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::get_readiness,
            commands::prepare,
            commands::start_live,
            commands::stop_session,
            commands::get_transcript,
            commands::export_transcript,
            commands::transcribe_file,
            commands::pick_audio_file,
            commands::pick_export_path,
            commands::start_capture,
            commands::stop_capture,
            commands::reveal_recording,
        ])
        .run(tauri::generate_context!())
        .expect("error while running ASR-Memo");
}
