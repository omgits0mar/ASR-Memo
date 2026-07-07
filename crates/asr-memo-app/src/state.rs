//! Shared app state: the running session and the accumulated transcript.

use std::sync::Mutex;

use asr_memo_core::session::Session;
use asr_memo_core::types::SegmentDto;

#[derive(Default)]
pub struct AppState {
    pub session: Mutex<Option<Session>>,
    pub transcript: Mutex<Vec<SegmentDto>>,
}
