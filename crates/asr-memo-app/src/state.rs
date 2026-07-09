//! Shared app state: the running session, the accumulated transcript, and the
//! capture/record session (Phase 2a — audio only, no transcript).

use std::sync::{Arc, Mutex};

use asr_memo_core::audio::mixer::MixedCapture;
use asr_memo_core::audio::record::WavWriter;
use asr_memo_core::session::Session;
use asr_memo_core::types::SegmentDto;

#[derive(Default)]
pub struct AppState {
    pub session: Mutex<Option<Session>>,
    pub transcript: Mutex<Vec<SegmentDto>>,
    pub capture: Mutex<Option<CaptureSession>>,
}

/// A live capture/record session: the mixer plus the shared WAV writer.
///
/// `writer` is `Arc<Mutex<Option<WavWriter>>>` because two owners need it:
/// the mixer thread (frame callback writes samples) and `stop_capture`
/// (closes by value). `stop_capture` joins the mixer first — dropping the
/// thread's `Arc` clone — so by the time `take()` runs we are the sole owner
/// and `WavWriter::close(self)` (by value) is sound.
pub struct CaptureSession {
    pub mixer: MixedCapture,
    pub writer: Arc<Mutex<Option<WavWriter>>>,
    pub path: String,
}
