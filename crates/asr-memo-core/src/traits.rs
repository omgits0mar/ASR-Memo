//! Backend traits — the Rust equivalents of src/meeting_asr's protocols.
//! Real implementations arrive in Phases 2–4; Phase 1 ships fakes only.

use crate::types::AudioSource;

/// A block of 16 kHz mono f32 PCM on the session clock.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    pub pcm: Vec<f32>,
    pub t_start: f64,
    pub t_end: f64,
    pub source: AudioSource,
}

/// One decoded word with timestamps (the aligner's input unit).
#[derive(Debug, Clone, PartialEq)]
pub struct AsrToken {
    pub text: String,
    pub t_start: f64,
    pub t_end: f64,
    pub confidence: f64,
}

/// One 80 ms diarizer frame: active speaker index (arrival order) or silence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarFrame {
    pub t: f64,
    pub speaker: Option<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("{code}: {message}")]
    Capture { code: String, message: String },
}

pub trait SpeechTranscriber: Send {
    fn reset(&mut self);
    /// Push one frame; returns the tokens finalized by this push.
    fn push(&mut self, frame: &AudioFrame) -> Vec<AsrToken>;
    /// End of stream; returns any remaining tokens.
    fn flush(&mut self) -> Vec<AsrToken>;
}

pub trait SpeakerDiarizer: Send {
    fn reset(&mut self);
    /// Push one frame; returns the 80 ms speaker frames it covers.
    fn push(&mut self, frame: &AudioFrame) -> Vec<DiarFrame>;
}

pub trait AudioCapture: Send {
    /// Start delivering frames to `on_frame` (from any thread). Idempotent stop.
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError>;
    fn stop(&mut self);
}
