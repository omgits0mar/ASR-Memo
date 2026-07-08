//! Deterministic fakes mirroring tests/_fakes.py — no audio, no models, no sleep.

use std::thread::JoinHandle;

use crate::traits::*;
use crate::types::AudioSource;

/// Emits a scripted frame sequence on a background thread, then finishes.
///
/// `stop()` joins the spawned thread; the script has no I/O and no sleep, so it
/// drains to completion in microseconds. There is no early-abort path: the
/// Python reference (`FixtureCapture`) uses a `threading.Event` for that, but a
/// Rust port racing `store(true)` against an unscheduled worker reliably drops
/// every frame when `stop()` follows `start()` immediately (as the test does).
/// Keeping `stop()` a plain `join` makes emission deterministic.
pub struct FakeCapture {
    script: Option<Vec<AudioFrame>>,
    handle: Option<JoinHandle<()>>,
}

impl FakeCapture {
    pub fn from_script(script: Vec<AudioFrame>) -> Self {
        Self {
            script: Some(script),
            handle: None,
        }
    }
}

impl AudioCapture for FakeCapture {
    fn start(&mut self, mut on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        let script = self.script.take().ok_or_else(|| CoreError::Capture {
            code: "capture.restart".into(),
            message: "FakeCapture can only start once".into(),
        })?;
        self.handle = Some(std::thread::spawn(move || {
            for f in script {
                on_frame(f);
            }
        }));
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Releases scripted tokens once the stream time passes their t_end.
pub struct FakeTranscriber {
    pending: Vec<AsrToken>,
}

impl FakeTranscriber {
    pub fn with_tokens(mut tokens: Vec<AsrToken>) -> Self {
        tokens.sort_by(|a, b| a.t_end.total_cmp(&b.t_end));
        Self { pending: tokens }
    }
}

impl SpeechTranscriber for FakeTranscriber {
    fn reset(&mut self) {
        self.pending.clear();
    }

    fn push(&mut self, frame: &AudioFrame) -> Vec<AsrToken> {
        let split = self.pending.partition_point(|t| t.t_end <= frame.t_end);
        self.pending.drain(..split).collect()
    }

    fn flush(&mut self) -> Vec<AsrToken> {
        std::mem::take(&mut self.pending)
    }
}

/// Speaker 0/1 alternating every `period_secs`; one DiarFrame per 80 ms.
pub struct FakeDiarizer {
    period_secs: f64,
}

impl FakeDiarizer {
    pub fn alternating(period_secs: f64) -> Self {
        Self { period_secs }
    }
}

impl SpeakerDiarizer for FakeDiarizer {
    fn reset(&mut self) {}

    fn push(&mut self, frame: &AudioFrame) -> Vec<DiarFrame> {
        const STEP: f64 = 0.08;
        let mut out = Vec::new();
        let mut t = frame.t_start;
        while t < frame.t_end - 1e-9 {
            let speaker = ((t / self.period_secs) as u64 % 2) as u8;
            out.push(DiarFrame {
                t,
                speaker: Some(speaker),
            });
            t += STEP;
        }
        out
    }
}

/// Canned live-demo content: 0.5 s silent frames plus a scripted two-speaker
/// conversation, `seconds` long. Task 5 streams this to the UI in fake mode.
pub fn demo_script(seconds: f64) -> (Vec<AudioFrame>, Vec<AsrToken>) {
    let mut frames = Vec::new();
    let mut t = 0.0;
    while t < seconds {
        let t1 = (t + 0.5).min(seconds);
        frames.push(AudioFrame {
            pcm: vec![0.0; ((t1 - t) * 16000.0) as usize],
            t_start: t,
            t_end: t1,
            source: AudioSource::Microphone,
        });
        t = t1;
    }
    let words = [
        "this",
        "is",
        "a",
        "fake",
        "session",
        "streaming",
        "live",
        "segments",
    ];
    let tokens = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let t0 = 0.2 + i as f64 * (seconds - 0.4) / words.len() as f64;
            AsrToken {
                text: (*w).into(),
                t_start: t0,
                t_end: t0 + 0.25,
                confidence: 0.92,
            }
        })
        .filter(|tok| tok.t_end < seconds)
        .collect();
    (frames, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioSource;
    use std::sync::mpsc;

    fn frame(t0: f64, t1: f64) -> AudioFrame {
        let n = ((t1 - t0) * 16000.0) as usize;
        AudioFrame {
            pcm: vec![0.0; n],
            t_start: t0,
            t_end: t1,
            source: AudioSource::Microphone,
        }
    }

    #[test]
    fn fake_capture_emits_script_then_stops() {
        let script = vec![frame(0.0, 0.5), frame(0.5, 1.0)];
        let mut cap = FakeCapture::from_script(script);
        let (tx, rx) = mpsc::channel();
        cap.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        cap.stop(); // joins the emit thread
        let got: Vec<AudioFrame> = rx.try_iter().collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].t_start, 0.0);
        assert_eq!(got[1].t_end, 1.0);
    }

    #[test]
    fn fake_transcriber_releases_tokens_by_frame_time() {
        let tokens = vec![
            AsrToken {
                text: "hello".into(),
                t_start: 0.2,
                t_end: 0.5,
                confidence: 0.9,
            },
            AsrToken {
                text: "world".into(),
                t_start: 0.6,
                t_end: 0.9,
                confidence: 0.8,
            },
        ];
        let mut asr = FakeTranscriber::with_tokens(tokens);
        let out1 = asr.push(&frame(0.0, 0.5));
        assert_eq!(out1.len(), 1); // only "hello" has t_end <= 0.5
        assert_eq!(out1[0].text, "hello");
        let out2 = asr.flush();
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].text, "world");
    }

    #[test]
    fn fake_diarizer_alternates_speakers_by_period() {
        let mut diar = FakeDiarizer::alternating(1.0);
        let frames = diar.push(&frame(0.0, 2.0));
        // one DiarFrame per 80 ms over [0.0, 2.0) → 25 frames
        assert_eq!(frames.len(), 25);
        assert_eq!(frames[0].speaker, Some(0)); // t=0.00 → first period
        assert_eq!(frames[13].speaker, Some(1)); // t=1.04 → second period
    }

    #[test]
    fn demo_script_is_nonempty_and_time_ordered() {
        let (frames, tokens) = demo_script(4.0);
        assert!(!frames.is_empty() && !tokens.is_empty());
        assert!(frames.windows(2).all(|w| w[0].t_end <= w[1].t_start + 1e-9));
        assert!(tokens.windows(2).all(|w| w[0].t_start <= w[1].t_start));
    }
}
