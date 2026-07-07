//! Session orchestrator: capture → (transcriber ∥ diarizer) → fused SegmentDto
//! events. Phase-1 fusion is the midpoint aligner (see plan note); the event
//! interface is what Phase 4's full aligner port will slot into.

use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::traits::*;
use crate::types::{ErrorInfoDto, SegmentDto};

#[derive(Debug)]
pub enum SessionEvent {
    Status(String),
    Segment(SegmentDto),
    Error(ErrorInfoDto),
}

const SEGMENT_GAP_SECS: f64 = 1.0;

/// Groups tokens by the speaker active at each token's midpoint.
struct MidpointAligner {
    diar: Vec<DiarFrame>,
    pending: Vec<(AsrToken, String)>, // token + speaker label
    seen_speakers: Vec<u8>,           // arrival order → "S1".."S4"
    next_segment_id: u64,
    language: Option<String>,
}

impl MidpointAligner {
    fn new(language: Option<String>) -> Self {
        Self { diar: Vec::new(), pending: Vec::new(), seen_speakers: Vec::new(), next_segment_id: 1, language }
    }

    fn label_for(&mut self, raw: u8) -> String {
        if !self.seen_speakers.contains(&raw) {
            self.seen_speakers.push(raw);
        }
        let idx = self.seen_speakers.iter().position(|s| *s == raw).unwrap();
        format!("S{}", idx + 1)
    }

    fn speaker_at(&self, t: f64) -> Option<u8> {
        self.diar.iter().rev().find(|f| f.t <= t).and_then(|f| f.speaker)
    }

    fn feed(&mut self, tokens: Vec<AsrToken>, diar: Vec<DiarFrame>) -> Vec<SegmentDto> {
        self.diar.extend(diar);
        let mut out = Vec::new();
        for tok in tokens {
            let mid = (tok.t_start + tok.t_end) / 2.0;
            let label = match self.speaker_at(mid) {
                Some(raw) => self.label_for(raw),
                None => "S1".to_string(), // no diar info yet → first speaker
            };
            let split = match self.pending.last() {
                Some((prev, prev_label)) => {
                    *prev_label != label || tok.t_start - prev.t_end > SEGMENT_GAP_SECS
                }
                None => false,
            };
            if split {
                out.push(self.close_segment());
            }
            self.pending.push((tok, label));
        }
        out
    }

    fn close_segment(&mut self) -> SegmentDto {
        let toks = std::mem::take(&mut self.pending);
        let label = toks[0].1.clone();
        let text = toks.iter().map(|(t, _)| t.text.as_str()).collect::<Vec<_>>().join(" ");
        let confidence =
            toks.iter().map(|(t, _)| t.confidence).sum::<f64>() / toks.len() as f64;
        let band = if confidence >= 0.8 { "high" } else if confidence >= 0.5 { "medium" } else { "low" };
        let seg = SegmentDto {
            segment_id: format!("seg-{}", self.next_segment_id),
            speaker_label: label,
            start: toks[0].0.t_start,
            end: toks.last().unwrap().0.t_end,
            text,
            language: self.language.clone(),
            confidence,
            confidence_band: Some(band.to_string()),
            source: None,
            is_final: true,
        };
        self.next_segment_id += 1;
        seg
    }

    fn flush(&mut self) -> Vec<SegmentDto> {
        if self.pending.is_empty() { Vec::new() } else { vec![self.close_segment()] }
    }
}

pub struct Session {
    capture: Box<dyn AudioCapture>,
    frame_tx: mpsc::Sender<Option<AudioFrame>>,
    worker: JoinHandle<()>,
}

impl Session {
    pub fn run(
        mut capture: Box<dyn AudioCapture>,
        mut transcriber: Box<dyn SpeechTranscriber>,
        mut diarizer: Box<dyn SpeakerDiarizer>,
        language_hint: Option<String>,
        mut on_event: Box<dyn FnMut(SessionEvent) + Send>,
    ) -> Result<Session, CoreError> {
        let (frame_tx, frame_rx) = mpsc::channel::<Option<AudioFrame>>();
        let worker = std::thread::spawn(move || {
            let mut aligner = MidpointAligner::new(language_hint);
            on_event(SessionEvent::Status("recording".into()));
            while let Ok(Some(frame)) = frame_rx.recv() {
                let tokens = transcriber.push(&frame);
                let diar = diarizer.push(&frame);
                for seg in aligner.feed(tokens, diar) {
                    on_event(SessionEvent::Segment(seg));
                }
            }
            // end of stream: flush the transcriber, then the aligner
            for seg in aligner.feed(transcriber.flush(), Vec::new()) {
                on_event(SessionEvent::Segment(seg));
            }
            for seg in aligner.flush() {
                on_event(SessionEvent::Segment(seg));
            }
            on_event(SessionEvent::Status("stopped".into()));
        });

        let cb_tx = frame_tx.clone();
        capture.start(Box::new(move |f| {
            let _ = cb_tx.send(Some(f));
        }))?;

        Ok(Session { capture, frame_tx, worker })
    }

    pub fn stop(mut self) {
        self.capture.stop(); // joins the fake's emit thread → all frames sent
        let _ = self.frame_tx.send(None); // end-of-stream sentinel
        let _ = self.worker.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::*;
    use crate::types::AudioSource;
    use std::sync::mpsc;

    #[test]
    fn session_fuses_tokens_and_speakers_into_segments() {
        let frames: Vec<AudioFrame> = (0..4)
            .map(|i| AudioFrame {
                pcm: vec![0.0; 8000],
                t_start: i as f64 * 0.5,
                t_end: (i + 1) as f64 * 0.5,
                source: AudioSource::Microphone,
            })
            .collect();
        let tokens = vec![
            AsrToken { text: "hello".into(), t_start: 0.10, t_end: 0.40, confidence: 0.9 },
            AsrToken { text: "there".into(), t_start: 0.45, t_end: 0.80, confidence: 0.9 },
            AsrToken { text: "hi".into(), t_start: 1.20, t_end: 1.50, confidence: 0.7 },
        ];
        let (tx, rx) = mpsc::channel();
        let session = Session::run(
            Box::new(FakeCapture::from_script(frames)),
            Box::new(FakeTranscriber::with_tokens(tokens)),
            Box::new(FakeDiarizer::alternating(1.0)), // speaker flips at t=1.0
            Some("en".into()),
            Box::new(move |e| tx.send(e).unwrap()),
        )
        .unwrap();
        session.stop();

        let events: Vec<SessionEvent> = rx.try_iter().collect();
        let segments: Vec<&SegmentDto> = events
            .iter()
            .filter_map(|e| match e {
                SessionEvent::Segment(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(segments.len(), 2, "speaker change at t=1.0 splits segments");
        assert_eq!(segments[0].text, "hello there");
        assert_eq!(segments[0].speaker_label, "S1");
        assert_eq!(segments[1].text, "hi");
        assert_eq!(segments[1].speaker_label, "S2");
        assert!(segments[0].is_final && segments[1].is_final);
        assert_eq!(segments[0].language.as_deref(), Some("en"));
        // Final status is "stopped"
        assert!(matches!(events.last(), Some(SessionEvent::Status(s)) if s == "stopped"));
    }
}
