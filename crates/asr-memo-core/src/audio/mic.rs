//! Microphone capture via cpal. The cpal callback resamples the device's native
//! rate to 16 kHz mono and forwards AudioFrames. Hardware-gated (#[ignore]).
//!
//! `cpal::Stream` is `!Send` on every platform (cpal attaches a
//! `NotSendSyncAcrossAllPlatforms` marker — see cpal-0.15.3 `src/platform/mod.rs`),
//! but `AudioCapture: Send`. To satisfy both, the stream is built and owned on a
//! dedicated worker thread; `start` blocks until the stream is playing, and
//! `stop` signals the worker (whose scope exit drops the stream, halting capture).

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};

use crate::audio::resample::PersistentResampler;
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const OUT_RATE: u32 = 16000;

/// The boxed `on_frame` callback shared with the cpal data callback. Wrapped in
/// `Arc<Mutex<...>>` so the `FnMut` can be invoked from the audio thread.
type FrameSink = Arc<Mutex<Box<dyn FnMut(AudioFrame) + Send>>>;

pub struct CpalMicrophoneCapture {
    // `JoinHandle` + `Sender` are `Send`; the `!Send` `cpal::Stream` lives only
    // on the worker thread, so `CpalMicrophoneCapture: Send` holds.
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CpalMicrophoneCapture {
    pub fn new() -> Self {
        Self {
            stop_tx: None,
            handle: None,
        }
    }
}

impl Default for CpalMicrophoneCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioCapture for CpalMicrophoneCapture {
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        // Build + play the stream on a worker so the `!Send` cpal::Stream never
        // crosses a thread boundary (it is created, held, and dropped on the
        // worker; only `JoinHandle` + `Sender` escape).
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), CoreError>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let frames_tx: FrameSink = Arc::new(Mutex::new(on_frame));

        let handle = std::thread::Builder::new()
            .name("cpal-mic".into())
            .spawn(move || {
                // Build the stream inside the worker so the `!Send` Stream never
                // has to cross thread boundaries.
                let stream = match build_and_play(frames_tx) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // Stream is live; release the caller, then park until `stop`.
                let _ = ready_tx.send(Ok(()));
                let _ = stop_rx.recv();
                // Scope exit drops `stream` → cpal's Drop stops capture on
                // macOS Core Audio. (`_stream` bound here ensures ordering.)
                drop(stream);
            })
            .map_err(|e| CoreError::Capture {
                code: "mic.thread".into(),
                message: e.to_string(),
            })?;

        // Wait for the worker to report build success/failure. On failure the
        // worker thread has already exited; join to clean up the handle.
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                let _ = handle.join();
                return Err(CoreError::Capture {
                    code: "mic.thread".into(),
                    message: "stream worker exited before signalling readiness".into(),
                });
            }
        }

        self.stop_tx = Some(stop_tx);
        self.handle = Some(handle);
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Build the cpal input stream on the worker thread and start it playing. The
/// returned `cpal::Stream` is held by the caller; dropping it stops capture.
/// Errors here surface through `ready_tx` back to `start`.
fn build_and_play(frames_tx: FrameSink) -> Result<cpal::Stream, CoreError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| CoreError::Capture {
            code: "mic.no_device".into(),
            message: "no default input device".into(),
        })?;
    let config = device
        .default_input_config()
        .map_err(|e| CoreError::Capture {
            code: "mic.config".into(),
            message: e.to_string(),
        })?;
    let in_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let fmt = config.sample_format();

    let resampler = Arc::new(Mutex::new(PersistentResampler::new(in_rate, OUT_RATE, 512)));
    let mut emitted: usize = 0;
    let err_cb = |e: cpal::StreamError| eprintln!("mic stream error: {e}");

    let stream = match fmt {
        SampleFormat::F32 => device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _| on_pcm(data, channels, &resampler, &frames_tx, &mut emitted),
                err_cb,
                None,
            )
            .map_err(|e| CoreError::Capture {
                code: "mic.build".into(),
                message: e.to_string(),
            })?,
        SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[i16], _| on_pcm(data, channels, &resampler, &frames_tx, &mut emitted),
                err_cb,
                None,
            )
            .map_err(|e| CoreError::Capture {
                code: "mic.build".into(),
                message: e.to_string(),
            })?,
        other => {
            return Err(CoreError::Capture {
                code: "mic.format".into(),
                message: format!("unsupported sample format: {other}"),
            })
        }
    };
    stream.play().map_err(|e| CoreError::Capture {
        code: "mic.play".into(),
        message: e.to_string(),
    })?;
    Ok(stream)
}

/// Downmix interleaved PCM to mono f32, resample device-rate → 16 kHz, and
/// forward one `AudioFrame` per resampled chunk. `emitted` advances the session
/// clock so each frame's timestamps are contiguous across callbacks.
fn on_pcm<T: Sample + Into<f32>>(
    data: &[T],
    channels: usize,
    resampler: &Arc<Mutex<PersistentResampler>>,
    frames_tx: &FrameSink,
    emitted: &mut usize,
) {
    let ch = channels.max(1);
    // Downmix to mono f32 (average interleaved samples across channels).
    let mono: Vec<f32> = data
        .chunks_exact(ch)
        .map(|frame| frame.iter().map(|s| (*s).into()).sum::<f32>() / ch as f32)
        .collect();
    let resampled = resampler.lock().unwrap().process(&mono);
    if resampled.is_empty() {
        return;
    }
    let t_start = *emitted as f64 / OUT_RATE as f64;
    let t_end = (*emitted + resampled.len()) as f64 / OUT_RATE as f64;
    *emitted += resampled.len();
    let frame = AudioFrame {
        pcm: resampled,
        t_start,
        t_end,
        source: AudioSource::Microphone,
    };
    if let Ok(mut cb) = frames_tx.lock() {
        cb(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires a real microphone + mic permission
    fn captures_two_seconds_of_mic_audio() {
        let mut cap = CpalMicrophoneCapture::new();
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        cap.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(2));
        cap.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(!frames.is_empty());
        let samples: usize = frames.iter().map(|f| f.pcm.len()).sum();
        assert!(
            samples >= 16000,
            "expected >=1s of 16kHz audio, got {samples} samples"
        );
    }
}
