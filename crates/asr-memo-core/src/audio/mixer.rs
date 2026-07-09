//! Aligns mic + system 16 kHz streams in 50 ms windows, applies RMS-based
//! dynamic mixing (ratios 10–90%) with soft-scaling, emits mixed frames +
//! health snapshots. Tested with fake sub-captures (no cpal/cidre).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use crate::audio::health::{HealthMonitor, HealthSnapshot};
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const RATE: u32 = 16000;
const WINDOW_SECS: f64 = 0.05;
const WINDOW_SAMPLES: usize = (RATE as f64 * WINDOW_SECS) as usize; // 800
const MIN_RATIO: f32 = 0.1;
const MAX_RATIO: f32 = 0.9;

pub struct MixedCapture {
    mic: Option<Box<dyn AudioCapture>>,
    system: Option<Box<dyn AudioCapture>>,
    // Stored as Option so `start` can move the closure into the mixer thread.
    on_health: Option<Box<dyn Fn(HealthSnapshot) + Send>>,
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl MixedCapture {
    pub fn new(
        mic: Option<Box<dyn AudioCapture>>,
        system: Option<Box<dyn AudioCapture>>,
        on_health: Box<dyn Fn(HealthSnapshot) + Send>,
    ) -> Self {
        Self {
            mic,
            system,
            on_health: Some(on_health),
            stop: Arc::new(AtomicBool::new(false)),
            handles: Vec::new(),
        }
    }

    /// RMS energy of a window (f64 accumulator to avoid f32 drift on 800 samples).
    fn rms(window: &[f32]) -> f32 {
        if window.is_empty() {
            return 0.0;
        }
        let s: f64 = window.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (s / window.len() as f64).sqrt() as f32
    }

    /// Sum two windows with RMS-derived dynamic ratios + soft-scaling.
    fn mix_window(mic_w: &[f32], sys_w: &[f32]) -> Vec<f32> {
        let mic_r = Self::rms(mic_w);
        let sys_r = Self::rms(sys_w);
        let (mr, sr) = dynamic_ratios(mic_r, sys_r);
        let max_len = mic_w.len().max(sys_w.len());
        let mut mixed = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let m = if i < mic_w.len() { mic_w[i] * mr } else { 0.0 };
            let s = if i < sys_w.len() { sys_w[i] * sr } else { 0.0 };
            mixed.push(m + s);
        }
        soft_scale(&mut mixed);
        mixed
    }
}

/// Per-window mix weights from RMS ratio, clamped to [MIN_RATIO, MAX_RATIO] so a
/// near-silent source is never fully zeroed (keeps a floor for speech onsets).
fn dynamic_ratios(mic_r: f32, sys_r: f32) -> (f32, f32) {
    if mic_r == 0.0 && sys_r == 0.0 {
        return (0.5, 0.5);
    }
    if mic_r == 0.0 {
        return (0.0, 1.0);
    }
    if sys_r == 0.0 {
        return (1.0, 0.0);
    }
    let total = mic_r + sys_r;
    let mr = (mic_r / total).clamp(MIN_RATIO, MAX_RATIO);
    (mr, 1.0 - mr)
}

/// Scale-down only when peak > 1.0 (never amplifies; keeps |x| ≤ 1).
fn soft_scale(mixed: &mut [f32]) {
    let peak = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        let g = 1.0 / peak;
        for s in mixed.iter_mut() {
            *s *= g;
        }
    }
}

impl AudioCapture for MixedCapture {
    fn start(&mut self, mut on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        let stop = self.stop.clone();
        let (mic_tx, mic_rx) = mpsc::channel::<AudioFrame>();
        let (sys_tx, sys_rx) = mpsc::channel::<AudioFrame>();
        let system_active = self.system.is_some();

        if let Some(mut mic) = self.mic.take() {
            let tx = mic_tx.clone();
            mic.start(Box::new(move |f| {
                let _ = tx.send(f);
            }))?;
            // Phase-2a: sub-capture threads outlive `stop` (full teardown is a
            // follow-up; AudioCapture::stop can't return the capture to reclaim).
            std::mem::forget(mic);
        }
        if let Some(mut sys) = self.system.take() {
            let tx = sys_tx.clone();
            sys.start(Box::new(move |f| {
                let _ = tx.send(f);
            }))?;
            std::mem::forget(sys);
        }

        // Move the health callback into the mixer thread. `start` consumes it.
        let on_health = self
            .on_health
            .take()
            .expect("start called twice; on_health already moved into mixer thread");

        self.handles.push(std::thread::spawn(move || {
            let mut health = HealthMonitor::new(0.001, 5);
            let mut mic_buf: Vec<f32> = Vec::new();
            let mut sys_buf: Vec<f32> = Vec::new();
            let mut t = 0.0f64;
            loop {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                // Drain newly arrived frames into the per-source buffers.
                while let Ok(f) = mic_rx.try_recv() {
                    mic_buf.extend(f.pcm);
                }
                while let Ok(f) = sys_rx.try_recv() {
                    sys_buf.extend(f.pcm);
                }
                if mic_buf.len() >= WINDOW_SAMPLES {
                    let mic_w: Vec<f32> = mic_buf.drain(..WINDOW_SAMPLES).collect();
                    let sys_w: Vec<f32> = if sys_buf.len() >= WINDOW_SAMPLES {
                        sys_buf.drain(..WINDOW_SAMPLES).collect()
                    } else {
                        // System lagging this window: pad with zeros, leave real
                        // samples buffered for the next aligned window.
                        vec![0.0; WINDOW_SAMPLES.min(sys_buf.len())]
                    };
                    let snap = health.update(&mic_w, &sys_w, system_active);
                    on_health(snap);
                    let dominant = if Self::rms(&mic_w) >= Self::rms(&sys_w) {
                        AudioSource::Microphone
                    } else {
                        AudioSource::System
                    };
                    let pcm = Self::mix_window(&mic_w, &sys_w);
                    on_frame(AudioFrame {
                        pcm,
                        t_start: t,
                        t_end: t + WINDOW_SECS,
                        source: dominant,
                    });
                    t += WINDOW_SECS;
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }));
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::health::HealthFlag;
    use crate::traits::{AudioCapture, AudioFrame};
    use crate::types::AudioSource;
    use std::sync::mpsc;

    /// A fake capture that emits a scripted sequence of 50 ms frames then ends.
    struct ScriptCapture {
        frames: Vec<AudioFrame>,
    }
    impl AudioCapture for ScriptCapture {
        fn start(
            &mut self,
            mut on_frame: Box<dyn FnMut(AudioFrame) + Send>,
        ) -> Result<(), CoreError> {
            let frames: Vec<AudioFrame> = self.frames.drain(..).collect();
            std::thread::spawn(move || {
                for f in frames {
                    on_frame(f);
                }
            });
            Ok(())
        }
        fn stop(&mut self) {}
    }

    fn win(t0: f64, level: f32, src: AudioSource) -> AudioFrame {
        AudioFrame {
            pcm: vec![level; WINDOW_SAMPLES],
            t_start: t0,
            t_end: t0 + WINDOW_SECS,
            source: src,
        }
    }

    #[test]
    fn emits_mixed_frames_aligned_to_50ms_and_reports_health() {
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![
                win(0.0, 0.4, AudioSource::Microphone),
                win(0.05, 0.4, AudioSource::Microphone),
            ],
        });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![
                win(0.0, 0.2, AudioSource::System),
                win(0.05, 0.0, AudioSource::System),
            ],
        });
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        let (htx, hrx) = mpsc::channel::<HealthSnapshot>();
        let mut mix = MixedCapture::new(
            Some(mic),
            Some(sys),
            Box::new(move |h| htx.send(h).unwrap()),
        );
        mix.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        // give the mixer thread time to drain
        std::thread::sleep(std::time::Duration::from_millis(200));
        mix.stop();

        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(
            frames.len() >= 2,
            "expected >=2 mixed windows, got {}",
            frames.len()
        );
        for f in &frames {
            assert_eq!(f.pcm.len(), WINDOW_SAMPLES);
            assert!((f.t_end - f.t_start - WINDOW_SECS).abs() < 1e-6);
        }
        // Mic louder than system on window 0 → dominant source is Microphone.
        assert_eq!(frames[0].source, AudioSource::Microphone);
        // Health snapshots were delivered.
        let healths: Vec<HealthSnapshot> = hrx.try_iter().collect();
        assert!(!healths.is_empty());
    }

    #[test]
    fn soft_scales_when_sum_exceeds_unity() {
        // Two loud windows summed naively would clip; soft-scale keeps |x|<=1.
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![win(0.0, 0.9, AudioSource::Microphone)],
        });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![win(0.0, 0.9, AudioSource::System)],
        });
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        let mut mix = MixedCapture::new(Some(mic), Some(sys), Box::new(|_| {}));
        mix.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(150));
        mix.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        let max_abs = frames
            .iter()
            .flat_map(|f| f.pcm.iter().copied())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0 + 1e-6, "clipped: {max_abs}");
    }

    #[test]
    fn system_silence_is_flagged() {
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: (0..6)
                .map(|i| win(i as f64 * 0.05, 0.4, AudioSource::Microphone))
                .collect(),
        });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: (0..6)
                .map(|i| win(i as f64 * 0.05, 0.0, AudioSource::System))
                .collect(),
        });
        let (htx, hrx) = mpsc::channel::<HealthSnapshot>();
        let mut mix = MixedCapture::new(
            Some(mic),
            Some(sys),
            Box::new(move |h| htx.send(h).unwrap()),
        );
        mix.start(Box::new(|_| {})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(250));
        mix.stop();
        let healths: Vec<HealthSnapshot> = hrx.try_iter().collect();
        assert!(healths
            .iter()
            .any(|h| h.flags.contains(&HealthFlag::SystemSilent)));
    }
}
