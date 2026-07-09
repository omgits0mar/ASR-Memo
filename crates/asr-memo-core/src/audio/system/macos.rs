//! macOS system-audio capture via a cidre Core Audio global mono process tap,
//! wrapped in a tap-only private aggregate device (never tap + output device —
//! that caused meetily's echo). Reads the device sample rate each cycle so a
//! Bluetooth profile switch reconfigures the resampler (the rate-tracking fix).
//! Faithful port of meetily's audio/capture/core_audio.rs (MIT).

#![cfg(target_os = "macos")]

use std::sync::{Arc, Mutex};

use cidre::{cat, cf, core_audio as ca, ns, os};

use crate::audio::resample::PersistentResampler;
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const OUT_RATE: u32 = 16000;

/// The boxed `on_frame` callback shared with the Core Audio IOProc. Wrapped in
/// `Arc<Mutex<...>>` so the `FnMut` can be invoked from the audio thread.
type FrameSink = Arc<Mutex<Box<dyn FnMut(AudioFrame) + Send>>>;

/// macOS system-audio capture: a cidre global mono process tap wrapped in a
/// tap-only private aggregate device. Implements `AudioCapture`.
///
/// This is a unit struct on purpose: the cidre objects (aggregate device, tap,
/// IOProc ID, started device) are created in `start` and intentionally leaked
/// for the session lifetime (the IOProc runs on Core Audio's own thread, so no
/// worker thread is needed). The leak is a recorded Phase 2a follow-up. With no
/// cidre fields, `CoreAudioTapCapture` is trivially `Send` (unlike cpal's
/// `Stream`), so no worker thread dance is required (cf. `mic.rs`).
pub struct CoreAudioTapCapture;

impl CoreAudioTapCapture {
    pub fn new() -> Self {
        CoreAudioTapCapture
    }
}

impl Default for CoreAudioTapCapture {
    fn default() -> Self {
        Self::new()
    }
}

// State shared with the Core Audio IOProc callback (runs on Core Audio's
// thread). Leaked via `Box::into_raw` for the session lifetime.
struct TapCtx {
    resampler: Mutex<PersistentResampler>,
    last_rate: Mutex<u32>,
    on_frame: FrameSink,
    emitted: Mutex<usize>,
}

impl AudioCapture for CoreAudioTapCapture {
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        // 1) Default output device + its UID (the aggregate's main sub-device;
        //    not a second sub-device — that caused meetily's echo).
        let output = ca::System::default_output_device().map_err(|e| CoreError::Capture {
            code: "tap.output".into(),
            message: format!("{e:?}"),
        })?;
        let output_uid = output.uid().map_err(|e| CoreError::Capture {
            code: "tap.uid".into(),
            message: format!("{e:?}"),
        })?;

        // 2) Global mono process tap, excluding no processes.
        let tap_desc =
            ca::TapDesc::with_mono_global_tap_excluding_processes(&ns::Array::<ns::Number>::new());
        let tap = tap_desc
            .create_process_tap()
            .map_err(|e| CoreError::Capture {
                code: "tap.create".into(),
                message: format!("{e:?}"),
            })?;
        let tap_asbd = tap.asbd().map_err(|e| CoreError::Capture {
            code: "tap.asbd".into(),
            message: format!("{e:?}"),
        })?;
        let in_rate = tap_asbd.sample_rate as u32;

        // 3) Tap-only aggregate: main_sub_device = output UID, tap_list = [tap].
        //    Do NOT also list the output device as a second sub_device — that
        //    duplicated audio and caused the echo meetily fixed.
        let sub_tap = cf::DictionaryOf::with_keys_values(
            &[ca::sub_device_keys::uid()],
            &[tap
                .uid()
                .unwrap_or_else(|_| cf::Uuid::new().to_cf_string())
                .as_type_ref()],
        );
        let agg_desc = cf::DictionaryOf::with_keys_values(
            &[
                ca::aggregate_device_keys::is_private(),
                ca::aggregate_device_keys::is_stacked(),
                ca::aggregate_device_keys::tap_auto_start(),
                ca::aggregate_device_keys::name(),
                ca::aggregate_device_keys::main_sub_device(),
                ca::aggregate_device_keys::uid(),
                ca::aggregate_device_keys::tap_list(),
            ],
            &[
                cf::Boolean::value_true().as_type_ref(),
                cf::Boolean::value_false(),
                cf::Boolean::value_true(),
                cf::str!(c"asr-memo-audio-tap").as_type_ref(),
                &output_uid,
                &cf::Uuid::new().to_cf_string(),
                &cf::ArrayOf::from_slice(&[sub_tap.as_ref()]),
            ],
        );
        let agg = ca::AggregateDevice::with_desc(&agg_desc).map_err(|e| CoreError::Capture {
            code: "tap.aggregate".into(),
            message: format!("{e:?}"),
        })?;

        // 4) Shared context for the IOProc. Leaked via `Box::into_raw` for the
        //    session lifetime; the pointer is also passed as client_data.
        let ctx = Box::new(TapCtx {
            resampler: Mutex::new(PersistentResampler::new(in_rate, OUT_RATE, 512)),
            last_rate: Mutex::new(in_rate),
            on_frame: Arc::new(Mutex::new(on_frame)),
            emitted: Mutex::new(0),
        });
        let ctx_ptr = Box::into_raw(ctx);

        // The IOProc: rate-track → resample → forward an AudioFrame with
        // contiguous session timestamps. No downmix is needed (tap is mono
        // global) but the slice path handles it generically.
        extern "C" fn io_proc(
            device: ca::Device,
            _now: &cat::AudioTimeStamp,
            in_data: &cat::AudioBufList<1>,
            _in_time: &cat::AudioTimeStamp,
            _out_data: &mut cat::AudioBufList<1>,
            _out_time: &cat::AudioTimeStamp,
            ctx: Option<&mut TapCtx>,
        ) -> os::Status {
            let Some(ctx) = ctx else {
                return os::Status::NO_ERR;
            };
            // Rate tracking: reconfigure the resampler if the device rate
            // changed (e.g. a Bluetooth headset switching profile mid-call).
            if let Ok(rate) = device.nominal_sample_rate() {
                let rate = rate as u32;
                let mut last = ctx.last_rate.lock().unwrap();
                if *last != rate {
                    ctx.resampler.lock().unwrap().set_rates(rate, OUT_RATE);
                    *last = rate;
                }
            }
            // Read interleaved f32 from the first buffer (tap is mono global).
            let buf = &in_data.buffers[0];
            let n = buf.data_bytes_size as usize / std::mem::size_of::<f32>();
            if n == 0 || buf.data.is_null() {
                return os::Status::NO_ERR;
            }
            // SAFETY: `buf.data` points to `n` f32 samples owned by Core Audio
            // for the duration of the IOProc. Cast to a borrow slice; no drop.
            let samples = unsafe { std::slice::from_raw_parts(buf.data as *const f32, n) };
            let resampled = ctx.resampler.lock().unwrap().process(samples);
            if resampled.is_empty() {
                return os::Status::NO_ERR;
            }
            let mut emitted = ctx.emitted.lock().unwrap();
            let t_start = *emitted as f64 / OUT_RATE as f64;
            let t_end = (*emitted + resampled.len()) as f64 / OUT_RATE as f64;
            *emitted += resampled.len();
            drop(emitted);
            let frame = AudioFrame {
                pcm: resampled,
                t_start,
                t_end,
                source: AudioSource::System,
            };
            if let Ok(mut cb) = ctx.on_frame.lock() {
                cb(frame);
            }
            os::Status::NO_ERR
        }

        // `DeviceIoProcId` is a `Copy` fn-pointer type, so `Some(proc_id)`
        // below does not move proc_id out — we can still forget it afterwards
        // to suppress the StartedDevice's eventual Drop.
        let proc_id = agg
            .create_io_proc_id(io_proc, Some(unsafe { &mut *ctx_ptr }))
            .map_err(|e| CoreError::Capture {
                code: "tap.ioproc".into(),
                message: format!("{e:?}"),
            })?;
        // `agg` is moved into `device_start` and re-wrapped (under
        // ManuallyDrop) inside the returned `StartedDevice`. Forgetting
        // `started` therefore keeps both the aggregate device AND `proc_id`
        // alive — no separate forget of `agg` is possible (it has moved).
        let started = ca::device_start(agg, Some(proc_id)).map_err(|e| CoreError::Capture {
            code: "tap.start".into(),
            message: format!("{e:?}"),
        })?;

        // 5) Intentional session-scoped leak. `started` holds the AggregateDevice
        //    (under ManuallyDrop) + proc_id; forgetting it prevents Drop from
        //    running, which would otherwise call AudioDeviceStop and
        //    AudioHardwareDestroyAggregateDevice. `tap` (TapGuard) is forgotten
        //    so its Drop doesn't destroy the process tap. The TapCtx pointer
        //    (`ctx_ptr`) is leaked via `Box::into_raw`; reclaiming it is a
        //    follow-up. Capture runs for the process lifetime — the recorder's
        //    close() has already flushed audio to disk before `stop()` is hit.
        let _ = ctx_ptr;
        std::mem::forget(started);
        std::mem::forget(tap);
        Ok(())
    }

    fn stop(&mut self) {
        // Phase 2a: capture ends with the process. Full teardown (remove IO
        // proc, destroy aggregate, reclaim ctx via Box::from_raw) is a recorded
        // follow-up; recording is already flushed by the recorder's close().
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    #[ignore] // requires macOS 14.4+, audio-capture permission, and audio playing
    fn captures_two_seconds_of_system_audio() {
        let mut cap = CoreAudioTapCapture::new();
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        cap.start(Box::new(move |f| {
            let _ = tx.send(f);
        }))
        .unwrap();
        std::thread::sleep(std::time::Duration::from_secs(2));
        cap.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(!frames.is_empty());
        let samples: usize = frames.iter().map(|f| f.pcm.len()).sum();
        assert!(samples >= 16000);
    }
}
