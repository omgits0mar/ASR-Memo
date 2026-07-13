//! Per-source RMS/peak + sustained-silence detection (the permission-denied
//! signature: a macOS process tap returns all-zeros when permission is missing).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceHealth {
    pub rms: f32,
    pub peak: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthFlag {
    SystemSilent,
    RateChanged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HealthSnapshot {
    pub mic: SourceHealth,
    pub system: SourceHealth,
    pub flags: Vec<HealthFlag>,
}

pub struct HealthMonitor {
    silence_threshold: f32,
    silence_window_frames: usize,
    consecutive_silent: usize,
    rate_changed_pending: bool,
}

impl HealthMonitor {
    pub fn new(silence_threshold: f32, silence_window_frames: usize) -> Self {
        Self {
            silence_threshold,
            silence_window_frames,
            consecutive_silent: 0,
            rate_changed_pending: false,
        }
    }

    /// Mark that the device sample rate changed; the next snapshot reports
    /// `RateChanged` once and clears the pending flag.
    pub fn notify_rate_changed(&mut self) {
        self.rate_changed_pending = true;
    }

    pub fn update(
        &mut self,
        mic_window: &[f32],
        system_window: &[f32],
        system_active: bool,
    ) -> HealthSnapshot {
        let mic = source_health(mic_window);
        let system = source_health(system_window);
        let mut flags = Vec::new();

        if system_active && system.rms <= self.silence_threshold {
            self.consecutive_silent += 1;
            if self.consecutive_silent >= self.silence_window_frames {
                flags.push(HealthFlag::SystemSilent);
            }
        } else {
            self.consecutive_silent = 0;
        }
        if self.rate_changed_pending {
            flags.push(HealthFlag::RateChanged);
            self.rate_changed_pending = false;
        }
        HealthSnapshot { mic, system, flags }
    }
}

fn source_health(window: &[f32]) -> SourceHealth {
    if window.is_empty() {
        return SourceHealth {
            rms: 0.0,
            peak: 0.0,
        };
    }
    let sum_sq: f64 = window.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / window.len() as f64).sqrt() as f32;
    let peak = window.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    SourceHealth { rms, peak }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_rms_and_peak() {
        let mut m = HealthMonitor::new(0.001, 5);
        let snap = m.update(&[0.5, -0.5, 0.25], &[0.0, 0.0, 0.0], true);
        // RMS = sqrt((0.25 + 0.25 + 0.0625) / 3) = sqrt(0.1875) ≈ 0.433
        assert!((snap.mic.rms - 0.433).abs() < 0.01);
        assert_eq!(snap.mic.peak, 0.5);
    }

    #[test]
    fn flags_system_silent_only_after_sustained_zeros() {
        let mut m = HealthMonitor::new(0.001, 3);
        for _ in 0..2 {
            let s = m.update(&[0.4; 64], &[0.0; 64], true);
            assert!(!s.flags.contains(&HealthFlag::SystemSilent));
        }
        let s = m.update(&[0.4; 64], &[0.0; 64], true); // 3rd silent window
        assert!(s.flags.contains(&HealthFlag::SystemSilent));
    }

    #[test]
    fn non_silent_system_resets_silence_counter() {
        let mut m = HealthMonitor::new(0.001, 3);
        for _ in 0..2 {
            m.update(&[0.4; 64], &[0.0; 64], true);
        }
        m.update(&[0.4; 64], &[0.1; 64], true); // real signal resets
        let s = m.update(&[0.4; 64], &[0.0; 64], true);
        assert!(!s.flags.contains(&HealthFlag::SystemSilent)); // counter restarted
    }

    #[test]
    fn system_inactive_never_flags_silent() {
        let mut m = HealthMonitor::new(0.001, 2);
        for _ in 0..5 {
            let s = m.update(&[0.4; 64], &[0.0; 64], false);
            assert!(!s.flags.contains(&HealthFlag::SystemSilent));
        }
    }
}
