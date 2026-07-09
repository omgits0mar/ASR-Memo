//! One persistent `rubato` resampler per stream. Never recreate per chunk —
//! meetily found that amplifies energy ~173% and yields wrong output sizes.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub struct PersistentResampler {
    from_rate: u32,
    to_rate: u32,
    chunk: usize,
    resampler: SincFixedIn<f32>,
    buffer: Vec<f32>,
}

impl PersistentResampler {
    pub fn new(from_rate: u32, to_rate: u32, chunk: usize) -> Self {
        Self {
            from_rate,
            to_rate,
            chunk,
            resampler: build_resampler(from_rate, to_rate, chunk),
            buffer: Vec::with_capacity(chunk * 2),
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return input.to_vec();
        }
        self.buffer.extend_from_slice(input);
        let mut out = Vec::new();
        while self.buffer.len() >= self.chunk {
            let frame: Vec<f32> = self.buffer.drain(..self.chunk).collect();
            let waves = vec![frame];
            if let Ok(mut resampled) = self.resampler.process(&waves, None) {
                if let Some(ch) = resampled.get_mut(0) {
                    out.append(ch);
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return std::mem::take(&mut self.buffer);
        }
        let remainder = std::mem::take(&mut self.buffer);
        if remainder.is_empty() {
            return Vec::new();
        }
        // Pad the final partial chunk with zeros to flush the resampler.
        let mut padded = remainder;
        padded.resize(self.chunk, 0.0);
        let waves = vec![padded];
        match self.resampler.process(&waves, None) {
            Ok(mut resampled) => resampled.pop().unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    pub fn set_rates(&mut self, from_rate: u32, to_rate: u32) {
        self.from_rate = from_rate;
        self.to_rate = to_rate;
        self.buffer.clear();
        self.resampler = build_resampler(from_rate, to_rate, self.chunk);
    }
}

fn build_resampler(from_rate: u32, to_rate: u32, chunk: usize) -> SincFixedIn<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    SincFixedIn::<f32>::new(to_rate as f64 / from_rate as f64, 2.0, params, chunk, 1).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f32::consts::PI;

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / x.len() as f64).sqrt()
    }

    #[test]
    fn decimates_48k_to_16k_at_correct_ratio() {
        // 480 samples @48k = 10ms → 160 samples @16k.
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let input = vec![0.5f32; 480 * 4]; // 40ms
        let mut out = Vec::new();
        out.extend(r.process(&input));
        out.extend(r.flush());
        let ratio = out.len() as f64 / input.len() as f64;
        assert!((ratio - 1.0 / 3.0).abs() < 0.02, "ratio={ratio}");
    }

    #[test]
    fn preserves_energy_across_variable_chunks() {
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let signal: Vec<f32> = (0..48000)
            .map(|i| (i as f32 * 0.02 * 2.0 * PI).sin() * 0.8)
            .collect();
        let mut out = Vec::new();
        // Feed in awkward chunk sizes that don't divide chunk=512.
        for chunk in [320usize, 512, 1024, 700, 256] {
            let mut i = 0;
            while i < signal.len() {
                let end = (i + chunk).min(signal.len());
                out.extend(r.process(&signal[i..end]));
                i = end;
            }
        }
        out.extend(r.flush());
        let in_rms = rms(&signal);
        let out_rms = rms(&out);
        // Resampling preserves amplitude within a few %.
        assert!(
            (out_rms - in_rms).abs() / in_rms < 0.05,
            "in={in_rms:.4} out={out_rms:.4}"
        );
    }

    #[test]
    fn set_rates_reconfigures_for_rate_change() {
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let _ = r.process(&vec![0.3f32; 1024]);
        let _ = r.flush();
        // Bluetooth switches output to 44100.
        r.set_rates(44100, 16000);
        let out = r.process(&vec![0.3f32; 44100 / 10]); // 100ms @44100
        assert!(!out.is_empty());
    }
}
