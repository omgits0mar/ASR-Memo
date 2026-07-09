//! Incremental 48 kHz WAV writer. Header written on open; PCM appended per
//! window; lengths patched on close so a crash leaves a valid (shorter) file.

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

pub struct WavWriter {
    file: File,
    data_bytes: u32,
    sample_rate: u32,
}

impl WavWriter {
    pub fn create(path: &Path, sample_rate: u32) -> io::Result<Self> {
        let mut file = File::create(path)?;
        // Open-time header with data length 0; patched on close.
        file.write_all(b"RIFF")?;
        file.write_all(&0u32.to_le_bytes())?; // file size placeholder
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?; // PCM
        file.write_all(&1u16.to_le_bytes())?; // mono
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
        file.write_all(&2u16.to_le_bytes())?; // block align
        file.write_all(&16u16.to_le_bytes())?; // bits per sample
        file.write_all(b"data")?;
        file.write_all(&0u32.to_le_bytes())?; // data length placeholder
        file.sync_all()?;
        Ok(Self {
            file,
            data_bytes: 0,
            sample_rate,
        })
    }

    pub fn write_samples(&mut self, pcm: &[f32]) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for &s in pcm {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_sample = (clamped * 32767.0) as i16;
            bytes.extend_from_slice(&i16_sample.to_le_bytes());
        }
        self.file.write_all(&bytes)?;
        self.data_bytes = self.data_bytes.saturating_add(bytes.len() as u32);
        Ok(())
    }

    pub fn close(mut self) -> io::Result<()> {
        self.file.flush()?;
        // Patch RIFF file size (everything after "RIFF") and data length.
        let riff_size = 36 + self.data_bytes;
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&riff_size.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&self.data_bytes.to_le_bytes())?;
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    #[test]
    fn writes_valid_wav_header_and_patches_lengths_on_close() {
        let tmp = tempfile();
        let path = std::path::Path::new(&tmp);
        {
            let mut w = WavWriter::create(path, 48000).unwrap();
            w.write_samples(&[0.0, 0.5, -0.5, 1.0]).unwrap();
            w.close().unwrap();
        }
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(read_u32(&bytes, 16), 16); // PCM fmt chunk size
        assert_eq!(read_u32(&bytes, 24), 48000); // sample rate
        assert_eq!(read_u32(&bytes, 28), 48000 * 2); // byte rate (mono s16)
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(read_u32(&bytes, 40), 8); // 4 samples * 2 bytes
        assert_eq!(bytes.len(), 44 + 8);
    }

    #[test]
    fn drop_without_close_still_leaves_valid_header() {
        // If the process is killed, the file on disk has the open-time header
        // (data length 0). It's a valid WAV with whatever was flushed by the OS.
        let tmp = tempfile();
        let path = std::path::Path::new(&tmp);
        {
            let mut w = WavWriter::create(path, 48000).unwrap();
            w.write_samples(&[0.25, -0.25]).unwrap();
            // intentionally no close() — flush what we can
            let _ = w.file.sync_all();
        }
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert!(bytes.len() >= 44);
    }

    fn tempfile() -> String {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.join(format!("asr-memo-test-{nanos}.wav"))
            .to_string_lossy()
            .into_owned()
    }
}
