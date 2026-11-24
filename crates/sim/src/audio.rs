//! Sound in and out as WAV files: 16-bit PCM, the format a microphone's driver or a
//! speaker's would hand over. Only what voice needs: mono, any sample rate.

use std::path::Path;

/// A clip's samples in [−1, 1] and its rate.
#[derive(Clone, Debug)]
pub struct Clip {
    pub rate: u32,
    pub samples: Vec<f32>,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

impl Clip {
    /// A 16-bit PCM WAV; a stereo one is mixed down to mono.
    pub fn read(path: &Path) -> Result<Self, String> {
        Self::parse(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("not a WAV file".into());
        }
        let (mut at, mut format, mut data) = (12, None, None);
        while at + 8 <= bytes.len() {
            let (id, size) = (&bytes[at..at + 4], u32_at(bytes, at + 4) as usize);
            let body = at + 8..(at + 8 + size).min(bytes.len());
            match id {
                b"fmt " if size >= 16 => format = Some((u16_at(bytes, at + 8), u16_at(bytes, at + 10), u32_at(bytes, at + 12), u16_at(bytes, at + 22))),
                b"data" => data = Some(body.clone()),
                _ => {}
            }
            at = body.end + size % 2;
        }
        let (kind, channels, rate, bits) = format.ok_or("the WAV has no format")?;
        if kind != 1 || bits != 16 || channels == 0 {
            return Err("only 16-bit PCM WAV files are read".into());
        }
        let data = &bytes[data.ok_or("the WAV has no data")?];
        let channels = channels as usize;
        let samples = data
            .chunks_exact(2 * channels)
            .map(|frame| frame.as_chunks::<2>().0.iter().map(|s| i16::from_le_bytes(*s) as f32 / 32768.0).sum::<f32>() / channels as f32)
            .collect();
        Ok(Self { rate, samples })
    }

    /// As a mono 16-bit PCM WAV.
    pub fn to_wav(&self) -> Vec<u8> {
        let data = self.samples.len() as u32 * 2;
        let mut out = Vec::with_capacity(44 + data as usize);
        out.extend(b"RIFF");
        out.extend((36 + data).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(self.rate.to_le_bytes());
        out.extend((self.rate * 2).to_le_bytes());
        out.extend(2u16.to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend(data.to_le_bytes());
        for s in &self.samples {
            out.extend(((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        }
        out
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_wav()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// At another rate, by linear interpolation: enough to bring speech to a model's rate.
    pub fn resampled(&self, rate: u32) -> Self {
        if rate == self.rate || self.samples.is_empty() {
            return Self { rate, samples: self.samples.clone() };
        }
        let count = (self.samples.len() as u64 * rate as u64 / self.rate as u64) as usize;
        let step = self.rate as f64 / rate as f64;
        let samples = (0..count)
            .map(|i| {
                let at = i as f64 * step;
                let (left, part) = (at.floor() as usize, (at - at.floor()) as f32);
                let next = self.samples.get(left + 1).copied().unwrap_or(self.samples[left]);
                self.samples[left] * (1.0 - part) + next * part
            })
            .collect();
        Self { rate, samples }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_goes_out_and_back_as_wav() {
        let clip = Clip { rate: 16_000, samples: (0..100).map(|i| (i as f32 / 10.0).sin() * 0.5).collect() };
        let back = Clip::parse(&clip.to_wav()).unwrap();
        assert_eq!(back.rate, 16_000);
        assert!(clip.samples.iter().zip(&back.samples).all(|(a, b)| (a - b).abs() < 1e-4));
        assert_eq!(clip.resampled(8_000).samples.len(), 50);
    }
}
