//! Samples as bytes and back: 16-bit pcm and wav written directly, wav read
//! without ffmpeg, any other container decoded, opus, mp3 and flac encoded
//! (encode.rs), resampling by libsoxr at the quality the Python server uses.

pub fn content_type(fmt: &str) -> &'static str {
    match fmt {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "opus" => "audio/ogg",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        _ => "application/octet-stream",
    }
}

/// 16-bit samples; a synthesis that overshoots 1.0 is scaled down, not clipped.
pub fn to_pcm16(x: &[f32]) -> Vec<u8> {
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };
    let mut out = Vec::with_capacity(x.len() * 2);
    for v in x {
        // как numpy astype(int16): отбрасывание дробной части
        let s = ((v * scale).clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

pub fn to_wav(x: &[f32], sr: u32) -> Vec<u8> {
    to_wav_ch(x, sr, 1)
}

/// `x` — отсчёты вперемешку по каналам (L R L R …).
pub fn to_wav_ch(x: &[f32], sr: u32, ch: u16) -> Vec<u8> {
    let pcm = to_pcm16(x);
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2 * ch as u32).to_le_bytes());
    out.extend_from_slice(&(2 * ch).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(&pcm);
    out
}

/// A wav header for a stream of unknown length: sizes at the maximum, which
/// players read as "until the data ends".
pub fn wav_stream_header(sr: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&u32::MAX.to_le_bytes());
    out
}

/// Mono float samples from a 16-bit or float wav — what ffmpeg writes back.
pub fn read_wav(data: &[u8]) -> Option<(Vec<f32>, u32)> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return None;
    }
    let (mut pos, mut fmt) = (12usize, None);
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?) as usize;
        let body = &data[pos + 8..(pos + 8 + size).min(data.len())];
        if id == b"fmt " && body.len() >= 16 {
            let tag = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            let sr = u32::from_le_bytes(body[4..8].try_into().ok()?);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            fmt = Some((tag, channels.max(1), sr, bits));
        } else if id == b"data" {
            let (tag, channels, sr, bits) = fmt?;
            let frames: Vec<f32> = match (tag, bits) {
                (1, 16) | (0xFFFE, 16) => body
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                (3, 32) | (0xFFFE, 32) => body
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => return None,
            };
            let mono = frames.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
            return Some((mono, sr));
        }
        pos += 8 + size + (size & 1);
    }
    None
}

/// Any container a browser, a phone or a recorder makes — wav, mp3, flac,
/// ogg, opus, webm, mp4 — → mono samples at `sr`, without ffmpeg.
pub fn decode(raw: &[u8], sr: u32) -> anyhow::Result<Vec<f32>> {
    crate::native::decode::decode(raw, sr).map_err(|e| anyhow::anyhow!(e))
}

/// Samples → a file in `fmt`: wav, pcm (16-bit, no header), opus, mp3 or
/// flac, by the encoders built in. aac needs ffmpeg and is not here.
pub fn encode(x: &[f32], sr: u32, fmt: &str) -> anyhow::Result<Vec<u8>> {
    encode_ch(x, sr, 1, fmt)
}

/// The same for `ch` channels interleaved (L R L R …): music is stereo.
pub fn encode_ch(x: &[f32], sr: u32, ch: u16, fmt: &str) -> anyhow::Result<Vec<u8>> {
    match fmt {
        "pcm" => Ok(to_pcm16(x)),
        "wav" => Ok(to_wav_ch(x, sr, ch)),
        f if crate::encode::own(f) => crate::encode::encode_ch(x, sr, ch, f),
        f => anyhow::bail!("unsupported format: {f}; wav, pcm, opus, mp3 and flac are"),
    }
}

// ----------------------------------------------------------- пересчёт частоты

/// libsoxr at its default ("HQ") quality — the same as `soxr.resample` and
/// `soxr.ResampleStream` in the Python server.
pub struct Resampler {
    raw: soxr_sys::soxr_t,
    ratio: f64,
}

unsafe impl Send for Resampler {}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        let mut err: soxr_sys::soxr_error_t = std::ptr::null();
        let raw = unsafe {
            soxr_sys::soxr_create(
                from as f64,
                to as f64,
                1,
                &mut err,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert!(err.is_null() && !raw.is_null(), "soxr_create failed");
        Resampler { raw, ratio: to as f64 / from as f64 }
    }

    fn process(&mut self, input: Option<&[f32]>) -> Vec<f32> {
        let len = input.map_or(0, |i| i.len());
        let mut out = vec![0f32; (len as f64 * self.ratio) as usize + 1024];
        let mut produced_total = Vec::new();
        let mut consumed = 0usize;
        loop {
            let mut idone = 0usize;
            let mut odone = 0usize;
            let (ip, ilen) = match input {
                Some(i) => (i[consumed..].as_ptr() as soxr_sys::soxr_in_t, i.len() - consumed),
                None => (std::ptr::null(), 0),
            };
            let err = unsafe {
                soxr_sys::soxr_process(
                    self.raw,
                    ip,
                    ilen,
                    &mut idone,
                    out.as_mut_ptr() as soxr_sys::soxr_out_t,
                    out.len(),
                    &mut odone,
                )
            };
            assert!(err.is_null(), "soxr_process failed");
            consumed += idone;
            produced_total.extend_from_slice(&out[..odone]);
            let more_input = input.is_some() && consumed < len;
            if !more_input && odone < out.len() {
                return produced_total;
            }
        }
    }

    /// A piece of a stream: no clicks at the joins between pieces.
    pub fn chunk(&mut self, x: &[f32]) -> Vec<f32> {
        self.process(Some(x))
    }

    /// Everything at once, flushed.
    pub fn whole(from: u32, to: u32, x: &[f32]) -> Vec<f32> {
        if from == to {
            return x.to_vec();
        }
        let mut r = Resampler::new(from, to);
        let mut y = r.process(Some(x));
        y.extend(r.process(None));
        y
    }
}

impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe { soxr_sys::soxr_delete(self.raw) }
    }
}
