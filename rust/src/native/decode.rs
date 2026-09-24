//! Any container a browser or a phone produces → mono float32 at a given rate.
//!
//! The Python server reads files through PyAV, which carries its own ffmpeg.
//! Here containers are symphonia's (wav, flac, mp3, ogg, mp4, webm) and so are
//! most decoders; opus it has none, so those packets go to libopus — the same
//! library that encodes opus (encode.rs). What is left for the system ffmpeg is
//! whatever symphonia cannot even open.
//!
//! Opus matters more than it looks: that is what a browser records
//! (MediaRecorder gives webm with opus inside), and without it a recording from
//! the console could not be recognised on a machine without ffmpeg.

use std::io::{Cursor, Write};
use std::process::{Command, Stdio};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_OPUS, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::audio::Resampler;

fn with_symphonia(raw: &[u8]) -> Option<(Vec<f32>, u32)> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(raw.to_vec())), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?.clone();
    if track.codec_params.codec == CODEC_TYPE_OPUS {
        return with_libopus(&mut format, &track);
    }
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).ok()?;
    let mut sr = track.codec_params.sample_rate.unwrap_or(0);
    let mut out = vec![];
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else { continue };
        let spec = *decoded.spec();
        sr = spec.rate;
        let channels = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        out.extend(buf.samples().chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
    }
    (sr > 0 && !out.is_empty()).then_some((out, sr))
}

/// Opus packets → mono float32. Opus always decodes at 48 kHz, whatever the
/// original rate was; `pre_skip` из заголовка OpusHead — задержка кодировщика,
/// эти отсчёты проигрыватель обязан отбросить, иначе звук начнётся раньше.
fn with_libopus(format: &mut Box<dyn symphonia::core::formats::FormatReader>,
                track: &symphonia::core::formats::Track) -> Option<(Vec<f32>, u32)> {
    const SR: u32 = 48000;
    let channels = track.codec_params.channels.map_or(1, |c| c.count().max(1));
    let mut err = 0;
    let dec = unsafe { audiopus_sys::opus_decoder_create(SR as i32, channels as i32, &mut err) };
    if dec.is_null() || err != 0 {
        return None;
    }
    let pre_skip = track.codec_params.extra_data.as_deref().filter(|h| h.starts_with(b"OpusHead") && h.len() >= 12)
        .map_or(0, |h| u16::from_le_bytes([h[10], h[11]]) as usize);
    let mut frame = vec![0f32; 5760 * channels]; // 120 мс — самый длинный пакет opus
    let mut out: Vec<f32> = vec![];
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        let n = unsafe {
            audiopus_sys::opus_decode_float(dec, packet.data.as_ptr(), packet.data.len() as i32,
                                            frame.as_mut_ptr(), 5760, 0)
        };
        if n < 0 {
            continue; // битый пакет: пропускаем его, а не весь файл
        }
        let got = n as usize * channels;
        out.extend(frame[..got].chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
    }
    unsafe { audiopus_sys::opus_decoder_destroy(dec) };
    if out.len() <= pre_skip {
        return None;
    }
    Some((out.split_off(pre_skip), SR))
}

fn with_ffmpeg(raw: &[u8], sr: u32) -> Result<Vec<f32>, String> {
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar"])
        .arg(sr.to_string())
        .arg("pipe:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| "cannot decode audio: format not supported without ffmpeg".to_string())?;
    let mut stdin = child.stdin.take().expect("stdin");
    let data = raw.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let out = child.wait_with_output().map_err(|e| format!("cannot decode audio: {e}"))?;
    let _ = writer.join();
    if !out.status.success() || out.stdout.is_empty() {
        return Err(format!("cannot decode audio: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub fn decode(raw: &[u8], sr: u32) -> Result<Vec<f32>, String> {
    if raw.is_empty() {
        return Err("cannot decode audio: empty".into());
    }
    match with_symphonia(raw) {
        Some((x, from)) => Ok(Resampler::whole(from, sr, &x)),
        None => with_ffmpeg(raw, sr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f64, sr: u32) -> Vec<f32> {
        (0..(seconds * sr as f64) as usize)
            .map(|i| (i as f64 * 440.0 * 2.0 * std::f64::consts::PI / sr as f64).sin() as f32 * 0.5)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    /// Свой opus читается своим декодером: длительность и громкость на месте.
    #[test]
    fn opus_round_trip() {
        let x = tone(1.0, 24000);
        let ogg = crate::encode::encode(&x, 24000, "opus").expect("кодирование");
        let y = decode(&ogg, 24000).expect("декодирование");
        let off = (y.len() as f64 - x.len() as f64).abs() / x.len() as f64;
        assert!(off < 0.05, "длительность уехала: {} против {}", y.len(), x.len());
        assert!((rms(&y) / rms(&x) - 1.0).abs() < 0.2, "громкость уехала: {} против {}", rms(&y), rms(&x));
    }

    /// То, что пишет браузер: webm с opus внутри. Без этого запись из консоли
    /// не распознать на машине без ffmpeg. Образец — тон 440 Гц на 0.5 с,
    /// у ffmpeg он декодируется в RMS 0.0883.
    #[test]
    fn webm_opus_decodes() {
        let y = decode(include_bytes!("testdata/tone.webm"), 16000).expect("декодирование webm");
        let seconds = y.len() as f64 / 16000.0;
        assert!((seconds - 0.5).abs() < 0.05, "длительность {seconds:.3} с вместо 0.5");
        assert!((rms(&y) / 0.0883 - 1.0).abs() < 0.2, "громкость {} вместо 0.088", rms(&y));
        // тон, а не шум: переходов через ноль столько же, сколько у 440 Гц
        let middle = &y[y.len() / 4..3 * y.len() / 4];
        let zeros = middle.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        let want = 2.0 * 440.0 * middle.len() as f64 / 16000.0;
        assert!((zeros as f64 - want).abs() / want < 0.1, "частота уехала: {zeros} против {want:.0}");
    }
}
