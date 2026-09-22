//! Any container a browser or a phone produces → mono float32 at a given rate.
//!
//! The Python server reads files through PyAV, which carries its own ffmpeg.
//! Here it is symphonia (pure Rust: wav, flac, mp3, ogg vorbis, aac and alac in
//! mp4) and, for what it cannot read — opus, webm — the system ffmpeg.

use std::io::{Cursor, Write};
use std::process::{Command, Stdio};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
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
