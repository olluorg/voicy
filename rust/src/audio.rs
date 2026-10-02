//! Encoding to the formats the OpenAI audio API advertises: wav and pcm are
//! written directly, opus, mp3 and flac by the encoders built into the binary
//! (encode.rs); the tempo change is tempo.rs, and only aac still goes through
//! ffmpeg. Opus at 24 kbps is half the size of 32 kbps mp3 and sounds better
//! on speech. Resampling is libsoxr.

use std::path::Path;

use crate::errors::{ApiError, ApiResult};

pub use voicy_core::audio::*;

fn ffmpeg_args(fmt: &str) -> Option<&'static [&'static str]> {
    Some(match fmt {
        "aac" => &["-c:a", "aac", "-b:a", "96k"],
        _ => return None,
    })
}

async fn ffmpeg(args: &[&str], purpose: &str) -> ApiResult<()> {
    let out = tokio::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y"])
        .args(args)
        .output()
        .await;
    match out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(ApiError::internal(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ))),
        // ОС сама этого не скажет: на Windows это «не удаётся найти указанный файл»
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ApiError::internal(format!(
            "ffmpeg is not installed on the server; it is needed for {purpose}. \
             wav and pcm work without it"
        ))),
        Err(e) => Err(ApiError::internal(format!("ffmpeg: {e}"))),
    }
}

pub async fn encode(x: &[f32], sr: u32, fmt: &str) -> ApiResult<(Vec<u8>, &'static str)> {
    let fmt = fmt.to_lowercase();
    if fmt == "pcm" {
        return Ok((to_pcm16(x), content_type("pcm")));
    }
    let wav = to_wav(x, sr);
    if fmt == "wav" {
        return Ok((wav, content_type("wav")));
    }
    if crate::encode::own(&fmt) {
        let (x, sr, f) = (x.to_vec(), sr, fmt.clone());
        let data = tokio::task::spawn_blocking(move || crate::encode::encode(&x, sr, &f))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .map_err(|e| ApiError::internal(format!("{e:#}")))?;
        return Ok((data, content_type(&fmt)));
    }
    let Some(args) = ffmpeg_args(&fmt) else {
        return Err(ApiError::internal(format!("unsupported response_format: {fmt}")));
    };
    let dir = tempfile::tempdir()?;
    let (src, dst) = (dir.path().join("in.wav"), dir.path().join(format!("out.{fmt}")));
    tokio::fs::write(&src, &wav).await?;
    let mut all = vec!["-i", path_str(&src)];
    all.extend_from_slice(args);
    all.push(path_str(&dst));
    ffmpeg(&all, &format!("the {fmt} format")).await?;
    Ok((tokio::fs::read(&dst).await?, content_type(&fmt)))
}

/// Change tempo without pitch (tempo.rs). Applied to finished audio: the
/// model's own speed control raised the error rate fivefold (ADR 0005).
pub async fn stretch(x: Vec<f32>, sr: u32, factor: f64) -> ApiResult<Vec<f32>> {
    if (factor - 1.0).abs() < 1e-3 {
        return Ok(x);
    }
    tokio::task::spawn_blocking(move || crate::tempo::stretch(&x, sr, factor))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("temp path is utf-8")
}
