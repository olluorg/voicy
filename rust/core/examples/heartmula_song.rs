//! Песня родным HeartMuLa, без сервера:
//!
//!     cargo run --release -p voicy-core --features internal --example heartmula_song -- \
//!         DIR BACKBONE.gguf "теги" "текст" MAX_SECONDS out.wav [seed]
use std::path::PathBuf;
use std::time::Instant;

use voicy_core::native::heartmula::{Files, HeartMuLa, SAMPLE_RATE, Stage};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let files = Files { dir: PathBuf::from(&a[0]), backbone: a[1].clone(), decoder: std::env::var("HEARTMULA_DECODER").unwrap_or_else(|_| "heartmula_decoder.q8_0.gguf".into()) };
    let t = Instant::now();
    let m = HeartMuLa::load(&files, true, std::env::var("MUSIC_KEEP_LOADED").is_ok_and(|v| v == "1"))?;
    eprintln!("загружено за {:.1} с", t.elapsed().as_secs_f64());
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1);
    let t = Instant::now();
    let mut frames_at = 0.0;
    let mut last_frame = 0.0;
    let audio = m.generate(&a[2], &a[3].replace("\\n", "\n"), a[4].parse()?, seed, |s| {
        match s {
            Stage::Frames(n) => {
                last_frame = t.elapsed().as_secs_f64();
                if n % 125 == 0 {
                    eprintln!("кадров {n} за {last_frame:.1} с");
                }
            }
            Stage::Codec { window, windows } => {
                if window == 0 {
                    frames_at = t.elapsed().as_secs_f64();
                }
                eprintln!("кодек: окно {} из {windows}", window + 1)
            }

        }
        true
    })?.expect("не отменяли");
    let secs = audio.len() as f64 / 2.0 / SAMPLE_RATE as f64;
    let total = t.elapsed().as_secs_f64();
    eprintln!("{secs:.1} с песни за {total:.1} с: кадры {last_frame:.1} с, загрузка кодека {:.1} с, кодек {:.1} с",
              frames_at - last_frame, total - frames_at);
    std::fs::write(&a[5], voicy_core::audio::encode_ch(&audio, SAMPLE_RATE, 2, "wav")?)?;
    Ok(())
}
