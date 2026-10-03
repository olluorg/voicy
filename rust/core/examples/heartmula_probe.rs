//! Сверка родного HeartMuLa с heartlib: логиты кодбука 0 после подсказки и
//! логиты кодбука 1 из декодера при заданном коде 0 — в два файла f32.
//!
//!     cargo run --release -p voicy-core --features internal --example heartmula_probe -- \
//!         DIR BACKBONE.gguf "теги" "текст" C0 OUT_PREFIX
use std::path::PathBuf;

use voicy_core::native::heartmula::{Files, HeartMuLa};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let files = Files { dir: PathBuf::from(&a[0]), backbone: a[1].clone(), decoder: std::env::var("HEARTMULA_DECODER").unwrap_or_else(|_| "heartmula_decoder.q8_0.gguf".into()) };
    let m = HeartMuLa::load(&files, true, std::env::var("MUSIC_KEEP_LOADED").is_ok_and(|v| v == "1"))?;
    let (l0, l1) = m.probe(&a[2], &a[3].replace("\\n", "\n"), a[4].parse()?)?;
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(format!("{}.c0.f32", a[5]), bytes(&l0))?;
    std::fs::write(format!("{}.c1.f32", a[5]), bytes(&l1))?;
    eprintln!("c0 argmax {}, c1 argmax {}", argmax(&l0), argmax(&l1));
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0)
}
