//! Описание и текст песни → музыка, через acestep.cpp, без сервера:
//!
//!     cargo run --release -p voicy-core --features internal --example music -- \
//!         MODELS_DIR "acoustic folk, warm" "[verse]\n…" 60 out.wav [ru]
//!
//! MODELS_DIR — каталог с GGUF ACE-Step (LM, кодировщик текста, DiT, VAE);
//! библиотека voicy-acestep — в каталоге библиотек voicy (VOICY_LIB_DIR).
//! Пустой текст — инструментал. ACESTEP_DIT / ACESTEP_LM выбирают файлы.

use std::path::{Path, PathBuf};
use std::time::Instant;

use voicy_core::native::acestep::{AceStep, Files, Stage};

fn pick(dir: &Path, var: &str, prefix: &str) -> PathBuf {
    if let Ok(name) = std::env::var(var) {
        return dir.join(name);
    }
    let mut names: Vec<_> = std::fs::read_dir(dir).expect("models dir").flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".gguf"))
        .collect();
    names.sort();
    dir.join(names.first().unwrap_or_else(|| panic!("нет {prefix}*.gguf в {}", dir.display())))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&a[0]);
    let (caption, lyrics) = (&a[1], a[2].replace("\\n", "\n"));
    let seconds: f64 = a[3].parse()?;
    let out = &a[4];
    let lang = a.get(5).cloned().unwrap_or_else(|| "en".into());
    let files = Files {
        lm: pick(&dir, "ACESTEP_LM", "acestep-5Hz-lm-"),
        text_encoder: pick(&dir, "ACESTEP_TEXT", "Qwen3-Embedding"),
        dit: pick(&dir, "ACESTEP_DIT", "acestep-v15-"),
        vae: pick(&dir, "ACESTEP_VAE", "vae"),
    };
    eprintln!("{files:#?}");
    let lib = voicy_core::native::lib_dir()?;
    let keep = std::env::var("ACESTEP_KEEP").is_ok_and(|v| v == "1");
    let t = Instant::now();
    let m = AceStep::load(&lib, &files, keep)?;
    eprintln!("загружено за {:.1} с", t.elapsed().as_secs_f64());

    for take in 0..std::env::var("TAKES").ok().and_then(|v| v.parse().ok()).unwrap_or(1) {
        let instrumental = lyrics.trim().is_empty();
        let req = serde_json::json!({
            "caption": caption,
            "lyrics": if instrumental { "[Instrumental]".to_string() } else { lyrics.clone() },
            "vocal_language": if instrumental { "unknown" } else { lang.as_str() },
            "duration": seconds, "seed": 1000 + take, "lm_seed": 1000 + take,
        });
        let t = Instant::now();
        let mut last = (Stage::Plan, 0);
        let mut marks: Vec<(Stage, f64)> = vec![];
        let song = m.generate(&req, |s, n| {
            if marks.last().map(|m| m.0) != Some(s) {
                marks.push((s, t.elapsed().as_secs_f64()));
            }
            last = (s, n);
            true
        })?.expect("не отменяли");
        let end = t.elapsed().as_secs_f64();
        let bounds: Vec<f64> = marks.iter().map(|m| m.1).chain([end]).collect();
        let spans: Vec<String> = marks.iter().enumerate()
            .map(|(i, (s, _))| format!("{s:?} {:.1}", bounds[i + 1] - bounds[i])).collect();
        eprintln!("этапы (с первого обращения): {}", spans.join(", "));
        let secs = song.audio.len() as f64 / 2.0 / song.sample_rate as f64;
        eprintln!("дубль {take}: {secs:.1} с музыки за {:.1} с, VRAM {:.2} ГБ, последний этап {last:?}",
                  t.elapsed().as_secs_f64(), m.vram() as f64 / 1e9);
        eprintln!("план: bpm {} key {}", song.plan["bpm"], song.plan["keyscale"]);
        let path = if take == 0 { out.clone() } else { out.replacen('.', &format!(".{take}."), 1) };
        let ext = path.rsplit('.').next().unwrap_or("wav");
        std::fs::write(&path, voicy_core::audio::encode_ch(&song.audio, song.sample_rate, 2, ext)?)?;
    }
    Ok(())
}
