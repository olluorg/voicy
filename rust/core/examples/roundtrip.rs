//! Текст → звук → текст, одним процессом, без сервера:
//!
//!     cargo run --release -p voicy-core --example roundtrip -- "Текст." out.opus

use std::time::Instant;

use voicy_core::voices::Voices;
use voicy_core::{SpeakOptions, Stt, TranscribeOptions, Tts};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args.next().unwrap_or_else(|| "Проверка связи: библиотека говорит и слышит сама.".into());
    let out = args.next().unwrap_or_else(|| "roundtrip.opus".into());

    let t = Instant::now();
    let tts = Tts::load(&Default::default())?;
    eprintln!("синтез загружен за {:.1} с на {}", t.elapsed().as_secs_f64(), tts.device());
    let voice = Voices::in_cache()?.default().expect("голоса по умолчанию");
    let t = Instant::now();
    let audio = tts.speak(&text, &voice, &SpeakOptions::default())?;
    let secs = audio.len() as f64 / tts.sample_rate() as f64;
    eprintln!("{secs:.2} с звука за {:.2} с", t.elapsed().as_secs_f64());
    let ext = out.rsplit('.').next().unwrap_or("wav");
    std::fs::write(&out, voicy_core::audio::encode(&audio, tts.sample_rate(), ext)?)?;

    let stt = Stt::load(&Default::default())?;
    let heard = stt.transcribe_file(&std::fs::read(&out)?, &TranscribeOptions { language: Some("ru".into()), ..Default::default() })?;
    println!("{}", heard.text);
    Ok(())
}
