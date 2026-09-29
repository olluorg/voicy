//! voicy's speech engines as a library — the same ones the voicy server runs,
//! in the caller's process, without HTTP.
//!
//! ```no_run
//! use voicy_core::{Stt, Tts, SpeakOptions, TranscribeOptions, voices::Voices};
//!
//! let tts = Tts::load(&Default::default())?;
//! let voice = Voices::in_cache()?.get("turgenev").expect("ships with voicy");
//! let audio = tts.speak("Проверка связи.", &voice, &SpeakOptions::default())?;
//! std::fs::write("out.opus", voicy_core::audio::encode(&audio, tts.sample_rate(), "opus")?)?;
//!
//! let stt = Stt::load(&Default::default())?;
//! let heard = stt.transcribe_file(&std::fs::read("out.opus")?, &TranscribeOptions::default())?;
//! println!("{}", heard.text);
//! # anyhow::Ok(())
//! ```
//!
//! - [`Tts`] — Qwen3-TTS on llama.cpp and ONNX Runtime: text and a reference
//!   voice in, 24 kHz mono samples out; dictionary, legato and stress marks
//!   as in the server ([`textprep`]).
//! - [`Stt`] — Whisper large-v3-turbo as faster-whisper runs it, on CTranslate2.
//! - [`Vad`] and [`Turn`] — Silero voice detection and Smart Turn, the end of
//!   a speaker's turn, for a live conversation.
//! - [`segment::Segmenter`] — text in any pieces (an LLM's tokens) cut where
//!   a synthesiser can start early.
//! - [`audio`] — decoding any common container, encoding to wav, pcm, opus,
//!   mp3 and flac, resampling; [`tempo`] — speed without pitch.
//!
//! What is not here is what the Python engines do in the server: this crate
//! has only the in-process ones. Their runtimes are prebuilt shared libraries
//! and the models are files, both in the cache (`~/.cache/voicy`,
//! `VOICY_CACHE`), loaded at run time: `voicy setup` puts them there, or
//! [`setup::run`] with the `download` feature. [`use_dirs`] points the crate
//! at another cache from code.
//!
//! The cache is found as `voicy setup` finds it — `VOICY_CACHE`, else the
//! default place; nothing else is read from the environment unless asked:
//! `TtsConfig::from_env` and `SttConfig::from_env` read what the server reads.
//! Messages go through `log`: what is missing (no RUAccent — speech without
//! stress marks) at `warn`, `setup` progress at `info`.
//!
//! Every call blocks: the engines hold the GPU for as long as they work. The
//! `tokio` feature adds `*_async` twins that run on the blocking pool. Loaded
//! engines are `Send + Sync`; each serialises its own calls, so one shared
//! `Arc<Tts>` is the way to use it from several threads.

pub mod audio;
// Привязки к движкам и кодеки — внутреннее: снаружи их видит только сервер
// voicy (фича `internal`), и меняться они вправе без оглядки на semver.
#[cfg(feature = "internal")]
#[doc(hidden)]
pub mod encode;
#[cfg(not(feature = "internal"))]
#[allow(dead_code)]
mod encode;
#[cfg(feature = "internal")]
#[doc(hidden)]
pub mod native;
#[cfg(not(feature = "internal"))]
#[allow(dead_code)]
mod native;
pub mod segment;
#[cfg(feature = "download")]
pub mod setup;
mod stt;
pub mod tempo;
pub mod textprep;
mod tts;
pub mod voices;

use std::path::PathBuf;

pub use native::listen::{Turn, VAD_FRAME, Vad, VadStream};
pub use stt::{STT_SAMPLE_RATE, Segment, Stt, SttConfig, TranscribeOptions, Transcript, Word, subtitles};
pub use tts::{LANGUAGES, SpeakOptions, TTS_SAMPLE_RATE, Tts, TtsConfig};
pub use voices::Voices;
pub use voices::Voice;

/// The default synthesis model: the one that obeys stress marks when it is
/// installed, the official weights otherwise.
pub const TTS_STRESS_DIR: &str = "qwen3-tts-12hz-1.7b-ru-stress-gguf";
pub const TTS_BASE_DIR: &str = "qwen3-tts-12hz-1.7b-base-gguf";
/// Beside the weights: the model was trained to obey U+0301 (docs/adr/0023).
pub const STRESS_MARKER: &str = "stress.json";

/// The cache: libraries in `lib/<platform>`, models in `models/`.
pub fn cache_dir() -> PathBuf {
    native::cache_dir()
}

/// Use this cache and, if given, this library directory instead of what the
/// environment says (`VOICY_CACHE`, `VOICY_LIB_DIR`). Once per process and
/// before the first engine loads: the runtimes, once loaded, stay.
pub fn use_dirs(cache: impl Into<PathBuf>, lib: Option<PathBuf>) -> anyhow::Result<()> {
    anyhow::ensure!(native::set_dirs(Some(cache.into()), lib), "voicy_core: the directories are already set for this process");
    Ok(())
}

/// The Qwen3-TTS directory in the cache: the model with stress marks once
/// setup has begun to put it here, else the official weights if they are what
/// is installed.
pub fn tts_dir() -> PathBuf {
    let root = cache_dir().join("models");
    let (stress, base) = (root.join(TTS_STRESS_DIR), root.join(TTS_BASE_DIR));
    if !stress.is_dir() && base.is_dir() { base } else { stress }
}

fn models() -> PathBuf {
    cache_dir().join("models")
}

/// ONNX Runtime from the engine libraries, loaded now. The `ort` crate is one
/// per program, and in voicy-core it loads ONNX Runtime at run time: a program
/// that makes its own `ort` sessions calls this first. The engines call it
/// themselves.
pub fn init_onnx_runtime() -> anyhow::Result<()> {
    native::init_onnx(&native::lib_dir()?)
}

impl Vad {
    /// Silero v6 from the cache.
    pub fn from_cache() -> anyhow::Result<Vad> {
        native::init_onnx(&native::lib_dir()?)?;
        Vad::load(&models().join("vad").join("silero_vad_v6.onnx"))
    }
}

impl Turn {
    /// Smart Turn v3.2 from the cache.
    pub fn from_cache() -> anyhow::Result<Turn> {
        native::init_onnx(&native::lib_dir()?)?;
        Turn::load(&models().join("turn").join("smart-turn-v3.2-cpu.onnx"))
    }
}

#[cfg(test)]
mod tests {
    /// Один загруженный движок на всю программу: Arc<Tts> между потоками.
    #[test]
    fn engines_are_shareable() {
        fn shareable<T: Send + Sync>() {}
        shareable::<super::Tts>();
        shareable::<super::Stt>();
        shareable::<super::Vad>();
        shareable::<super::Turn>();
    }
}
