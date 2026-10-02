//! Synthesis: Qwen3-TTS in-process, with what the server does around it —
//! the text prepared (textprep), the tempo changed on the finished sound.

use std::path::PathBuf;

use anyhow::Context as _;

use crate::native::qwen::{self, Qwen};
use crate::voices::Voice;
use crate::{STRESS_MARKER, textprep};

pub const TTS_SAMPLE_RATE: u32 = qwen::SAMPLE_RATE;

/// Languages the model speaks: (ISO code, English name, the model's id).
pub use qwen::LANGUAGES;

#[derive(Clone, Debug)]
pub struct TtsConfig {
    /// The model directory; by default the one `voicy setup` installs (`tts_dir`).
    pub dir: Option<PathBuf>,
    /// Quantisation of the talker: q5_k (2.4 GB), q8_0 or f16. q5_k is as
    /// intelligible and as close to the voice as f16 (experiments/20).
    pub talker: String,
    pub gpu: bool,
}

impl Default for TtsConfig {
    fn default() -> Self {
        TtsConfig { dir: None, talker: "q5_k".into(), gpu: true }
    }
}

impl TtsConfig {
    /// What the server reads: `VOICY_TTS_GGUF_DIR`, `TTS_GGUF_TALKER`, and
    /// `FORCE_CPU=1` for the CPU.
    pub fn from_env() -> Self {
        let d = TtsConfig::default();
        TtsConfig {
            dir: std::env::var_os("VOICY_TTS_GGUF_DIR").filter(|v| !v.is_empty()).map(PathBuf::from),
            talker: std::env::var("TTS_GGUF_TALKER").unwrap_or(d.talker),
            gpu: std::env::var("FORCE_CPU").as_deref() != Ok("1"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeakOptions {
    /// ISO code or English name: ru, en, de, fr, es, it, pt, ja, ko, zh.
    pub language: String,
    /// 0.8–1.2, by stretching the finished sound: the model's own slowing
    /// down made it five times less intelligible (docs/adr/0005).
    pub speed: f64,
    pub seed: Option<u32>,
    /// The pronunciation dictionary: terms spelt the way the model reads them.
    pub prepare: bool,
    /// Drop commas inside short sentences: a comma makes the model stop.
    pub legato: bool,
    /// Stress marks by RUAccent — for Russian, and only when the model
    /// understands them; the official weights mangle a marked word.
    pub stress: bool,
}

impl Default for SpeakOptions {
    fn default() -> Self {
        SpeakOptions { language: "ru".into(), speed: 1.0, seed: None, prepare: false, legato: false, stress: true }
    }
}

pub struct Tts {
    engine: Qwen,
    stress: bool,
}

impl Tts {
    /// Load the model: seconds to a minute, and 3.5 GB of memory on the GPU.
    /// Asked for the GPU where there is none, it loads on the CPU.
    pub fn load(cfg: &TtsConfig) -> anyhow::Result<Tts> {
        let dir = cfg.dir.clone().unwrap_or_else(crate::tts_dir);
        let files = qwen::Files {
            talker: format!("qwen3_tts_talker.{}.gguf", cfg.talker),
            predictor: "qwen3_tts_predictor.q8_0.gguf".into(),
            dir,
        };
        anyhow::ensure!(files.dir.join(&files.talker).is_file(),
                        "no {} in {} — voicy setup models", files.talker, files.dir.display());
        let stress = files.dir.join(STRESS_MARKER).is_file();
        Ok(Tts { engine: Qwen::load(&files, crate::native::use_gpu(cfg.gpu))?, stress })
    }

    pub fn sample_rate(&self) -> u32 {
        TTS_SAMPLE_RATE
    }

    pub fn device(&self) -> &str {
        &self.engine.device
    }

    /// Whether the model reads U+0301 as a stress mark (docs/adr/0023).
    pub fn understands_stress(&self) -> bool {
        self.stress
    }

    /// `text` read in `voice`: mono samples at `sample_rate()`.
    pub fn speak(&self, text: &str, voice: &Voice, o: &SpeakOptions) -> anyhow::Result<Vec<f32>> {
        self.speak_with(text, voice, o, |_| true).map(|a| a.unwrap_or_default())
    }

    /// The same, telling `progress` how many seconds of sound are made so far;
    /// it returns false to stop, and then the result is None.
    pub fn speak_with(&self, text: &str, voice: &Voice, o: &SpeakOptions, mut progress: impl FnMut(f64) -> bool)
                      -> anyhow::Result<Option<Vec<f32>>> {
        let lang = qwen::language_id(&o.language).with_context(|| {
            let all: Vec<&str> = LANGUAGES.iter().map(|l| l.0).collect();
            format!("language '{}' is not supported; supported: {}", o.language, all.join(", "))
        })?;
        anyhow::ensure!((0.8..=1.2).contains(&o.speed), "speed must be 0.8–1.2, got {}", o.speed);
        anyhow::ensure!(!voice.text.is_empty(), "voice '{}' has no reference transcript", voice.name);
        let russian = LANGUAGES.iter().any(|l| l.0 == "ru" && l.2 == lang);
        let text = textprep::for_speech(text, o.prepare, o.legato, o.stress && russian && self.stress);
        let frame = qwen::SAMPLES_PER_FRAME as f64 / TTS_SAMPLE_RATE as f64;
        let audio = self.engine.speak(&text, &voice.path, &voice.text, Some(lang), o.seed, |p| {
            progress(p.frames as f64 * frame) // кадр — ровно 80 мс готового звука
        })?;
        Ok(audio.map(|a| if (o.speed - 1.0).abs() < 1e-3 { a } else { crate::tempo::stretch(&a, TTS_SAMPLE_RATE, o.speed) }))
    }
}

#[cfg(feature = "tokio")]
impl Tts {
    /// `load` on the blocking pool.
    pub async fn load_async(cfg: TtsConfig) -> anyhow::Result<Tts> {
        tokio::task::spawn_blocking(move || Tts::load(&cfg)).await?
    }

    /// `speak` on the blocking pool.
    pub async fn speak_async(self: std::sync::Arc<Self>, text: String, voice: Voice, o: SpeakOptions)
                             -> anyhow::Result<Vec<f32>> {
        tokio::task::spawn_blocking(move || self.speak(&text, &voice, &o)).await?
    }
}
