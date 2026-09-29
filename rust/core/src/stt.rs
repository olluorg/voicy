//! Recognition: Whisper large-v3-turbo as faster-whisper runs it, on
//! CTranslate2 (native/fwhisper.rs), with the transcript as typed data.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::native::fwhisper::{self, FasterWhisper};
use crate::native::whisper::Request;

pub const STT_SAMPLE_RATE: u32 = 16000;

#[derive(Clone, Debug)]
pub struct SttConfig {
    /// The CTranslate2 model directory; by default large-v3-turbo from the cache.
    pub dir: Option<PathBuf>,
    /// Silero, for `live`; by default from the cache, if it is there.
    pub vad: Option<PathBuf>,
    pub gpu: bool,
    /// CTranslate2's compute type; by default float16 on the GPU, int8 on the CPU.
    pub compute_type: Option<String>,
}

impl Default for SttConfig {
    fn default() -> Self {
        SttConfig { dir: None, vad: None, gpu: true, compute_type: None }
    }
}

impl SttConfig {
    /// What the server reads: `FORCE_CPU=1` or `STT_DEVICE=cpu` for the CPU,
    /// `STT_COMPUTE_TYPE`.
    pub fn from_env() -> Self {
        SttConfig {
            gpu: std::env::var("FORCE_CPU").as_deref() != Ok("1")
                && std::env::var("STT_DEVICE").map_or(true, |d| d.starts_with("cuda")),
            compute_type: std::env::var("STT_COMPUTE_TYPE").ok().filter(|c| !c.is_empty()),
            ..SttConfig::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TranscribeOptions {
    /// ISO code; detected when None.
    pub language: Option<String>,
    /// Text that sets the context: the previous phrase, names, the subject.
    pub prompt: Option<String>,
    /// Terms to be heard as written: Kafka, not «кавка».
    pub hotwords: Vec<String>,
    pub temperature: f32,
    pub word_timestamps: bool,
    /// Into English instead of the language spoken.
    pub translate: bool,
    /// A phrase of a live conversation: silence cut out by Silero, no
    /// conditioning on the text before.
    pub live: bool,
    /// Faster and a little less accurate: beam 2 instead of 5.
    pub draft: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub word: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Empty unless `word_timestamps` was asked for.
    #[serde(default)]
    pub words: Vec<Word>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub language: String,
    pub language_probability: f64,
    /// Seconds of audio heard.
    pub duration: f64,
    pub segments: Vec<Segment>,
}

impl Transcript {
    pub fn srt(&self) -> String {
        subtitles(self.segments.iter().map(|s| (s.start, s.end, s.text.as_str())), false)
    }

    pub fn vtt(&self) -> String {
        subtitles(self.segments.iter().map(|s| (s.start, s.end, s.text.as_str())), true)
    }
}

fn srt_time(t: f64) -> String {
    let (h, rem) = ((t / 3600.0).floor(), t % 3600.0);
    let (m, s) = ((rem / 60.0).floor(), rem % 60.0);
    format!("{:02}:{:02}:{:02},{:03}", h as i64, m as i64, s as i64, ((s % 1.0) * 1000.0) as i64)
}

/// (start, end, text) as SubRip, or as WebVTT when `vtt`.
pub fn subtitles<'a>(segments: impl Iterator<Item = (f64, f64, &'a str)>, vtt: bool) -> String {
    let mut lines = if vtt { vec!["WEBVTT".to_string(), String::new()] } else { vec![] };
    for (i, (start, end, text)) in segments.enumerate() {
        if vtt {
            lines.push(format!("{} --> {}", srt_time(start).replace(',', "."), srt_time(end).replace(',', ".")));
        } else {
            lines.push((i + 1).to_string());
            lines.push(format!("{} --> {}", srt_time(start), srt_time(end)));
        }
        lines.push(text.into());
        lines.push(String::new());
    }
    lines.join("\n")
}

pub struct Stt {
    engine: FasterWhisper,
}

impl Stt {
    /// Load the model: seconds, and 1.5 GB of memory on the GPU. Asked for the
    /// GPU where CUDA sees none, it loads on the CPU.
    pub fn load(cfg: &SttConfig) -> anyhow::Result<Stt> {
        let models = crate::cache_dir().join("models");
        let dir = cfg.dir.clone().unwrap_or_else(|| models.join("whisper").join("faster-whisper-large-v3-turbo"));
        anyhow::ensure!(dir.join("model.bin").is_file(), "no Whisper model in {} — voicy setup models", dir.display());
        let vad = cfg.vad.clone().or_else(|| Some(models.join("vad").join("silero_vad_v6.onnx")).filter(|p| p.is_file()));
        Ok(Stt { engine: FasterWhisper::load(&dir, vad.as_deref(), crate::native::use_gpu(cfg.gpu), cfg.compute_type.as_deref())? })
    }

    pub fn device(&self) -> &str {
        &self.engine.device
    }

    pub fn compute_type(&self) -> &str {
        &self.engine.compute_type
    }

    /// 16 kHz mono samples → text.
    pub fn transcribe(&self, samples: &[f32], o: &TranscribeOptions) -> anyhow::Result<Transcript> {
        self.transcribe_with(samples, o, |_, _, _| true)?.ok_or_else(|| anyhow::anyhow!("stopped"))
    }

    /// Any file a phone, a browser or a recorder makes — wav, mp3, flac, ogg,
    /// opus, webm, mp4 — → text.
    pub fn transcribe_file(&self, bytes: &[u8], o: &TranscribeOptions) -> anyhow::Result<Transcript> {
        let mut samples = crate::audio::decode(bytes, STT_SAMPLE_RATE)?;
        // как faster-whisper читает файл: через 16-битные отсчёты
        fwhisper::through_s16(&mut samples);
        self.transcribe(&samples, o)
    }

    /// The same as `transcribe`, telling `on_segment` each segment as it is
    /// heard: (where it ends, s; the whole duration, s; its text). It returns
    /// false to stop, and then the result is None.
    pub fn transcribe_with(&self, samples: &[f32], o: &TranscribeOptions,
                           mut on_segment: impl FnMut(f64, f64, &str) -> bool) -> anyhow::Result<Option<Transcript>> {
        let hotwords = (!o.hotwords.is_empty()).then_some(o.hotwords.as_slice());
        let req = Request {
            language: o.language.as_deref(),
            prompt: o.prompt.as_deref(),
            hotwords,
            temperature: o.temperature,
            word_timestamps: o.word_timestamps,
            translate: o.translate,
            live: o.live,
            draft: o.draft,
        };
        match self.engine.transcribe(samples, &req, &mut on_segment)? {
            Some(v) => Ok(Some(serde_json::from_value(v)?)),
            None => Ok(None),
        }
    }
}

#[cfg(feature = "tokio")]
impl Stt {
    /// `load` on the blocking pool.
    pub async fn load_async(cfg: SttConfig) -> anyhow::Result<Stt> {
        tokio::task::spawn_blocking(move || Stt::load(&cfg)).await?
    }

    /// `transcribe_file` on the blocking pool.
    pub async fn transcribe_file_async(self: std::sync::Arc<Self>, bytes: Vec<u8>, o: TranscribeOptions)
                                       -> anyhow::Result<Transcript> {
        tokio::task::spawn_blocking(move || self.transcribe_file(&bytes, &o)).await?
    }

    /// `transcribe` on the blocking pool.
    pub async fn transcribe_async(self: std::sync::Arc<Self>, samples: Vec<f32>, o: TranscribeOptions)
                                  -> anyhow::Result<Transcript> {
        tokio::task::spawn_blocking(move || self.transcribe(&samples, &o)).await?
    }
}
