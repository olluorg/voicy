//! The engines behind the server, and what they said about themselves at start.
//!
//! Each kind runs in this process when it can — Qwen3-TTS on llama.cpp and ONNX
//! Runtime, Whisper as faster-whisper runs it on CTranslate2, Silero and Smart
//! Turn on ONNX Runtime (native/) — and otherwise in the Python host process
//! (host.rs), as the Python server runs them. In-process is chosen when the
//! runtime libraries and the model are in the cache and no other engine is
//! asked for by TTS_ENGINE, STT_ENGINE, TURN_ENGINE or VAD_ENGINE. Whisper on
//! whisper.cpp is less accurate and runs only when asked for
//! (STT_ENGINE=whisper.cpp). When all four are in-process, Python is not
//! started at all.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use tokio::sync::OnceCell;

use crate::errors::ApiError;
use crate::host::{Host, HostError, Msg, f32_bytes, f32_from};
use crate::native;

pub struct Engines {
    host: Option<Arc<Host>>,
    pub info: Value,
    tts: Option<NativeTts>,
    stt: Option<NativeStt>,
    vad: Option<Arc<native::listen::Vad>>,
    turn: Option<Arc<native::listen::Turn>>,
    pub music: Music,
}

/// One live session's voice detector: in the host process or in this one.
pub enum VadHandle {
    Host(u64),
    Native(std::sync::Mutex<native::listen::VadStream>),
}

pub enum Stop {
    /// Stopped because the caller asked (a job cancelled, a client gone).
    Cancelled,
    Failed(HostError),
}

impl From<Stop> for ApiError {
    fn from(s: Stop) -> Self {
        match s {
            Stop::Cancelled => ApiError::new(409, "cancelled"),
            Stop::Failed(e) => e.into(),
        }
    }
}

pub struct Speech {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

fn err(kind: &str, message: impl Into<String>) -> HostError {
    HostError { kind: kind.into(), message: message.into() }
}

/// The engine the environment asks for, if any; in-process only when it is
/// unset or names this engine (or its Python twin, which it replaces).
fn wanted(var: &str, names: &[&str]) -> Option<bool> {
    match std::env::var(var).unwrap_or_default().as_str() {
        "" => Some(false),
        v if names.contains(&v) => Some(true),
        _ => None,
    }
}

fn models() -> PathBuf {
    native::cache_dir().join("models")
}

// ----------------------------------------------------------------- синтез

/// Qwen3-TTS in-process: loaded on first use, as the Python engines are.
struct NativeTts {
    files: native::qwen::Files,
    engine: OnceCell<Arc<native::qwen::Qwen>>,
    variant: String,
    /// stress.json рядом с весами: модель дообучена слушаться знака ударения (docs/adr/0023)
    stress: Option<Value>,
}

use voicy_core::STRESS_MARKER;

impl NativeTts {
    fn find() -> Option<NativeTts> {
        let explicit = wanted("TTS_ENGINE", &["qwen3-tts-gguf"])?;
        // q5_k: та же разборчивость и то же сходство голоса, что у f16 и torch, при 2.4 ГБ (experiments/20)
        let variant = voicy_core::TtsConfig::from_env().talker;
        let talker = format!("qwen3_tts_talker.{variant}.gguf");
        let dir = crate::setup::tts_dir();
        let files = native::qwen::Files {
            talker,
            predictor: "qwen3_tts_predictor.q8_0.gguf".into(),
            dir,
        };
        let ready = files.dir.join(&files.talker).is_file() && native::lib_dir().is_ok();
        if !ready && explicit {
            eprintln!("voicy: TTS_ENGINE=qwen3-tts-gguf, but no model in {} or no engine libraries", files.dir.display());
        }
        let stress = std::fs::read(files.dir.join(STRESS_MARKER)).ok().and_then(|b| serde_json::from_slice(&b).ok());
        ready.then(|| NativeTts { files, engine: OnceCell::new(), variant, stress })
    }

    async fn get(&self) -> Result<Arc<native::qwen::Qwen>, HostError> {
        self.engine
            .get_or_try_init(|| async {
                let files = native::qwen::Files {
                    dir: self.files.dir.clone(),
                    talker: self.files.talker.clone(),
                    predictor: self.files.predictor.clone(),
                };
                tokio::task::spawn_blocking(move || native::qwen::Qwen::load(&files, tts_gpu()))
                    .await
                    .map_err(|e| err("internal", e.to_string()))?
                    .map(Arc::new)
                    .map_err(|e| err("internal", format!("{e:#}")))
            })
            .await
            .cloned()
    }

    fn status(&self) -> Value {
        let model = self.stress.as_ref().and_then(|s| s["model"].as_str()).unwrap_or("Qwen/Qwen3-TTS-12Hz-1.7B-Base");
        json!({"engine": "qwen3-tts", "model": format!("{model} ({})", self.variant), "stress_marks": self.stress.is_some(),
               "runtime": "llama.cpp + onnxruntime", "loaded": self.engine.initialized(),
               "device": self.engine.get().map_or_else(tts_device, |e| e.device.clone())})
    }

    fn info(&self) -> Value {
        let mut v = self.status();
        v["sample_rate"] = json!(native::qwen::SAMPLE_RATE);
        v["needs_reference_text"] = json!(true);
        v["reference_seconds"] = json!([3.0, 60.0]);
        v["reference_best"] = json!([8.0, 14.0]);
        v
    }
}

// ---------------------------------------------------------- распознавание

const STT_FEATURES: [&str; 4] = ["hotwords", "prompt", "translate", "word_timestamps"];

/// Whisper large-v3-turbo in-process, loaded on first use: as faster-whisper
/// runs it — CTranslate2 and a port of its Python (native/fwhisper.rs) — or,
/// when asked for, on whisper.cpp.
struct NativeStt {
    kind: SttKind,
    name: String,
    engine: OnceCell<Arc<SttModel>>,
}

#[derive(Clone)]
enum SttKind {
    /// CTranslate2 model directory and Silero for live speech.
    Ct2 { dir: PathBuf, vad: PathBuf },
    /// ggml model and whisper.cpp's own Silero.
    Cpp { model: PathBuf, vad: PathBuf },
}

enum SttModel {
    Ct2(native::fwhisper::FasterWhisper),
    Cpp(native::whisper::Whisper),
}

/// Whether synthesis runs on the GPU: unless FORCE_CPU=1, the backend is the
/// CPU, or CUDA sees no device.
fn tts_gpu() -> bool {
    native::use_gpu(voicy_core::TtsConfig::from_env().gpu)
}

/// cuda, vulkan, rocm, sycl or cpu.
fn tts_device() -> String {
    native::device_name(tts_gpu())
}

/// faster-whisper has a GPU only through CUDA; whisper.cpp, on any backend.
fn stt_device(kind: &SttKind) -> String {
    let wanted = voicy_core::SttConfig::from_env().gpu;
    match kind {
        SttKind::Ct2 { .. } => if native::use_cuda(wanted) { "cuda".into() } else { "cpu".into() },
        SttKind::Cpp { .. } => native::device_name(native::use_gpu(wanted)),
    }
}

impl NativeStt {
    /// faster-whisper when its library and model are in the cache — the same
    /// recogniser as in Python (experiments/22); whisper.cpp only when asked
    /// for: less accurate (experiments/21).
    fn find() -> Option<NativeStt> {
        let name = std::env::var("STT_MODEL").unwrap_or_else(|_| "large-v3-turbo".into());
        let dir = models().join("whisper");
        let lib = native::lib_dir().ok();
        let has = |stem: &str| lib.as_ref().is_some_and(|d| {
            [format!("lib{stem}.so"), format!("{stem}.dll"), format!("lib{stem}.dylib")].iter().any(|n| d.join(n).exists())
        });
        let kind = match std::env::var("STT_ENGINE").unwrap_or_default().as_str() {
            "whisper.cpp" => {
                let model = dir.join(format!("ggml-{name}.bin"));
                if !(model.is_file() && has("whisper")) {
                    eprintln!("voicy: STT_ENGINE=whisper.cpp, but no {} or no libwhisper", model.display());
                    return None;
                }
                SttKind::Cpp { model, vad: dir.join("ggml-silero-v6.2.0.bin") }
            }
            "" | "faster-whisper" => {
                let ct2 = dir.join(format!("faster-whisper-{name}"));
                // на Windows обёртка едет внутри бинарника и кладётся на место при загрузке
                let shim = cfg!(all(windows, target_env = "msvc")) || has("ct2shim");
                if !(ct2.join("model.bin").is_file() && shim) {
                    return None;
                }
                SttKind::Ct2 { dir: ct2, vad: models().join("vad").join("silero_vad_v6.onnx") }
            }
            _ => return None,
        };
        Some(NativeStt { kind, name, engine: OnceCell::new() })
    }

    async fn get(&self) -> Result<Arc<SttModel>, HostError> {
        self.engine
            .get_or_try_init(|| async {
                let kind = self.kind.clone();
                let cfg = voicy_core::SttConfig::from_env();
                tokio::task::spawn_blocking(move || match kind {
                    SttKind::Ct2 { dir, vad } => {
                        native::fwhisper::FasterWhisper::load(&dir, Some(&vad), native::use_cuda(cfg.gpu), cfg.compute_type.as_deref()).map(SttModel::Ct2)
                    }
                    SttKind::Cpp { model, vad } => native::whisper::Whisper::load(&model, Some(vad), native::use_gpu(cfg.gpu)).map(SttModel::Cpp),
                })
                .await
                .map_err(|e| err("internal", e.to_string()))?
                .map(Arc::new)
                .map_err(|e| err("internal", format!("{e:#}")))
            })
            .await
            .cloned()
    }

    fn runtime(&self) -> &'static str {
        match self.kind {
            SttKind::Ct2 { .. } => "ctranslate2",
            SttKind::Cpp { .. } => "whisper.cpp",
        }
    }

    fn status(&self) -> Value {
        let engine = match self.kind {
            SttKind::Ct2 { .. } => "faster-whisper",
            SttKind::Cpp { .. } => "whisper.cpp",
        };
        let mut v = json!({"engine": engine, "model": self.name, "runtime": self.runtime(),
                           "loaded": self.engine.initialized(), "features": STT_FEATURES});
        match self.engine.get().map(|e| &**e) {
            Some(SttModel::Ct2(w)) => {
                v["device"] = json!(w.device);
                v["compute_type"] = json!(w.compute_type);
            }
            Some(SttModel::Cpp(w)) => v["device"] = json!(w.device),
            None => v["device"] = json!(stt_device(&self.kind)),
        }
        v
    }
}

// ------------------------------------------------------- слушающая сторона

/// Silero and Smart Turn in-process — the same ONNX files the Python engines run.
fn native_listen() -> (Option<Arc<native::listen::Vad>>, Option<Arc<native::listen::Turn>>) {
    let Ok(lib) = native::lib_dir() else { return (None, None) };
    if native::init_onnx(&lib).is_err() {
        return (None, None);
    }
    let m = models();
    let vad = m.join("vad").join("silero_vad_v6.onnx");
    let turn = m.join("turn").join("smart-turn-v3.2-cpu.onnx");
    let vad = (wanted("VAD_ENGINE", &["silero"]).is_some() && vad.is_file())
        .then(|| native::listen::Vad::load(&vad).map(Arc::new).map_err(|e| eprintln!("voicy: vad: {e:#}")).ok())
        .flatten();
    let turn = (wanted("TURN_ENGINE", &["smart-turn"]).is_some() && turn.is_file())
        .then(|| native::listen::Turn::load(&turn).map(Arc::new).map_err(|e| eprintln!("voicy: turn: {e:#}")).ok())
        .flatten();
    (vad, turn)
}

// ------------------------------------------------------------------ музыка

/// Языки, на которых поёт ACE-Step: перечень его LM (metadata-fsm.h).
const ACE_LANGUAGES: [&str; 50] = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr",
    "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro",
    "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
];

/// What a music request asks, in the API's terms; each engine turns it into
/// what its model takes (docs/adr/0021).
#[derive(Clone)]
pub struct MusicAsk {
    pub prompt: String,
    /// `Some("")` never comes: no lyrics is `None`
    pub lyrics: Option<String>,
    pub instrumental: bool,
    pub language: Option<String>,
    pub seconds: Option<f64>,
    pub seed: Option<i64>,
}

/// One step of progress: the stage, and its step of total when the model knows.
pub struct MusicStep {
    pub stage: &'static str,
    pub step: usize,
    pub total: Option<usize>,
}

pub struct MusicPiece {
    /// Interleaved, L R L R …
    pub audio: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
    /// What the model chose itself: bpm, key, language, lyrics; null when nothing.
    pub plan: Value,
}

#[derive(Clone)]
enum MusicKind {
    Ace(native::acestep::Files),
    Sa3 { models: PathBuf, variant: String },
    HeartMuLa(native::heartmula::Files),
}

/// One music model the server offers, and what it says about itself
/// (docs/adr/0028): the request picks it by `model`.
pub struct MusicModel {
    pub id: String,
    pub card: Value,
    kind: MusicKind,
}

#[derive(Clone)]
enum Loaded {
    Ace(Arc<native::acestep::AceStep>),
    Sa3(Arc<native::sa3::Sa3>),
    HeartMuLa(Arc<native::heartmula::HeartMuLa>),
}

/// Every music model installed, at most one of them in memory: a model the
/// next request does not want is dropped before the next one loads — two do
/// not fit on a 10 GB card beside speech.
pub struct Music {
    pub models: Vec<MusicModel>,
    pub default: Option<String>,
    loaded: tokio::sync::Mutex<Option<(String, Loaded)>>,
}

/// GGUF in `dir` whose name starts with `prefix`; with several, the first by name.
fn gguf(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let mut names: Vec<String> = std::fs::read_dir(dir).ok()?.flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(prefix) && n.ends_with(".gguf"))
        .collect();
    names.sort();
    names.first().map(|n| dir.join(n))
}

/// Keep every part of the model in VRAM between pieces (MUSIC_KEEP_LOADED=1):
/// faster, but beside speech it fits only on a big card.
fn music_keep_loaded() -> bool {
    std::env::var("MUSIC_KEEP_LOADED").is_ok_and(|v| v == "1")
}

impl Music {
    fn find() -> Music {
        let mut models = vec![];
        if let Ok(lib) = native::lib_dir() {
            if native::acestep::lib_path(&lib).exists() {
                models.extend(Self::ace_models());
            }
            if native::sa3::lib_path(&lib).exists() {
                models.extend(Self::sa3_models());
            }
            // HeartMuLa — на llama.cpp и ONNX Runtime, своей библиотеки ей не нужно
            models.extend(Self::heartmula_models());
        }
        let default = std::env::var("MUSIC_MODEL").ok().filter(|m| models.iter().any(|x: &MusicModel| &x.id == m))
            .or_else(|| models.first().map(|m| m.id.clone()));
        Music { models, default, loaded: Default::default() }
    }

    /// ACE-Step in `models/music/ace-step`: the turbo DiT is `ace-step`, the
    /// 4B XL turbo `ace-step-xl`; they share the LM (the larger one when there
    /// are two), the text encoder and the VAE.
    fn ace_models() -> Vec<MusicModel> {
        let dir = crate::setup::music_dir();
        let lm = std::env::var("MUSIC_LM").ok().map(|n| dir.join(n))
            .or_else(|| gguf(&dir, "acestep-5Hz-lm-1.7B"))
            .or_else(|| gguf(&dir, "acestep-5Hz-lm-0.6B"));
        let (Some(lm), Some(text_encoder), Some(vae)) = (lm, gguf(&dir, "Qwen3-Embedding"), gguf(&dir, "vae")) else {
            return vec![];
        };
        let mut out = vec![];
        for (id, prefix, name, size) in [
            ("ace-step", "acestep-v15-turbo-Q", "ACE-Step 1.5 turbo", "2B DiT"),
            ("ace-step-xl", "acestep-v15-xl-turbo-Q", "ACE-Step 1.5 XL turbo", "4B DiT"),
        ] {
            let Some(dit) = gguf(&dir, prefix) else { continue };
            let file = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
            let card = json!({
                "id": id, "name": name, "engine": "acestep.cpp", "runtime": "ggml",
                "weights": {"dit": file(&dit), "lm": file(&lm), "size": size},
                "sample_rate": native::acestep::SAMPLE_RATE, "channels": 2,
                "min_seconds": 10, "max_seconds": 600, "length": "exact", "chooses_length": true,
                "vocals": true, "instrumental": true, "writes_lyrics": true,
                "languages": ACE_LANGUAGES.as_slice(),
                "prompt_languages": ACE_LANGUAGES.as_slice(),
                "license": "MIT", "commercial": true,
            });
            out.push(MusicModel { id: id.into(), card, kind: MusicKind::Ace(native::acestep::Files {
                lm: lm.clone(), text_encoder: text_encoder.clone(), dit, vae: vae.clone() }) });
        }
        out
    }

    /// Stable Audio 3 Medium in `models/music/sa3`, as sa3.cpp names its GGUF.
    fn sa3_models() -> Vec<MusicModel> {
        let dir = crate::setup::sa3_dir();
        if gguf(&dir, "stable-audio-3-medium-dit").is_none() || gguf(&dir, "t5gemma").is_none() {
            return vec![];
        }
        let card = json!({
            "id": "stable-audio-3", "name": "Stable Audio 3 Medium", "engine": "sa3.cpp", "runtime": "ggml",
            "weights": {"size": "1.5B DiT"},
            "sample_rate": 44100, "channels": 2,
            "min_seconds": 1, "max_seconds": 380, "length": "exact", "chooses_length": false, "default_seconds": 30,
            "vocals": false, "instrumental": true, "writes_lyrics": false,
            "languages": [], "prompt_languages": ["en"],
            "license": "Stability AI Community (до $1 млн выручки в год); T5Gemma — Gemma Terms", "commercial": true,
        });
        vec![MusicModel { id: "stable-audio-3".into(), card,
                          kind: MusicKind::Sa3 { models: dir, variant: "medium".into() } }]
    }

    /// HeartMuLa in `models/music/heartmula`, as scripts/convert_heartmula.py
    /// and convert_heartcodec.py lay it out.
    fn heartmula_models() -> Vec<MusicModel> {
        let dir = crate::setup::heartmula_dir();
        let pick = |stem: &str| ["q8_0", "f16"].iter().map(|q| format!("{stem}.{q}.gguf")).find(|f| dir.join(f).is_file());
        let (Some(backbone), Some(decoder)) = (pick("heartmula_backbone"), pick("heartmula_decoder")) else { return vec![] };
        if !dir.join("heartcodec_dit.onnx").is_file() {
            return vec![];
        }
        let card = json!({
            "id": "heartmula", "name": "HeartMuLa-oss-3B", "engine": "heartmula", "runtime": "llama.cpp + onnxruntime",
            "weights": {"backbone": backbone, "decoder": decoder, "size": "3B"},
            "sample_rate": native::heartmula::SAMPLE_RATE, "channels": 2,
            "min_seconds": 10, "max_seconds": 240, "length": "ceiling", "chooses_length": true, "default_seconds": 120,
            "vocals": true, "instrumental": false, "writes_lyrics": false, "needs_lyrics": true,
            "languages": ACE_LANGUAGES.as_slice(), "prompt_languages": ["en"],
            "license": "Apache-2.0", "commercial": true,
        });
        vec![MusicModel { id: "heartmula".into(), card,
                          kind: MusicKind::HeartMuLa(native::heartmula::Files { dir, backbone, decoder }) }]
    }

    pub fn get(&self, id: &str) -> Option<&MusicModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Cards for `/v1/music/models`, with what is in memory now.
    pub async fn cards(&self) -> Vec<Value> {
        let loaded = self.loaded.lock().await.as_ref().map(|(id, _)| id.clone());
        self.models.iter().map(|m| {
            let mut c = m.card.clone();
            c["default"] = json!(self.default.as_deref() == Some(m.id.as_str()));
            c["loaded"] = json!(loaded.as_deref() == Some(m.id.as_str()));
            c
        }).collect()
    }

    /// The model, loaded; another one in memory is dropped first.
    async fn model(&self, id: &str) -> Result<Loaded, HostError> {
        let mut slot = self.loaded.lock().await;
        if let Some((have, m)) = slot.as_ref() {
            if have == id {
                return Ok(m.clone());
            }
        }
        *slot = None; // прежняя модель освобождает память до загрузки новой
        let kind = self.get(id).ok_or_else(|| err("unsupported", format!("no music model '{id}'")))?.kind.clone();
        let lib = native::lib_dir().map_err(|e| err("internal", e.to_string()))?;
        let keep = music_keep_loaded();
        let m = tokio::task::spawn_blocking(move || match kind {
            MusicKind::Ace(files) => native::acestep::AceStep::load(&lib, &files, keep).map(|m| Loaded::Ace(Arc::new(m))),
            MusicKind::Sa3 { models, variant } =>
                native::sa3::Sa3::load(&lib, &models, &variant, keep).map(|m| Loaded::Sa3(Arc::new(m))),
            MusicKind::HeartMuLa(files) =>
                native::heartmula::HeartMuLa::load(&files, native::use_gpu(true), keep).map(|m| Loaded::HeartMuLa(Arc::new(m))),
        })
        .await
        .map_err(|e| err("internal", e.to_string()))?
        .map_err(|e| err("internal", format!("{e:#}")))?;
        native::trim_heap();
        *slot = Some((id.to_string(), m.clone()));
        Ok(m)
    }

    /// One piece. `on_progress` gets each step and says whether to go on.
    pub async fn generate(&self, id: &str, ask: MusicAsk,
                          mut on_progress: impl FnMut(MusicStep) -> bool + Send + 'static)
                          -> Result<MusicPiece, Stop> {
        let m = self.model(id).await.map_err(Stop::Failed)?;
        let default_seconds = self.get(id).and_then(|m| m.card["default_seconds"].as_f64()).unwrap_or(30.0);
        let piece = tokio::task::spawn_blocking(move || match m {
            Loaded::Ace(m) => {
                use native::acestep::Stage;
                m.generate(&ace_request(&ask), |s, n| on_progress(match s {
                    Stage::Plan => MusicStep { stage: "planning", step: n, total: None },
                    // turbo-модели — 8 шагов DiT
                    Stage::Render => MusicStep { stage: "rendering", step: n, total: Some(8) },
                    Stage::Decode => MusicStep { stage: "decoding", step: n, total: None },
                }))
                .map(|o| o.map(|s| MusicPiece { audio: s.audio, channels: 2, sample_rate: s.sample_rate, plan: ace_plan(&s.plan) }))
            }
            Loaded::Sa3(m) => {
                use native::sa3::Stage;
                let seconds = ask.seconds.unwrap_or(default_seconds);
                m.generate(&ask.prompt, "", seconds, ask.seed.unwrap_or(-1), 0, 0.0, |s, step, total| {
                    let stage = match s {
                        Stage::Loading => "loading",
                        Stage::Encoding => "planning",
                        Stage::Sampling => "rendering",
                        Stage::Decoding => "decoding",
                        Stage::Done => return true,
                    };
                    on_progress(MusicStep { stage, step, total: (total > 0).then_some(total) })
                })
                .map(|o| o.map(|p| MusicPiece { audio: p.audio, channels: p.channels, sample_rate: p.sample_rate,
                                               plan: json!({"seed": ask.seed}) }))
            }
            Loaded::HeartMuLa(m) => {
                use native::heartmula::{FRAME_RATE, SAMPLE_RATE, Stage};
                // описание — теги через запятую, как их учили (assets/tags.txt у heartlib)
                let tags = ask.prompt.split(',').map(str::trim).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(",");
                let ceiling = ask.seconds.unwrap_or(default_seconds);
                let frames = (ceiling * FRAME_RATE).ceil() as usize;
                let seed = ask.seed.map(|s| s as u64).unwrap_or_else(|| {
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
                });
                m.generate(&tags, ask.lyrics.as_deref().unwrap_or(""), ceiling, seed, |s| on_progress(match s {
                    Stage::Frames(n) => MusicStep { stage: "rendering", step: n, total: Some(frames) },
                    Stage::Codec { window, windows } => MusicStep { stage: "decoding", step: window, total: Some(windows) },
                }))
                .map(|o| o.map(|audio| MusicPiece { audio, channels: 2, sample_rate: SAMPLE_RATE,
                                                   plan: json!({"seed": seed, "tags": tags}) }))
            }
        })
        .await
        .map_err(|e| Stop::Failed(err("internal", e.to_string())))?
        .map_err(|e| Stop::Failed(err("internal", format!("{e:#}"))))?;
        native::trim_heap();
        piece.ok_or(Stop::Cancelled)
    }
}

/// acestep.cpp's request. An instrumental is "[Instrumental]": so the model
/// was trained. Without lyrics and without that, its LM writes them.
fn ace_request(a: &MusicAsk) -> Value {
    let lyrics = if a.instrumental { "[Instrumental]" } else { a.lyrics.as_deref().unwrap_or("") };
    let mut r = json!({"caption": a.prompt, "lyrics": lyrics, "output_format": "wav32",
                       "vocal_language": if a.instrumental { "unknown" } else { a.language.as_deref().unwrap_or("") }});
    if let Some(s) = a.seconds {
        r["duration"] = json!(s);
    }
    if let Some(s) = a.seed {
        r["seed"] = json!(s);
        r["lm_seed"] = json!(s);
    }
    r
}

fn ace_plan(p: &Value) -> Value {
    json!({"bpm": p["bpm"], "key": p["keyscale"], "time_signature": p["timesignature"],
           "language": p["vocal_language"], "lyrics": p["lyrics"], "seed": p["seed"]})
}

const TURN_INFO: &str = r#"{"engine": "smart-turn", "model": "pipecat-ai/smart-turn-v3/smart-turn-v3.2-cpu.onnx",
    "loaded": true, "device": "cpu", "runtime": "onnxruntime", "sample_rate": 16000}"#;

impl Engines {
    /// In-process engines first; the Python host only if something is left for it.
    pub async fn start(python: &Path, server_dir: Option<&Path>) -> anyhow::Result<Self> {
        let tts = NativeTts::find();
        let stt = NativeStt::find();
        let (vad, turn) = native_listen();
        let all_native = tts.is_some() && stt.is_some() && vad.is_some() && turn.is_some();
        let mut info = json!({});
        let host = if all_native {
            eprintln!("voicy: all engines in-process, Python is not needed");
            info["device"] = json!(tts_device());
            info["cuda"] = json!(info["device"] == "cuda");
            None
        } else {
            let dir = server_dir.ok_or_else(|| anyhow::anyhow!(
                "some engines need the Python host (server/engines), and the repository is not found.\n\
                 To run them here instead: voicy setup"))?;
            if native::lib_dir().is_err() {
                eprintln!("voicy: движков в процессе нет — библиотеки не скачаны (voicy setup)");
            }
            eprintln!("voicy: engines via {}", python.display());
            let host = Host::spawn(python, dir).await?;
            let (i, _) = host.run("info", json!({}), &[]).await.map_err(|e| anyhow::anyhow!(e.message))?;
            info = i;
            Some(host)
        };
        if let Some(n) = &tts {
            info["tts"] = n.info();
            eprintln!("voicy: synthesis in-process (llama.cpp + onnxruntime), talker {}", n.variant);
        }
        if let Some(n) = &stt {
            info["stt"] = n.status();
            eprintln!("voicy: recognition in-process ({})", n.runtime());
        }
        if vad.is_some() {
            info["vad"] = json!({"engine": "silero", "sample_rate": 16000, "runtime": "onnxruntime"});
        }
        if turn.is_some() {
            info["turn"] = serde_json::from_str(TURN_INFO).expect("json");
        }
        let music = Music::find();
        for m in &music.models {
            eprintln!("voicy: music model {} in-process ({})", m.id, m.card["engine"].as_str().unwrap_or("?"));
        }
        Ok(Engines { host, info, tts, stt, vad, turn, music })
    }

    fn host(&self) -> Result<&Arc<Host>, HostError> {
        self.host.as_ref().ok_or_else(|| err("internal", "no engine for this"))
    }

    // ------------------------------------------------------------ сведения

    pub fn tts_rate(&self) -> u32 {
        self.info["tts"]["sample_rate"].as_u64().unwrap_or(24000) as u32
    }

    pub fn tts_name(&self) -> String {
        self.info["tts"]["engine"].as_str().unwrap_or_default().to_string()
    }

    pub fn needs_reference_text(&self) -> bool {
        self.info["tts"]["needs_reference_text"].as_bool().unwrap_or(true)
    }

    pub fn reference_seconds(&self, key: &str) -> (f64, f64) {
        let v = &self.info["tts"][key];
        (v[0].as_f64().unwrap_or(3.0), v[1].as_f64().unwrap_or(60.0))
    }

    pub fn stt_name(&self) -> String {
        self.info["stt"]["engine"].as_str().unwrap_or_default().to_string()
    }

    pub fn stt_features(&self) -> Vec<String> {
        self.info["stt"]["features"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default()
    }

    pub fn turn_rate(&self) -> u32 {
        self.info["turn"]["sample_rate"].as_u64().unwrap_or(16000) as u32
    }

    /// Refuse up front what the recogniser cannot do (base.require).
    pub fn require(&self, task: &str, prompt: bool, hotwords: bool, words: bool) -> Result<(), ApiError> {
        let have = self.stt_features();
        let mut missing: Vec<&str> = [("translate", task == "translate"), ("prompt", prompt),
                                      ("hotwords", hotwords), ("word_timestamps", words)]
            .into_iter()
            .filter(|(f, on)| *on && !have.iter().any(|h| h == f))
            .map(|(f, _)| f)
            .collect();
        missing.sort();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(ApiError::bad(format!("{} cannot do: {}", self.stt_name(), missing.join(", "))))
        }
    }

    pub async fn status(&self, kind: &str) -> Value {
        match kind {
            "tts" if self.tts.is_some() => return self.tts.as_ref().unwrap().status(),
            "stt" if self.stt.is_some() => return self.stt.as_ref().unwrap().status(),
            "turn" if self.turn.is_some() => return self.info["turn"].clone(),
            _ => {}
        }
        let Ok(host) = self.host() else { return self.info[kind].clone() };
        match host.run("status", json!({"kind": kind}), &[]).await {
            Ok((v, _)) => v,
            Err(e) => json!({"engine": self.info[kind]["engine"], "error": e.message}),
        }
    }

    pub async fn load(&self, kind: &str) -> Result<(), ApiError> {
        match kind {
            "tts" if self.tts.is_some() => return self.tts.as_ref().unwrap().get().await.map(|_| ()).map_err(Into::into),
            "stt" if self.stt.is_some() => return self.stt.as_ref().unwrap().get().await.map(|_| ()).map_err(Into::into),
            "turn" if self.turn.is_some() => return Ok(()),
            _ => {}
        }
        self.host()?.run("load", json!({"kind": kind}), &[]).await.map(|_| ()).map_err(Into::into)
    }

    // ---------------------------------------------------------------- синтез

    /// The engine checks the language before anything is queued.
    /// Whether the synthesis model reads U+0301 as a stress mark. The official
    /// weights mangle a word with it (docs/adr/0015), so marks go only to the
    /// model that was trained on them.
    pub fn stress_marks(&self) -> bool {
        self.tts.as_ref().is_some_and(|t| t.stress.is_some())
    }

    pub async fn language(&self, code: Option<&str>) -> Result<String, ApiError> {
        if self.tts.is_some() {
            let code = code.map(str::trim).filter(|c| !c.is_empty()).unwrap_or("ru");
            let low = code.to_lowercase();
            return match native::qwen::LANGUAGES.iter().find(|(iso, name, _)| *iso == low || *name == low) {
                Some((iso, _, _)) => Ok(iso.to_string()),
                None => {
                    let all: Vec<&str> = native::qwen::LANGUAGES.iter().map(|l| l.0).collect();
                    Err(err("unsupported", format!("language '{code}' is not supported by qwen3-tts; supported: {}",
                                                   all.join(", "))).into())
                }
            };
        }
        let (v, _) = self.host()?.run("tts.language", json!({"code": code}), &[]).await?;
        Ok(v["language"].as_str().unwrap_or("ru").to_string())
    }

    /// Synthesis. `on_progress` sees every progress event and returns false to stop.
    pub async fn speak(&self, args: Value, mut on_progress: impl FnMut(&Value) -> bool) -> Result<Speech, Stop> {
        if let Some(n) = &self.tts {
            return Self::speak_native(n, args, on_progress).await;
        }
        let mut call = self.host().map_err(Stop::Failed)?.call("tts.speak", args, &[]).await;
        let mut stopping = false;
        loop {
            match call.next().await {
                Msg::Event { data } => {
                    if !stopping && !on_progress(&data) {
                        stopping = true;
                        call.cancel();
                    }
                }
                Msg::Result { body, bin } => {
                    if stopping {
                        return Err(Stop::Cancelled);
                    }
                    return Ok(Speech {
                        audio: f32_from(&bin),
                        sample_rate: body["sample_rate"].as_u64().unwrap_or(24000) as u32,
                    });
                }
                Msg::Error(e) if e.kind == "cancelled" => return Err(Stop::Cancelled),
                Msg::Error(e) => return Err(Stop::Failed(e)),
            }
        }
    }

    async fn speak_native(n: &NativeTts, args: Value, mut on_progress: impl FnMut(&Value) -> bool)
                          -> Result<Speech, Stop> {
        let engine = n.get().await.map_err(Stop::Failed)?;
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<usize>();
        let flag = stop.clone();
        let text = args["text"].as_str().unwrap_or_default().to_string();
        let wav = PathBuf::from(args["ref_audio"].as_str().unwrap_or_default());
        let ref_text = args["ref_text"].as_str().unwrap_or_default().to_string();
        let lang = args["language"].as_str().and_then(native::qwen::language_id);
        let seed = args["seed"].as_i64().map(|s| s as u32);
        let job = tokio::task::spawn_blocking(move || {
            engine.speak(&text, &wav, &ref_text, lang, seed, |p| {
                let _ = tx.send(p.frames);
                !flag.load(Ordering::Relaxed)
            })
        });
        let frame = native::qwen::SAMPLES_PER_FRAME as f64 / native::qwen::SAMPLE_RATE as f64;
        while let Some(frames) = rx.recv().await {
            // кадр — ровно 80 мс готового звука
            if !stop.load(Ordering::Relaxed) && !on_progress(&json!({"produced": frames as f64 * frame})) {
                stop.store(true, Ordering::Relaxed);
            }
        }
        match job.await {
            Ok(Ok(Some(audio))) if !stop.load(Ordering::Relaxed) => Ok(Speech { audio, sample_rate: native::qwen::SAMPLE_RATE }),
            Ok(Ok(_)) => Err(Stop::Cancelled),
            Ok(Err(e)) => Err(Stop::Failed(err("internal", format!("{e:#}")))),
            Err(e) => Err(Stop::Failed(err("internal", e.to_string()))),
        }
    }

    // --------------------------------------------------------- распознавание

    /// Recognition of a file (`args["path"]`) or of 16 kHz samples (`audio`).
    /// `on_segment` sees (position, total, text) and returns false to stop.
    pub async fn transcribe(&self, args: Value, audio: &[u8],
                            mut on_segment: impl FnMut(&Value) -> bool) -> Result<Value, Stop> {
        if let Some(n) = &self.stt {
            return Self::transcribe_native(n, args, audio, on_segment).await;
        }
        let mut call = self.host().map_err(Stop::Failed)?.call("stt.transcribe", args, audio).await;
        let mut stopping = false;
        loop {
            match call.next().await {
                Msg::Event { data } => {
                    if !stopping && !on_segment(&data) {
                        stopping = true;
                        call.cancel();
                    }
                }
                Msg::Result { body, .. } if !stopping => return Ok(body),
                Msg::Result { .. } => return Err(Stop::Cancelled),
                Msg::Error(e) if e.kind == "cancelled" => return Err(Stop::Cancelled),
                Msg::Error(e) => return Err(Stop::Failed(e)),
            }
        }
    }

    async fn transcribe_native(n: &NativeStt, args: Value, audio: &[u8], mut on_segment: impl FnMut(&Value) -> bool)
                               -> Result<Value, Stop> {
        let engine = n.get().await.map_err(Stop::Failed)?;
        let samples = match args["path"].as_str() {
            Some(p) => {
                let raw = tokio::fs::read(p).await.map_err(|e| Stop::Failed(err("internal", e.to_string())))?;
                let mut x = tokio::task::spawn_blocking(move || native::decode::decode(&raw, 16000))
                    .await
                    .map_err(|e| Stop::Failed(err("internal", e.to_string())))?
                    .map_err(|e| Stop::Failed(err("bad_audio", e)))?;
                if let SttModel::Ct2(_) = &*engine {
                    native::fwhisper::through_s16(&mut x);
                }
                x
            }
            None => f32_from(audio),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        let flag = stop.clone();
        let job = tokio::task::spawn_blocking(move || {
            let strs = |k: &str| args[k].as_str().map(String::from);
            let hot: Option<Vec<String>> = args["hotwords"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect());
            let (lang, prompt) = (strs("language"), strs("prompt"));
            let req = native::whisper::Request {
                language: lang.as_deref(),
                prompt: prompt.as_deref(),
                hotwords: hot.as_deref(),
                temperature: args["temperature"].as_f64().unwrap_or(0.0) as f32,
                word_timestamps: args["word_timestamps"].as_bool().unwrap_or(false),
                translate: args["task"].as_str() == Some("translate"),
                live: args["live"].as_bool().unwrap_or(false),
                draft: args["draft"].as_bool().unwrap_or(false),
            };
            let mut on = |pos: f64, total: f64, text: &str| {
                let _ = tx.send(json!({"position": pos, "total": total, "text": text}));
                !flag.load(Ordering::Relaxed)
            };
            match &*engine {
                SttModel::Ct2(w) => w.transcribe(&samples, &req, &mut on),
                SttModel::Cpp(w) => w.transcribe(&samples, &req, &mut on),
            }
        });
        while let Some(seg) = rx.recv().await {
            if !stop.load(Ordering::Relaxed) && !on_segment(&seg) {
                stop.store(true, Ordering::Relaxed);
            }
        }
        match job.await {
            Ok(Ok(Some(tr))) if !stop.load(Ordering::Relaxed) => Ok(tr),
            Ok(Ok(_)) => Err(Stop::Cancelled),
            Ok(Err(e)) => Err(Stop::Failed(err("internal", format!("{e:#}")))),
            Err(e) => Err(Stop::Failed(err("internal", e.to_string()))),
        }
    }

    /// Any container a browser or a phone produces → mono float32 at `sr`:
    /// here first, then — for what only its ffmpeg reads — the Python host.
    pub async fn decode(&self, raw: &[u8], sr: u32) -> Result<Vec<f32>, ApiError> {
        let data = raw.to_vec();
        let native = tokio::task::spawn_blocking(move || native::decode::decode(&data, sr))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
        match (native, &self.host) {
            (Ok(x), _) => Ok(x),
            (Err(_), Some(host)) => {
                let (_, bin) = host.run("audio.decode", json!({"sample_rate": sr}), raw).await?;
                Ok(f32_from(&bin))
            }
            (Err(e), None) => Err(ApiError::bad(e)),
        }
    }

    // ----------------------------------------------------- слушающая сторона

    pub async fn vad_open(&self) -> Result<(VadHandle, usize), HostError> {
        if self.vad.is_some() {
            return Ok((VadHandle::Native(Default::default()), native::listen::VAD_FRAME));
        }
        let (v, _) = self.host()?.run("vad.open", json!({}), &[]).await?;
        Ok((VadHandle::Host(v["stream"].as_u64().unwrap_or(0)), v["frame"].as_u64().unwrap_or(512) as usize))
    }

    /// Probabilities of the frames completed by `x`, and how many samples wait.
    pub async fn vad_feed(&self, h: &VadHandle, x: &[f32]) -> Result<(Vec<f32>, usize), ApiError> {
        match h {
            VadHandle::Native(s) => {
                let vad = self.vad.as_ref().expect("native vad");
                let mut s = s.lock().unwrap();
                let probs = s.feed(vad, x).map_err(|e| ApiError::internal(format!("{e:#}")))?;
                Ok((probs, s.pending()))
            }
            VadHandle::Host(id) => {
                let (r, bin) = self.host()?.run("vad.feed", json!({"stream": id}), &f32_bytes(x)).await?;
                Ok((f32_from(&bin), r["pending"].as_u64().unwrap_or(0) as usize))
            }
        }
    }

    pub async fn vad_close(&self, h: &VadHandle) {
        if let (VadHandle::Host(id), Ok(host)) = (h, self.host()) {
            let _ = host.run("vad.close", json!({"stream": id}), &[]).await;
        }
    }

    /// Has the speaker finished? None if the detector failed.
    pub async fn turn_probability(&self, audio: Vec<f32>) -> Option<f64> {
        if let Some(t) = &self.turn {
            let t = t.clone();
            return tokio::task::spawn_blocking(move || t.probability(&audio)).await.ok()?.ok().map(|p| p as f64);
        }
        let (r, _) = self.host().ok()?.run("turn.probability", json!({}), &f32_bytes(&audio)).await.ok()?;
        r["probability"].as_f64()
    }
}
