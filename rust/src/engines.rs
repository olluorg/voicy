//! Typed calls to the engines, and what they said about themselves at start.
//!
//! Most engines live in the Python host process (host.rs). Synthesis may run
//! here instead, without Python (native/qwen.rs): chosen when its libraries and
//! converted model are in the cache, or asked for with TTS_ENGINE=qwen3-tts-gguf.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};

use crate::errors::ApiError;
use crate::host::{Host, HostError, Msg, f32_from};

pub struct Engines {
    pub host: Arc<Host>,
    pub info: Value,
    native: Option<NativeTts>,
    vad: Option<Arc<crate::native::listen::Vad>>,
    turn: Option<Arc<crate::native::listen::Turn>>,
}

/// One live session's voice detector: in the host process or in this one.
pub enum VadHandle {
    Host(u64),
    Native(std::sync::Mutex<crate::native::listen::VadStream>),
}

/// Silero and Smart Turn in-process, when their models are in the cache and no
/// other engine is asked for — the same ONNX files the Python engines run.
fn native_listen() -> (Option<Arc<crate::native::listen::Vad>>, Option<Arc<crate::native::listen::Turn>>) {
    let Ok(lib) = crate::native::lib_dir() else { return (None, None) };
    if crate::native::init_onnx(&lib).is_err() {
        return (None, None);
    }
    let m = crate::native::cache_dir().join("models");
    let allowed = |var: &str, name: &str| std::env::var(var).map_or(true, |v| v.is_empty() || v == name);
    let vad = m.join("vad").join("silero_vad_v6.onnx");
    let turn = m.join("turn").join("smart-turn-v3.2-cpu.onnx");
    let vad = (allowed("VAD_ENGINE", "silero") && vad.is_file())
        .then(|| crate::native::listen::Vad::load(&vad).map(Arc::new).map_err(|e| eprintln!("voicy: vad: {e:#}")).ok())
        .flatten();
    let turn = (allowed("TURN_ENGINE", "smart-turn") && turn.is_file())
        .then(|| crate::native::listen::Turn::load(&turn).map(Arc::new).map_err(|e| eprintln!("voicy: turn: {e:#}")).ok())
        .flatten();
    (vad, turn)
}

/// Qwen3-TTS in-process: loaded on first use, as the Python engines are.
struct NativeTts {
    files: crate::native::qwen::Files,
    engine: tokio::sync::OnceCell<Arc<crate::native::qwen::Qwen>>,
    variant: String,
}

impl NativeTts {
    /// The converted model and the runtime libraries, if they are there.
    fn find() -> Option<NativeTts> {
        let wanted = std::env::var("TTS_ENGINE").unwrap_or_default();
        if !wanted.is_empty() && wanted != "qwen3-tts-gguf" {
            return None;
        }
        let dir = std::env::var_os("VOICY_TTS_GGUF_DIR").map(PathBuf::from).unwrap_or_else(|| {
            crate::native::cache_dir().join("models").join("qwen3-tts-12hz-1.7b-base-gguf")
        });
        // q5_k: та же разборчивость и то же сходство голоса, что у f16 и torch, при 2.4 ГБ (experiments/20)
        let variant = std::env::var("TTS_GGUF_TALKER").unwrap_or_else(|_| "q5_k".into());
        let files = crate::native::qwen::Files {
            talker: format!("qwen3_tts_talker.{variant}.gguf"),
            predictor: "qwen3_tts_predictor.q8_0.gguf".into(),
            dir,
        };
        let ready = files.dir.join(&files.talker).is_file() && crate::native::lib_dir().is_ok();
        if !ready && wanted == "qwen3-tts-gguf" {
            eprintln!("voicy: TTS_ENGINE=qwen3-tts-gguf, but no model in {} or no engine libraries", files.dir.display());
        }
        ready.then(|| NativeTts { files, engine: tokio::sync::OnceCell::new(), variant })
    }

    async fn get(&self) -> Result<Arc<crate::native::qwen::Qwen>, HostError> {
        self.engine
            .get_or_try_init(|| async {
                let files = crate::native::qwen::Files {
                    dir: self.files.dir.clone(),
                    talker: self.files.talker.clone(),
                    predictor: self.files.predictor.clone(),
                };
                tokio::task::spawn_blocking(move || crate::native::qwen::Qwen::load(&files, true))
                    .await
                    .map_err(|e| HostError { kind: "internal".into(), message: e.to_string() })?
                    .map(Arc::new)
                    .map_err(|e| HostError { kind: "internal".into(), message: format!("{e:#}") })
            })
            .await
            .cloned()
    }

    fn status(&self) -> Value {
        json!({"engine": "qwen3-tts", "model": format!("Qwen/Qwen3-TTS-12Hz-1.7B-Base ({})", self.variant),
               "runtime": "llama.cpp + onnxruntime", "loaded": self.engine.initialized(),
               "device": self.engine.get().map_or("cuda".into(), |e| e.device.clone())})
    }
}

fn unsupported(message: String) -> HostError {
    HostError { kind: "unsupported".into(), message }
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

impl Engines {
    pub async fn start(host: Arc<Host>) -> anyhow::Result<Self> {
        let (mut info, _) = host.run("info", json!({}), &[]).await.map_err(|e| anyhow::anyhow!(e.message))?;
        let native = NativeTts::find();
        if let Some(n) = &native {
            let mut tts = n.status();
            tts["sample_rate"] = json!(crate::native::qwen::SAMPLE_RATE);
            tts["needs_reference_text"] = json!(true);
            tts["reference_seconds"] = json!([3.0, 60.0]);
            tts["reference_best"] = json!([8.0, 14.0]);
            info["tts"] = tts;
            eprintln!("voicy: synthesis in-process (llama.cpp + onnxruntime), talker {}", n.variant);
        }
        let (vad, turn) = native_listen();
        if vad.is_some() {
            info["vad"] = json!({"engine": "silero", "sample_rate": 16000, "runtime": "onnxruntime"});
        }
        if turn.is_some() {
            info["turn"] = json!({"engine": "smart-turn", "model": "pipecat-ai/smart-turn-v3/smart-turn-v3.2-cpu.onnx",
                                  "loaded": true, "device": "cpu", "runtime": "onnxruntime", "sample_rate": 16000});
        }
        Ok(Engines { host, info, native, vad, turn })
    }

    pub async fn vad_open(&self) -> Result<(VadHandle, usize), HostError> {
        if self.vad.is_some() {
            return Ok((VadHandle::Native(Default::default()), crate::native::listen::VAD_FRAME));
        }
        let (v, _) = self.host.run("vad.open", json!({}), &[]).await?;
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
                let (r, bin) = self.host.run("vad.feed", json!({"stream": id}), &crate::host::f32_bytes(x)).await?;
                Ok((f32_from(&bin), r["pending"].as_u64().unwrap_or(0) as usize))
            }
        }
    }

    pub async fn vad_close(&self, h: &VadHandle) {
        if let VadHandle::Host(id) = h {
            let _ = self.host.run("vad.close", json!({"stream": id}), &[]).await;
        }
    }

    /// Has the speaker finished? None if the detector failed.
    pub async fn turn_probability(&self, audio: Vec<f32>) -> Option<f64> {
        if let Some(t) = &self.turn {
            let t = t.clone();
            return tokio::task::spawn_blocking(move || t.probability(&audio)).await.ok()?.ok().map(|p| p as f64);
        }
        let (r, _) = self.host.run("turn.probability", json!({}), &crate::host::f32_bytes(&audio)).await.ok()?;
        r["probability"].as_f64()
    }

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
        if let (Some(n), "tts") = (&self.native, kind) {
            return n.status();
        }
        if kind == "turn" && self.turn.is_some() {
            return self.info["turn"].clone();
        }
        match self.host.run("status", json!({"kind": kind}), &[]).await {
            Ok((v, _)) => v,
            Err(e) => json!({"engine": self.info[kind]["engine"], "error": e.message}),
        }
    }

    pub async fn load(&self, kind: &str) -> Result<(), ApiError> {
        if let (Some(n), "tts") = (&self.native, kind) {
            return n.get().await.map(|_| ()).map_err(Into::into);
        }
        if kind == "turn" && self.turn.is_some() {
            return Ok(());
        }
        self.host.run("load", json!({"kind": kind}), &[]).await.map(|_| ()).map_err(Into::into)
    }

    /// The engine checks the language before anything is queued.
    pub async fn language(&self, code: Option<&str>) -> Result<String, ApiError> {
        if self.native.is_some() {
            let code = code.map(str::trim).filter(|c| !c.is_empty()).unwrap_or("ru");
            return match crate::native::qwen::LANGUAGES.iter().find(|(iso, name, _)| {
                *iso == code.to_lowercase() || *name == code.to_lowercase()
            }) {
                Some((iso, _, _)) => Ok(iso.to_string()),
                None => {
                    let all: Vec<&str> = crate::native::qwen::LANGUAGES.iter().map(|l| l.0).collect();
                    Err(unsupported(format!("language '{code}' is not supported by qwen3-tts; supported: {}",
                                            all.join(", "))).into())
                }
            };
        }
        let (v, _) = self.host.run("tts.language", json!({"code": code}), &[]).await?;
        Ok(v["language"].as_str().unwrap_or("ru").to_string())
    }

    /// Synthesis. `on_progress` sees every progress event and returns false to stop.
    pub async fn speak(&self, args: Value, mut on_progress: impl FnMut(&Value) -> bool) -> Result<Speech, Stop> {
        if let Some(n) = &self.native {
            return Self::speak_native(n, args, on_progress).await;
        }
        let mut call = self.host.call("tts.speak", args, &[]).await;
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
        let lang = args["language"].as_str().and_then(crate::native::qwen::language_id);
        let seed = args["seed"].as_i64().map(|s| s as u32);
        let job = tokio::task::spawn_blocking(move || {
            engine.speak(&text, &wav, &ref_text, lang, seed, |p| {
                let _ = tx.send(p.frames);
                !flag.load(Ordering::Relaxed)
            })
        });
        let frame = crate::native::qwen::SAMPLES_PER_FRAME as f64 / crate::native::qwen::SAMPLE_RATE as f64;
        while let Some(frames) = rx.recv().await {
            // кадр — ровно 80 мс готового звука
            if !stop.load(Ordering::Relaxed) && !on_progress(&json!({"produced": frames as f64 * frame})) {
                stop.store(true, Ordering::Relaxed);
            }
        }
        match job.await {
            Ok(Ok(Some(audio))) if !stop.load(Ordering::Relaxed) => {
                Ok(Speech { audio, sample_rate: crate::native::qwen::SAMPLE_RATE })
            }
            Ok(Ok(_)) => Err(Stop::Cancelled),
            Ok(Err(e)) => Err(Stop::Failed(HostError { kind: "internal".into(), message: format!("{e:#}") })),
            Err(e) => Err(Stop::Failed(HostError { kind: "internal".into(), message: e.to_string() })),
        }
    }

    /// Recognition. `on_segment` sees (position, total, text) and returns false to stop.
    pub async fn transcribe(&self, args: Value, audio: &[u8],
                            mut on_segment: impl FnMut(&Value) -> bool) -> Result<Value, Stop> {
        let mut call = self.host.call("stt.transcribe", args, audio).await;
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

    /// Any container a browser or a phone produces → mono float32 at `sr`.
    pub async fn decode(&self, raw: &[u8], sr: u32) -> Result<Vec<f32>, ApiError> {
        let (_, bin) = self.host.run("audio.decode", json!({"sample_rate": sr}), raw).await?;
        Ok(f32_from(&bin))
    }
}
