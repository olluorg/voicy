//! Typed calls to the engines in the host process, and what they said about
//! themselves at start (the `info` operation).

use std::sync::Arc;

use serde_json::{Value, json};

use crate::errors::ApiError;
use crate::host::{Host, HostError, Msg, f32_from};

pub struct Engines {
    pub host: Arc<Host>,
    pub info: Value,
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
        let (info, _) = host.run("info", json!({}), &[]).await.map_err(|e| anyhow::anyhow!(e.message))?;
        Ok(Engines { host, info })
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
        match self.host.run("status", json!({"kind": kind}), &[]).await {
            Ok((v, _)) => v,
            Err(e) => json!({"engine": self.info[kind]["engine"], "error": e.message}),
        }
    }

    pub async fn load(&self, kind: &str) -> Result<(), ApiError> {
        self.host.run("load", json!({"kind": kind}), &[]).await.map(|_| ()).map_err(Into::into)
    }

    /// The engine checks the language before anything is queued.
    pub async fn language(&self, code: Option<&str>) -> Result<String, ApiError> {
        let (v, _) = self.host.run("tts.language", json!({"code": code}), &[]).await?;
        Ok(v["language"].as_str().unwrap_or("ru").to_string())
    }

    /// Synthesis. `on_progress` sees every progress event and returns false to stop.
    pub async fn speak(&self, args: Value, mut on_progress: impl FnMut(&Value) -> bool) -> Result<Speech, Stop> {
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
