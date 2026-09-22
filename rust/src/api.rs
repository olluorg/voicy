//! The routes — server/app.py and server/jobs.py.
//!
//! Drop-in compatibility first: anything that speaks to `/v1/audio/speech` and
//! `/v1/audio/transcriptions` works by changing the base URL. The compatible
//! routes are jobs too: they submit and wait on the caller's behalf, so a
//! synchronous call does not overtake queued ones, and its id is in X-Job-Id.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Multipart, Path, Query, Request, State, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{FutureExt, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::App;
use crate::audio;
use crate::engines::Stop;
use crate::errors::{ApiError, ApiResult, Json};
use crate::jobs::{Job, JobResult, Queue, Webhook, WorkError, now, round};
use crate::speak::{self, Options};
use crate::textprep;
use crate::transcripts;
use crate::voices;
use crate::{live, webhooks};

type S = State<Arc<App>>;

fn tts1() -> String {
    "tts-1".into()
}
fn wav() -> String {
    "wav".into()
}
fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
pub struct SpeechRequest {
    #[serde(default = "tts1")]
    #[allow(dead_code)]
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    #[serde(default = "wav")]
    pub response_format: String,
    #[serde(default = "one")]
    pub speed: f64,
    pub language: Option<String>,
    #[serde(default)]
    pub prepare: bool,
    #[serde(default)]
    pub legato: bool,
    pub seed: Option<i64>,
    pub stream_format: Option<String>,
    pub webhook_url: Option<String>,
}

// ------------------------------------------------------------------ формы

/// Form fields as text, files as (name, bytes) — what FastAPI's Form/File give.
struct Form {
    fields: HashMap<String, String>,
    files: HashMap<String, (String, Vec<u8>)>,
}

impl Form {
    async fn read(mut mp: Multipart) -> ApiResult<Form> {
        let mut form = Form { fields: HashMap::new(), files: HashMap::new() };
        while let Some(field) = mp.next_field().await.map_err(|e| ApiError::bad(e.body_text()))? {
            let name = field.name().unwrap_or_default().to_string();
            if let Some(file) = field.file_name().map(String::from) {
                let data = field.bytes().await.map_err(|e| ApiError::bad(e.body_text()))?;
                form.files.insert(name, (file, data.to_vec()));
            } else {
                let text = field.text().await.map_err(|e| ApiError::bad(e.body_text()))?;
                form.fields.insert(name, text);
            }
        }
        Ok(form)
    }

    fn file(&mut self, name: &str) -> ApiResult<(String, Vec<u8>)> {
        self.files.remove(name).ok_or_else(|| {
            let mut e = ApiError::bad(format!("{name}: Field required"));
            e.param = Some(name.into());
            e
        })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }

    fn opt(&self, name: &str) -> Option<String> {
        self.get(name).filter(|v| !v.is_empty()).map(String::from)
    }

    fn flag(&self, name: &str) -> ApiResult<bool> {
        match self.get(name).map(|v| v.to_lowercase()) {
            None => Ok(false),
            Some(v) if ["true", "1", "yes", "on", "t", "y"].contains(&v.as_str()) => Ok(true),
            Some(v) if ["false", "0", "no", "off", "f", "n", ""].contains(&v.as_str()) => Ok(false),
            Some(_) => Err(ApiError::bad(format!("{name}: Input should be a valid boolean"))),
        }
    }

    fn number(&self, name: &str, default: f64) -> ApiResult<f64> {
        match self.get(name) {
            None | Some("") => Ok(default),
            Some(v) => v.parse().map_err(|_| ApiError::bad(format!("{name}: Input should be a valid number"))),
        }
    }
}

fn base_url(app: &App, headers: &HeaderMap) -> String {
    if !app.public_url.is_empty() {
        return app.public_url.clone();
    }
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("localhost");
    format!("http://{host}/")
}

// ------------------------------------------------------------------ задания

pub fn summary(job: &Job, queue: &Queue) -> Value {
    let base = format!("{}/v1/jobs/{}", job.base_url, job.id);
    let s = job.s.lock().unwrap();
    let mut out = json!({
        "id": job.id, "kind": job.kind, "state": s.state,
        "position": if s.state == "queued" { json!(null) } else { json!(null) },
        "created": round(job.created, 3),
        "started": s.started.map(|t| round(t, 3)),
        "finished": s.finished.map(|t| round(t, 3)),
        "error": s.error, "progress": s.last,
        "links": {"self": base, "events": format!("{base}/events")},
    });
    let state = s.state.clone();
    let cancel = job.cancelled() && !crate::jobs::TERMINAL.contains(&state.as_str());
    if state == "done" {
        match &s.result {
            Some(JobResult::Speech { format, content_type, voice, seconds, .. }) => {
                out["result"] = json!({"voice": voice, "seconds": seconds, "format": format,
                                       "content_type": content_type});
                out["links"]["audio"] = json!(format!("{base}/audio"));
            }
            Some(JobResult::Transcript(r)) => {
                out["result"] = json!({"text": r["text"], "language": r["language"], "duration": r["duration"]});
                out["links"]["result"] = json!(format!("{base}/result"));
            }
            None => {}
        }
    }
    if let Some(h) = &s.webhook {
        out["webhook"] = json!({"url": h.url, "attempts": h.attempts, "delivered": h.delivered, "error": h.error});
    }
    drop(s);
    if state == "queued" {
        out["position"] = json!(queue.position(job));
    }
    if cancel {
        out["cancel_requested"] = json!(true);
    }
    out
}

fn create(app: &App, kind: &str, headers: &HeaderMap, webhook_url: Option<&str>) -> ApiResult<Arc<Job>> {
    if let Some(url) = webhook_url.filter(|u| !u.is_empty()) {
        webhooks::validate(url).map_err(ApiError::bad)?;
    }
    let job = app.registry.create(kind, &base_url(app, headers));
    if let Some(url) = webhook_url.filter(|u| !u.is_empty()) {
        job.s.lock().unwrap().webhook = Some(Webhook { url: url.into(), ..Webhook::default() });
    }
    Ok(job)
}

fn submit(app: &App, job: Arc<Job>, work: crate::jobs::Work) -> ApiResult<Arc<Job>> {
    if let Err(e) = app.queue.submit(job.clone(), work) {
        app.registry.discard(&job.id);
        return Err(ApiError::new(429, e).with_header("retry-after", "30"));
    }
    Ok(job)
}

/// For the synchronous routes: wait, then fail the way a plain call would.
async fn wait(job: &Arc<Job>) -> ApiResult<()> {
    job.wait().await;
    let s = job.s.lock().unwrap();
    match s.state.as_str() {
        "error" => Err(ApiError::new(s.error_status, s.error.clone().unwrap_or("failed".into()))),
        "cancelled" => Err(ApiError::new(409, "job was cancelled")),
        _ => Ok(()),
    }
}

/// Syllables per second of the default voice — only for the estimate of the
/// total, which is labelled as such.
fn estimate_seconds(text: &str) -> f64 {
    let syllables = text.to_lowercase().chars().filter(|c| "аеёиоуыэюяaeiouy".contains(*c)).count();
    (syllables as f64 / 4.5).max(0.5)
}

pub async fn submit_speech(app: &Arc<App>, req: &SpeechRequest, headers: &HeaderMap,
                           webhook_url: Option<&str>) -> ApiResult<(Arc<Job>, f64)> {
    if req.input.trim().is_empty() {
        return Err(ApiError::bad("input is empty"));
    }
    let voice = speak::pick_voice(app, req.voice.as_deref())?;
    let language = app.engines.language(req.language.as_deref()).await?;
    let text = if req.prepare || req.legato {
        textprep::prepared_text(&req.input, req.prepare, req.legato)
    } else {
        req.input.clone()
    };
    let expected = estimate_seconds(&text);
    let o = Options { voice, language, speed: req.speed, prepare: false, legato: false, seed: req.seed };
    let format = req.response_format.clone();
    let app2 = app.clone();
    let work: crate::jobs::Work = Box::new(move |job: Arc<Job>| {
        async move {
            let out = app2
                .engines
                .speak(speak::speak_args(&text, &o), |p| {
                    if job.cancelled() {
                        return false;
                    }
                    let mut ev = json!({"stage": "synthesis"});
                    if let Some(v) = p.get("produced") {
                        ev["produced"] = json!(round(v.as_f64().unwrap_or(0.0), 2));
                    }
                    if let Some(v) = p.get("done") {
                        ev["done"] = json!(round(v.as_f64().unwrap_or(0.0), 3));
                    }
                    // длительность, известная движку заранее, — факт, а не оценка
                    let total = p.get("total").and_then(Value::as_f64);
                    ev["expected"] = json!(round(total.unwrap_or(expected), 1));
                    ev["expected_exact"] = json!(total.is_some());
                    job.emit(ev);
                    true
                })
                .await
                .map_err(|s| match s {
                    Stop::Cancelled => WorkError::Cancelled,
                    Stop::Failed(e) => WorkError::Failed(e.into()),
                })?;
            if job.cancelled() {
                return Err(WorkError::Cancelled);
            }
            let audio = audio::stretch(out.audio, out.sample_rate, o.speed).await?;
            let seconds = round(audio.len() as f64 / out.sample_rate as f64, 2);
            job.emit(json!({"stage": "encoding", "produced": seconds, "expected": seconds, "expected_exact": true}));
            let (data, ctype) = audio::encode(&audio, out.sample_rate, &format).await?;
            job.s.lock().unwrap().result = Some(JobResult::Speech {
                data: Arc::new(data),
                content_type: ctype.into(),
                format: format.clone(),
                voice: o.voice.name.clone(),
                seconds,
            });
            Ok(json!({"produced": seconds, "expected": seconds, "seconds": seconds, "voice": o.voice.name}))
        }
        .boxed()
    });
    let job = create(app, "speech", headers, webhook_url)?;
    Ok((submit(app, job, work)?, expected))
}

pub struct Transcribe {
    pub raw: Vec<u8>,
    pub filename: String,
    pub task: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub temperature: f64,
    pub word_timestamps: bool,
    pub context: Option<String>,
    pub hotwords: Option<String>,
    pub webhook_url: Option<String>,
}

pub fn submit_transcription(app: &Arc<App>, t: Transcribe, headers: &HeaderMap) -> ApiResult<Arc<Job>> {
    if t.raw.is_empty() {
        return Err(ApiError::bad("file is empty"));
    }
    let ctx = app
        .contexts
        .resolve(t.context.as_deref(), t.prompt.as_deref(), t.hotwords.as_deref())
        .map_err(|c| ApiError::bad(format!("unknown context '{c}'")))?;
    app.engines.require(&t.task, ctx.prompt.is_some(), ctx.hotwords.is_some(), t.word_timestamps)?;
    let suffix = std::path::Path::new(&t.filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_else(|| ".wav".into());
    let app2 = app.clone();
    let task = t.task.clone();
    let work: crate::jobs::Work = Box::new(move |job: Arc<Job>| {
        async move {
            let dir = tempfile::tempdir().map_err(ApiError::from)?;
            let path = dir.path().join(format!("in{suffix}"));
            tokio::fs::write(&path, &t.raw).await.map_err(ApiError::from)?;
            let args = json!({"path": path.to_string_lossy(), "language": t.language, "prompt": ctx.prompt,
                              "hotwords": ctx.hotwords, "temperature": t.temperature,
                              "word_timestamps": t.word_timestamps, "task": task});
            let tr = app2
                .engines
                .transcribe(args, &[], |seg| {
                    if job.cancelled() {
                        return false;
                    }
                    job.emit(json!({"stage": "recognition",
                                    "position": round(seg["position"].as_f64().unwrap_or(0.0), 2),
                                    "total": round(seg["total"].as_f64().unwrap_or(0.0), 2),
                                    "text": ctx.fix(seg["text"].as_str().unwrap_or_default())}));
                    true
                })
                .await
                .map_err(|s| match s {
                    Stop::Cancelled => WorkError::Cancelled,
                    Stop::Failed(e) => WorkError::Failed(e.into()),
                })?;
            let mut result = transcripts::to_dict(&tr, &ctx);
            result["task"] = json!(task);
            let summary = json!({"position": tr["duration"], "total": tr["duration"], "language": tr["language"]});
            job.s.lock().unwrap().result = Some(JobResult::Transcript(result));
            Ok(summary)
        }
        .boxed()
    });
    let job = create(app, &t.task, headers, t.webhook_url.as_deref())?;
    submit(app, job, work)
}

fn get_job(app: &App, id: &str) -> ApiResult<Arc<Job>> {
    app.registry.get(id).ok_or_else(|| ApiError::new(404, "unknown job"))
}

fn finished(app: &App, id: &str) -> ApiResult<Arc<Job>> {
    let job = get_job(app, id)?;
    let s = job.s.lock().unwrap();
    match s.state.as_str() {
        "error" => Err(ApiError::new(s.error_status, s.error.clone().unwrap_or("failed".into()))),
        "cancelled" => Err(ApiError::new(410, "job was cancelled")),
        "done" => {
            drop(s);
            Ok(job)
        }
        _ => Err(ApiError::new(409, "not finished")),
    }
}

// --------------------------------------------------------------- OpenAI: TTS

pub async fn speech(State(app): S, headers: HeaderMap, Json(req): Json<SpeechRequest>) -> ApiResult<Response> {
    if let Some(sf) = &req.stream_format {
        if sf != "audio" && sf != "sse" {
            let mut e = ApiError::bad("stream_format: Input should be 'audio' or 'sse'");
            e.param = Some("stream_format".into());
            return Err(e);
        }
        return speak::http_stream(app, req).await;
    }
    let (job, _) = submit_speech(&app, &req, &headers, None).await?;
    wait(&job).await?;
    let s = job.s.lock().unwrap();
    let Some(JobResult::Speech { data, content_type, voice, seconds, .. }) = &s.result else {
        return Err(ApiError::internal("no audio"));
    };
    let (started, finished) = (s.started.unwrap_or(job.created), s.finished.unwrap_or(now()));
    Ok((
        [
            (header::CONTENT_TYPE, content_type.clone()),
            (header::HeaderName::from_static("x-job-id"), job.id.clone()),
            (header::HeaderName::from_static("x-voice"), voice.clone()),
            (header::HeaderName::from_static("x-audio-seconds"), format!("{seconds:.2}")),
            (header::HeaderName::from_static("x-generation-seconds"), format!("{:.2}", finished - started)),
            (header::HeaderName::from_static("x-queue-seconds"), format!("{:.2}", started - job.created)),
        ],
        Body::from(data.as_ref().clone()),
    )
        .into_response())
}

// --------------------------------------------------------------- OpenAI: STT

/// `stream=true` in OpenAI's shape: text deltas, then the whole text. The
/// recogniser's segments are the deltas; a client that leaves cancels the job.
fn delta_stream(app: Arc<App>, job: Arc<Job>) -> Response {
    struct Guard(Arc<App>, Arc<Job>);
    impl Drop for Guard {
        fn drop(&mut self) {
            if !self.1.done() {
                self.0.queue.cancel(&self.1); // клиент ушёл — работа никому не нужна
            }
        }
    }
    let rx = job.subscribe();
    let guard = Guard(app, job.clone());
    let events = futures::stream::unfold((rx, guard, true, false), |(mut rx, g, mut first, over)| async move {
        if over {
            return None;
        }
        loop {
            if g.1.done() {
                let s = g.1.s.lock().unwrap();
                let payload = match (&s.state[..], &s.result) {
                    ("done", Some(JobResult::Transcript(r))) => json!({"type": "transcript.text.done",
                        "text": r["text"], "language": r["language"], "duration": r["duration"], "job_id": g.1.id}),
                    _ => json!({"type": "error", "error": {"message": s.error.clone().unwrap_or(s.state.clone())}}),
                };
                drop(s);
                return Some((Ok::<_, Infallible>(Event::default().data(payload.to_string())), (rx, g, first, true)));
            }
            match rx.recv().await {
                Ok(e) if e["stage"] == "recognition" && e["text"].as_str().is_some_and(|t| !t.is_empty()) => {
                    let t = e["text"].as_str().unwrap();
                    let delta = if first { t.to_string() } else { format!(" {t}") };
                    first = false;
                    let payload = json!({"type": "transcript.text.delta", "delta": delta,
                                         "end": e["position"], "duration": e["total"]});
                    return Some((Ok(Event::default().data(payload.to_string())), (rx, g, first, false)));
                }
                Ok(_) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        }
    });
    sse(events, Some(&job.id))
}

fn sse(events: impl Stream<Item = Result<Event, Infallible>> + Send + 'static, job_id: Option<&str>) -> Response {
    let mut resp = Sse::new(events)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(25)).text("keep-alive"))
        .into_response();
    let h = resp.headers_mut();
    h.insert(header::CACHE_CONTROL, "no-cache".parse().unwrap());
    h.insert("x-accel-buffering", "no".parse().unwrap());
    if let Some(id) = job_id {
        h.insert("x-job-id", id.parse().unwrap());
    }
    resp
}

async fn transcription_form(app: &Arc<App>, mp: Multipart, headers: &HeaderMap, task: &str)
                            -> ApiResult<(Arc<Job>, Form)> {
    let mut form = Form::read(mp).await?;
    let (filename, raw) = form.file("file")?;
    let want_words = form.get("timestamp_granularities").is_some_and(|g| g.contains("word"))
        || form.get("response_format") == Some("verbose_json");
    let job = submit_transcription(app, Transcribe {
        raw,
        filename,
        task: task.into(),
        language: if task == "translate" { None } else { form.opt("language") },
        prompt: form.opt("prompt"),
        temperature: form.number("temperature", 0.0)?,
        word_timestamps: task != "translate" && want_words,
        context: if task == "translate" { None } else { form.opt("context") },
        hotwords: if task == "translate" { None } else { form.opt("hotwords") },
        webhook_url: None,
    }, headers)?;
    Ok((job, form))
}

pub async fn transcriptions(State(app): S, headers: HeaderMap, mp: Multipart) -> ApiResult<Response> {
    let (job, form) = transcription_form(&app, mp, &headers, "transcribe").await?;
    if form.flag("stream")? {
        return Ok(delta_stream(app, job));
    }
    rendered(&job, form.get("response_format")).await
}

pub async fn translations(State(app): S, headers: HeaderMap, mp: Multipart) -> ApiResult<Response> {
    let (job, form) = transcription_form(&app, mp, &headers, "translate").await?;
    rendered(&job, form.get("response_format")).await
}

async fn rendered(job: &Arc<Job>, fmt: Option<&str>) -> ApiResult<Response> {
    wait(job).await?;
    let tr = match &job.s.lock().unwrap().result {
        Some(JobResult::Transcript(r)) => r.clone(),
        _ => return Err(ApiError::internal("no transcript")),
    };
    let mut resp = transcripts::render(&tr, fmt);
    resp.headers_mut().insert("x-job-id", job.id.parse().unwrap());
    Ok(resp)
}

// ------------------------------------------------------------ OpenAI: models

pub async fn models() -> impl IntoResponse {
    let created = now() as i64;
    let data: Vec<Value> = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", "whisper-1"]
        .iter()
        .map(|id| json!({"id": id, "object": "model", "created": created, "owned_by": "voicy"}))
        .collect();
    axum::Json(json!({"object": "list", "data": data}))
}

// ---------------------------------------------------------------- расширения

pub async fn get_voices(State(app): S) -> impl IntoResponse {
    let default = app.voices.default().map(|v| v.name);
    let data: Vec<Value> = app.voices.list().iter().map(|v| v.as_json()).collect();
    axum::Json(json!({"object": "list", "default": default, "data": data}))
}

/// Register a reference clip. An empty transcript is filled in by the recogniser:
/// the same clip with a transcript cut mid-phrase measured 7.3% error against
/// 1.0% when the two agreed exactly.
pub async fn add_voice(State(app): S, mp: Multipart) -> ApiResult<Response> {
    let mut form = Form::read(mp).await?;
    let (_, raw) = form.file("file")?;
    let name = form.get("name").map(String::from).ok_or_else(|| ApiError::bad("name: Field required"))?;
    if !voices::name_ok(&name) {
        return Err(ApiError::bad(voices::NAME_RULE));
    }
    if app.voices.get(&name).is_some() && !form.flag("replace")? {
        return Err(ApiError::new(409, format!("voice '{name}' exists — pass replace=true to overwrite")));
    }
    if raw.is_empty() {
        return Err(ApiError::bad("file is empty"));
    }
    let sr = voices::SAMPLE_RATE;
    let audio = app.engines.decode(&raw, sr).await?;
    let seconds = audio.len() as f64 / sr as f64;
    let (lo, hi) = app.engines.reference_seconds("reference_best");
    let (lo_ok, hi_ok) = app.engines.reference_seconds("reference_seconds");
    // пределы — у движка синтеза: образец, годный одной модели, другой может не подойти
    if !(lo_ok..=hi_ok).contains(&seconds) {
        return Err(ApiError::bad(format!(
            "sample is {seconds:.1} s; {} needs {lo_ok:.0}–{hi_ok:.0} s, best {lo:.0}–{hi:.0}", app.engines.tts_name())));
    }
    let mut warnings = vec![];
    if !(lo..=hi).contains(&seconds) {
        warnings.push(format!("sample is {seconds:.1} s; clones are best from {lo:.0}–{hi:.0} s"));
    }
    let mut text = form.get("text").unwrap_or_default().trim().to_string();
    if text.is_empty() {
        let pcm16k = app.engines.decode(&raw, 16000).await?;
        let tr = app.engines.transcribe(json!({}), &crate::host::f32_bytes(&pcm16k), |_| true).await
            .map_err(ApiError::from)?;
        text = tr["text"].as_str().unwrap_or_default().trim().to_string();
    }
    let v = app.voices.add(&name, &audio::to_wav(&audio, sr), &text, form.get("note").unwrap_or_default())?;
    let mut out = v.as_json();
    out["seconds"] = json!(round(seconds, 2));
    out["warnings"] = json!(warnings);
    Ok(axum::Json(out).into_response())
}

pub async fn get_contexts(State(app): S) -> impl IntoResponse {
    let data: Vec<Value> = app.contexts.list().iter().map(|c| c.as_json()).collect();
    axum::Json(json!({"object": "list", "data": data}))
}

pub async fn get_context(State(app): S, Path(name): Path<String>) -> ApiResult<Response> {
    let c = app.contexts.get(&name).ok_or_else(|| ApiError::new(404, "unknown context"))?;
    Ok(axum::Json(c.as_json()).into_response())
}

pub async fn put_context(State(app): S, Path(name): Path<String>, Json(p): Json<Value>) -> ApiResult<Response> {
    let hot = p.get("hotwords").cloned().unwrap_or(json!([]));
    let rep = p.get("replacements").cloned().unwrap_or(json!({}));
    let hot = if hot.is_null() { json!([]) } else { hot };
    let rep = if rep.is_null() { json!({}) } else { rep };
    let (Some(hot), Some(rep)) = (hot.as_array(), rep.as_object()) else {
        return Err(ApiError::bad("hotwords must be a list, replacements an object"));
    };
    let s = |v: &Value| v.as_str().map(String::from).unwrap_or_else(|| v.to_string());
    let text = |k: &str| p.get(k).filter(|v| !v.is_null()).map(s).unwrap_or_default();
    let c = app
        .contexts
        .put(&name, &text("note"), &text("prompt"), hot.iter().map(s).collect(),
             rep.iter().map(|(k, v)| (k.clone(), s(v))).collect())
        .map_err(ApiError::bad)?;
    Ok(axum::Json(c.as_json()).into_response())
}

pub async fn delete_context(State(app): S, Path(name): Path<String>) -> ApiResult<Response> {
    if name == crate::contexts::BUILTIN {
        return Err(ApiError::bad(format!("'{name}' is built in")));
    }
    if !app.contexts.delete(&name) {
        return Err(ApiError::new(404, "unknown context"));
    }
    Ok(axum::Json(json!({"deleted": name})).into_response())
}

pub async fn prepare_text(Json(p): Json<Value>) -> impl IntoResponse {
    let text = p["text"].as_str().unwrap_or_default();
    let flag = |k: &str, d: bool| p.get(k).map_or(d, |v| v.as_bool().unwrap_or(!v.is_null() && v != &json!(0)));
    axum::Json(textprep::prepare(text, flag("dictionary", true), flag("legato", false)))
}

pub async fn health(State(app): S) -> impl IntoResponse {
    let (tts, stt, turn) = tokio::join!(app.engines.status("tts"), app.engines.status("stt"),
                                         app.engines.status("turn"));
    let mut stt = stt;
    stt["features"] = json!(app.engines.stt_features());
    let voices: Vec<String> = app.voices.list().into_iter().map(|v| v.name).collect();
    axum::Json(json!({
        "status": "ok", "tts": tts, "stt": stt, "turn": turn,
        "vad": {"engine": app.engines.info["vad"]["engine"]},
        "cuda": app.engines.info["cuda"], "auth": !app.keys.is_empty(),
        "device": app.engines.info["device"], "voices": voices,
        "queue": {"pending": app.queue.pending(), "max": app.queue.max_pending, "running": app.queue.running()},
        "server": "rust",
    }))
}

// ------------------------------------------------------------------ /v1/jobs

pub async fn job_speech(State(app): S, headers: HeaderMap, Json(req): Json<SpeechRequest>) -> ApiResult<Response> {
    let (job, expected) = submit_speech(&app, &req, &headers, req.webhook_url.as_deref()).await?;
    let mut out = summary(&job, &app.queue);
    out["expected_seconds"] = json!(round(expected, 1));
    Ok((StatusCode::ACCEPTED, axum::Json(out)).into_response())
}

pub async fn job_transcribe(State(app): S, headers: HeaderMap, mp: Multipart) -> ApiResult<Response> {
    let mut form = Form::read(mp).await?;
    let (filename, raw) = form.file("file")?;
    let task = form.get("task").unwrap_or("transcribe").to_string();
    if task != "transcribe" && task != "translate" {
        return Err(ApiError::bad("task must be transcribe or translate"));
    }
    let job = submit_transcription(&app, Transcribe {
        raw,
        filename,
        task,
        language: form.opt("language"),
        prompt: form.opt("prompt"),
        temperature: form.number("temperature", 0.0)?,
        word_timestamps: form.flag("word_timestamps")?,
        context: form.opt("context"),
        hotwords: form.opt("hotwords"),
        webhook_url: form.opt("webhook_url"),
    }, &headers)?;
    Ok((StatusCode::ACCEPTED, axum::Json(summary(&job, &app.queue))).into_response())
}

#[derive(Deserialize)]
pub struct ListQuery {
    state: Option<String>,
}

pub async fn list_jobs(State(app): S, Query(q): Query<ListQuery>) -> impl IntoResponse {
    let data: Vec<Value> = app
        .registry
        .all()
        .iter()
        .filter(|j| q.state.as_ref().is_none_or(|s| &j.state() == s))
        .map(|j| summary(j, &app.queue))
        .collect();
    axum::Json(json!({"object": "list", "data": data,
                      "queue": {"pending": app.queue.pending(), "max": app.queue.max_pending,
                                "running": app.queue.running()}}))
}

pub async fn get_job_route(State(app): S, Path(id): Path<String>) -> ApiResult<Response> {
    let job = get_job(&app, &id)?;
    Ok(axum::Json(summary(&job, &app.queue)).into_response())
}

pub async fn cancel_job(State(app): S, Path(id): Path<String>) -> ApiResult<Response> {
    let job = get_job(&app, &id)?;
    if !app.queue.cancel(&job) {
        return Err(ApiError::new(409, format!("job already {}", job.state())));
    }
    Ok(axum::Json(summary(&job, &app.queue)).into_response())
}

pub async fn job_events(State(app): S, Path(id): Path<String>) -> ApiResult<Response> {
    let job = get_job(&app, &id)?;
    let rx = job.subscribe(); // сначала подписка, потом взгляд на state
    let last = job.s.lock().unwrap().last.clone();
    let done = job.done();
    let first = futures::stream::once(async move { Ok::<_, Infallible>(Event::default().data(last.to_string())) });
    let rest = futures::stream::unfold((rx, done), |(mut rx, over)| async move {
        if over {
            return None;
        }
        loop {
            match rx.recv().await {
                Ok(e) => {
                    let end = e["stage"].as_str().is_some_and(|s| crate::jobs::TERMINAL.contains(&s));
                    return Some((Ok(Event::default().data(e.to_string())), (rx, end)));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return None,
            }
        }
    });
    Ok(sse(first.chain(rest), None))
}

pub async fn job_audio(State(app): S, Path(id): Path<String>) -> ApiResult<Response> {
    let job = finished(&app, &id)?;
    let s = job.s.lock().unwrap();
    let Some(JobResult::Speech { data, content_type, voice, seconds, .. }) = &s.result else {
        return Err(ApiError::bad("not a speech job — see /result"));
    };
    Ok((
        [
            (header::CONTENT_TYPE, content_type.clone()),
            (header::HeaderName::from_static("x-job-id"), job.id.clone()),
            (header::HeaderName::from_static("x-voice"), voice.clone()),
            (header::HeaderName::from_static("x-audio-seconds"), seconds.to_string()),
        ],
        Body::from(data.as_ref().clone()),
    )
        .into_response())
}

#[derive(Deserialize)]
pub struct ResultQuery {
    format: Option<String>,
}

pub async fn job_result(State(app): S, Path(id): Path<String>, Query(q): Query<ResultQuery>) -> ApiResult<Response> {
    let job = finished(&app, &id)?;
    let result = job.s.lock().unwrap().result.clone();
    match result {
        Some(JobResult::Speech { format, content_type, voice, seconds, .. }) => Ok(axum::Json(json!({
            "content_type": content_type, "format": format, "voice": voice, "seconds": seconds}))
        .into_response()),
        Some(JobResult::Transcript(tr)) => match q.format.as_deref() {
            None => Ok(axum::Json(tr).into_response()),
            Some(f) if transcripts::FORMATS.contains(&f) => Ok(transcripts::render(&tr, Some(f))),
            Some(_) => Err(ApiError::bad(format!("format must be one of {}", transcripts::FORMATS.join(", ")))),
        },
        None => Err(ApiError::internal("no result")),
    }
}

// ------------------------------------------------------------------ WebSocket

pub async fn speech_socket(State(app): S, Query(q): Query<speak::StreamQuery>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| speak::socket(app, q, socket))
}

pub async fn live_socket(State(app): S, Query(q): Query<live::LiveQuery>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| live::socket(app, q, socket))
}

pub async fn not_found(req: Request) -> ApiError {
    let _ = req;
    ApiError::new(404, "Not Found")
}

pub async fn method_not_allowed() -> ApiError {
    ApiError::new(405, "Method Not Allowed")
}
