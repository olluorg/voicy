//! Speech as it is synthesised, for a voice agent that should not go silent —
//! server/speak.py.
//!
//! The synthesiser returns a piece of text only once the whole of it is
//! generated, so the text is cut into pieces and each is sent as soon as it is
//! ready. The cuts go where a listener expects a pause: the first piece at the
//! first clause boundary (the first sound is what the listener waits for), the
//! rest at sentence ends; short sentences are joined, long ones cut at their
//! last clause boundary.
//!
//!   POST /v1/audio/speech with `stream_format` — "audio" sends bytes as they
//!   are made (pcm, wav, mp3), "sse" sends `speech.audio.delta` events.
//!
//!   WS /v1/audio/speech/stream — text in as an LLM writes it, audio out in
//!   pieces, cancel when the user interrupts.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use fancy_regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::App;
use crate::audio;
use crate::engines::Stop;
use crate::errors::{ApiError, ApiResult};
use crate::jobs::round;
use crate::textprep;
use crate::voices::Voice;

const MAX_CHUNK: usize = 200; // символов — длиннее режем по запятой
const MIN_CHUNK: usize = 25; // символов — короче склеиваем с соседом, если он уже есть
const FIRST_CLAUSE_WORDS: usize = 3; // первый кусок можно отрезать по запятой после стольких слов
pub const STREAM_FORMATS: [&str; 3] = ["pcm", "wav", "mp3"];

fn sentence() -> &'static Regex {
    // Конец предложения — знак, пробел и слово не со строчной буквы: так «т. е.»,
    // «см. выше» и «3.5» не режутся.
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"[.!?…]+["»)\]]*(?=\s+[^\sa-zа-яё])|\n"#).unwrap())
}

fn clause() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[,;:—–](?=\s)").unwrap())
}

fn byte_at(s: &str, chars: usize) -> usize {
    s.char_indices().nth(chars).map_or(s.len(), |(i, _)| i)
}

// --------------------------------------------------------------------- нарезка

/// Text in any pieces — tokens, lines, whole paragraphs; speakable chunks out.
#[derive(Default)]
pub struct Segmenter {
    buf: String,
    emitted: usize, // кусков отдано наружу
    cuts: usize,    // кусков отрезано, включая ещё не отданные
}

impl Segmenter {
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        self.take(false)
    }

    pub fn flush(&mut self) -> Vec<String> {
        self.take(true)
    }

    /// The whole text at once: short sentences join their neighbours on both sides.
    pub fn split(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        self.take(true)
    }

    pub fn reset(&mut self) {
        *self = Segmenter::default();
    }

    fn cut(&mut self, fin: bool) -> Option<String> {
        let m = sentence().find(&self.buf).ok().flatten().map(|m| (m.start(), m.end()));
        if self.cuts == 0 {
            // первый кусок — до первой запятой, если она раньше конца предложения
            let end = m.map_or(self.buf.len(), |(s, _)| s);
            let region = &self.buf[..end];
            let at = clause()
                .find_iter(region)
                .flatten()
                .find(|c| region[..c.start()].split_whitespace().count() >= FIRST_CLAUSE_WORDS)
                .map(|c| c.end());
            if let Some(at) = at {
                return Some(self.split_at(at));
            }
        }
        if let Some((_, end)) = m {
            return Some(self.split_at(end));
        }
        if self.buf.chars().count() > MAX_CHUNK {
            let head_end = byte_at(&self.buf, MAX_CHUNK);
            let head = &self.buf[..head_end];
            let at = clause()
                .find_iter(head)
                .flatten()
                .last()
                .map(|c| c.end())
                .or_else(|| head.rfind(' ').map(|i| i + 1))
                .filter(|&i| i > 0)
                .unwrap_or(head_end);
            return Some(self.split_at(at));
        }
        if fin && !self.buf.trim().is_empty() {
            return Some(self.split_at(self.buf.len()));
        }
        None
    }

    fn split_at(&mut self, at: usize) -> String {
        let rest = self.buf.split_off(at);
        let piece = std::mem::replace(&mut self.buf, rest);
        if !piece.trim().is_empty() {
            self.cuts += 1;
        }
        piece
    }

    fn take(&mut self, fin: bool) -> Vec<String> {
        let mut pieces = vec![];
        while let Some(p) = self.cut(fin) {
            if !p.trim().is_empty() {
                pieces.push(p.trim().to_string());
            }
        }
        let mut out: Vec<String> = vec![];
        for p in pieces {
            // короткое склеиваем со следующим, раз оно уже есть; первый кусок
            // потока не трогаем — он ради того, чтобы звук начался раньше
            let stream_first = self.emitted == 0 && out.len() == 1;
            match out.last_mut() {
                Some(last) if last.chars().count() < MIN_CHUNK && !stream_first => {
                    last.push(' ');
                    last.push_str(&p);
                }
                _ => out.push(p),
            }
        }
        self.emitted += out.len();
        out
    }
}

// ---------------------------------------------------------------------- синтез

#[derive(Clone)]
pub struct Options {
    pub voice: Voice,
    pub language: String,
    pub speed: f64,
    pub prepare: bool,
    pub legato: bool,
    /// знаки ударения; решено заранее: запрошены, язык русский, модель их понимает
    pub stress: bool,
    pub seed: Option<i64>,
}

pub fn pick_voice(app: &App, name: Option<&str>) -> ApiResult<Voice> {
    let v = match name.filter(|n| !n.is_empty()) {
        Some(n) => app.voices.get(n).ok_or_else(|| ApiError::bad(format!("unknown voice '{n}'")))?,
        None => app.voices.default().ok_or_else(|| ApiError::bad("no voice available — add one via /v1/voices"))?,
    };
    if app.engines.needs_reference_text() && v.text.is_empty() {
        return Err(ApiError::bad(format!("voice '{}' has no reference transcript", v.name)));
    }
    Ok(v)
}

pub fn speak_args(text: &str, o: &Options) -> Value {
    json!({"text": text, "ref_audio": o.voice.path.to_string_lossy(), "ref_text": o.voice.text,
           "language": o.language, "seed": o.seed})
}

/// One chunk. `alive()` is asked at every progress event; false stops the work.
pub async fn synthesize(app: &App, text: &str, o: &Options, alive: impl Fn() -> bool)
                        -> Result<(Vec<f32>, u32, f64), ApiError> {
    let text = textprep::for_speech(text, o.prepare, o.legato, o.stress).await;
    let started = Instant::now();
    let out = app.engines.speak(speak_args(&text, o), |_| alive()).await.map_err(|s| match s {
        Stop::Cancelled => ApiError::new(409, "cancelled"),
        Stop::Failed(e) => e.into(),
    })?;
    let audio = audio::stretch(out.audio, out.sample_rate, o.speed).await?;
    Ok((audio, out.sample_rate, started.elapsed().as_secs_f64()))
}

// ------------------------------------------------------------------------ HTTP

pub async fn http_stream(app: Arc<App>, req: crate::api::SpeechRequest) -> ApiResult<Response> {
    let fmt = req.response_format.to_lowercase();
    let sse = req.stream_format.as_deref() == Some("sse");
    if !sse && !STREAM_FORMATS.contains(&fmt.as_str()) {
        return Err(ApiError::bad(format!(
            "stream_format=audio supports {}; use sse for {fmt}", STREAM_FORMATS.join(", "))));
    }
    if req.input.trim().is_empty() {
        return Err(ApiError::bad("input is empty"));
    }
    let language = app.engines.language(req.language.as_deref()).await?;
    let o = Options {
        voice: pick_voice(&app, req.voice.as_deref())?,
        stress: req.stress && language == "ru" && app.engines.stress_marks(),
        language,
        speed: req.speed,
        prepare: req.prepare,
        legato: req.legato,
        seed: req.seed,
    };
    let chunks = Segmenter::default().split(&req.input);
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, std::io::Error>>(4);
    let rate = app.engines.tts_rate();
    let (voice, n) = (o.voice.name.clone(), chunks.len());
    let fmt2 = fmt.clone();
    tokio::spawn(async move {
        let (mut total, t0) = (0.0, Instant::now());
        if !sse && fmt2 == "wav" && tx.send(Ok(audio::wav_stream_header(rate))).await.is_err() {
            return;
        }
        for text in &chunks {
            // клиент ушёл — дальше не синтезируем: отмена уходит в движок
            let (a, sr, _) = match synthesize(&app, text, &o, || !tx.is_closed()).await {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e.message))).await;
                    return;
                }
            };
            total += a.len() as f64 / sr as f64;
            let data = if !sse && (fmt2 == "pcm" || fmt2 == "wav") {
                audio::to_pcm16(&a)
            } else {
                match audio::encode(&a, sr, &fmt2).await {
                    Ok((d, _)) => d,
                    Err(e) => {
                        let _ = tx.send(Err(std::io::Error::other(e.message))).await;
                        return;
                    }
                }
            };
            let item = if sse {
                let ev = json!({"type": "speech.audio.delta", "text": text,
                                "audio": base64::engine::general_purpose::STANDARD.encode(&data)});
                format!("data: {ev}\n\n").into_bytes()
            } else {
                data
            };
            if tx.send(Ok(item)).await.is_err() {
                return;
            }
        }
        if sse {
            let done = json!({"type": "speech.audio.done", "chunks": chunks.len(), "seconds": round(total, 2),
                              "generation_seconds": round(t0.elapsed().as_secs_f64(), 2)});
            let _ = tx.send(Ok(format!("data: {done}\n\n").into_bytes())).await;
        }
    });
    let media = if sse { "text/event-stream" } else if fmt == "pcm" { "audio/pcm" } else { audio::content_type(&fmt) };
    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Ok((
        [
            (header::CONTENT_TYPE, media.to_string()),
            (header::CACHE_CONTROL, "no-cache".into()),
            (header::HeaderName::from_static("x-voice"), voice),
            (header::HeaderName::from_static("x-chunks"), n.to_string()),
            (header::HeaderName::from_static("x-sample-rate"), rate.to_string()),
            (header::HeaderName::from_static("x-accel-buffering"), "no".into()),
        ],
        body,
    )
        .into_response())
}

// ------------------------------------------------------------------- WebSocket

#[derive(Deserialize)]
pub struct StreamQuery {
    voice: Option<String>,
    language: Option<String>,
    #[serde(default = "one")]
    speed: f64,
    sample_rate: Option<u32>,
    #[serde(default)]
    prepare: bool,
    #[serde(default)]
    legato: bool,
    #[serde(default = "yes")]
    stress: bool,
    seed: Option<i64>,
}

fn one() -> f64 {
    1.0
}

fn yes() -> bool {
    true
}

enum Out {
    Json(Value),
    Audio(Value, Vec<u8>),
    Close(Option<u16>),
}

enum Todo {
    Text(u64, String),
    End,
    Gone,
}

pub async fn socket(app: Arc<App>, q: StreamQuery, ws: WebSocket) {
    use futures::{SinkExt, StreamExt};
    let (mut sink, mut stream) = ws.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Out>();
    // один писатель — события и звук не перемешиваются
    let writer = tokio::spawn(async move {
        while let Some(o) = out_rx.recv().await {
            let r = match o {
                Out::Json(v) => sink.send(Message::Text(v.to_string().into())).await,
                Out::Audio(v, pcm) => match sink.send(Message::Text(v.to_string().into())).await {
                    Ok(()) => sink.send(Message::Binary(pcm.into())).await,
                    e => e,
                },
                Out::Close(code) => {
                    let frame = code.map(|c| axum::extract::ws::CloseFrame { code: c, reason: "".into() });
                    let _ = sink.send(Message::Close(frame)).await;
                    break;
                }
            };
            if r.is_err() {
                break;
            }
        }
    });
    let send = |v: Value| {
        let _ = out_tx.send(Out::Json(v));
    };

    let rate = q.sample_rate.unwrap_or_else(|| app.engines.tts_rate());
    let options = async {
        if !(8000..=48000).contains(&rate) {
            return Err(ApiError::bad("sample_rate must be 8000–48000"));
        }
        let language = app.engines.language(q.language.as_deref()).await?;
        Ok(Options {
            voice: pick_voice(&app, q.voice.as_deref())?,
            stress: q.stress && language == "ru" && app.engines.stress_marks(),
            language,
            speed: if q.speed != 1.0 { q.speed.clamp(0.8, 1.2) } else { 1.0 },
            prepare: q.prepare,
            legato: q.legato,
            seed: q.seed,
        })
    }
    .await;
    let o = match options {
        Ok(o) => o,
        Err(e) => {
            send(json!({"type": "error", "error": e.message}));
            let _ = out_tx.send(Out::Close(Some(1003)));
            let _ = writer.await;
            return;
        }
    };

    let epoch = Arc::new(AtomicU64::new(0)); // отмена увеличивает эпоху; старые куски выбрасываются
    let (todo_tx, mut todo_rx) = mpsc::unbounded_channel::<Todo>();
    send(json!({"type": "ready", "sample_rate": rate, "voice": o.voice.name}));

    let reader = {
        let (epoch, todo_tx, out_tx) = (epoch.clone(), todo_tx.clone(), out_tx.clone());
        tokio::spawn(async move {
            let mut seg = Segmenter::default();
            let put = |pieces: Vec<String>| {
                for p in pieces {
                    let _ = todo_tx.send(Todo::Text(epoch.load(Ordering::SeqCst), p));
                }
            };
            while let Some(msg) = stream.next().await {
                let raw = match msg {
                    Ok(Message::Text(t)) => t.to_string(),
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => continue,
                };
                // просто текст — тоже текст
                let cmd: Value = serde_json::from_str(&raw).unwrap_or_else(|_| json!({"type": "text", "text": raw}));
                match cmd["type"].as_str() {
                    Some("text") => {
                        let t = match &cmd["text"] {
                            Value::String(s) => s.clone(),
                            Value::Null => String::new(),
                            v => v.to_string(),
                        };
                        put(seg.push(&t));
                    }
                    Some("flush") => put(seg.flush()),
                    Some("cancel") => {
                        epoch.fetch_add(1, Ordering::SeqCst);
                        seg.reset();
                        let _ = out_tx.send(Out::Json(json!({"type": "cancelled"})));
                    }
                    Some("end") => {
                        put(seg.flush());
                        let _ = todo_tx.send(Todo::End);
                        return;
                    }
                    _ => {}
                }
            }
            epoch.fetch_add(1, Ordering::SeqCst);
            let _ = todo_tx.send(Todo::Gone);
        })
    };
    drop(todo_tx);

    let (mut index, mut seconds) = (0usize, 0.0f64);
    while let Some(item) = todo_rx.recv().await {
        let (mine, text) = match item {
            Todo::Gone => break,
            Todo::End => {
                send(json!({"type": "done", "chunks": index, "seconds": round(seconds, 2)}));
                let _ = out_tx.send(Out::Close(None));
                break;
            }
            Todo::Text(e, t) => (e, t),
        };
        if mine != epoch.load(Ordering::SeqCst) {
            continue;
        }
        send(json!({"type": "synthesizing", "index": index, "text": text}));
        let ep = epoch.clone();
        let (a, sr, took) = match synthesize(&app, &text, &o, || ep.load(Ordering::SeqCst) == mine).await {
            Ok(v) => v,
            Err(e) if e.status.as_u16() == 409 => continue,
            Err(e) => {
                send(json!({"type": "error", "error": e.message}));
                let _ = out_tx.send(Out::Close(Some(1011)));
                break;
            }
        };
        if mine != epoch.load(Ordering::SeqCst) {
            continue; // отменили, пока кодировали
        }
        let a = if rate != sr { audio::Resampler::whole(sr, rate, &a) } else { a };
        let dur = a.len() as f64 / rate as f64;
        seconds += dur;
        let ev = json!({"type": "audio", "index": index, "text": text, "seconds": round(dur, 2),
                        "synth_seconds": round(took, 2)});
        let _ = out_tx.send(Out::Audio(ev, audio::to_pcm16(&a)));
        index += 1;
    }
    epoch.fetch_add(1, Ordering::SeqCst);
    reader.abort();
    drop(out_tx);
    let _ = writer.await;
}
