//! Speech as it is synthesised, for a voice agent that should not go silent.
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
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
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

pub use voicy_core::segment::Segmenter;

pub const STREAM_FORMATS: [&str; 3] = ["pcm", "wav", "mp3"];

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
    /// распознать готовый звук и дочитать хвост, если модель закончила раньше текста
    pub verify: bool,
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

/// Меньше 50 мс звука — модель так и не начала речь: с отдельными образцами
/// голоса она сразу выдаёт конец. Отдавать это как успех нельзя — клиент
/// запишет WAV из одного заголовка и не узнает, что синтеза не было.
pub fn require_speech(samples: &[f32], sample_rate: u32, voice: &str) -> ApiResult<()> {
    if (samples.len() as f64) < sample_rate as f64 * 0.05 {
        return Err(ApiError::internal(format!(
            "voice '{voice}' produced no speech — retry, or re-add the voice from another passage")));
    }
    Ok(())
}

// ------------------------------------------------------------ недочитанный хвост

/// Слов в хвосте, с которых дочитываем, и тишина перед дочитанным, с.
const TAIL_MISSING: usize = 3;
const TAIL_PAUSE: f64 = 0.25;

fn word_spans(text: &str) -> Vec<(usize, usize, String)> {
    // слово с буквами, цифрами, ударением и дефисом внутри; форма для сверки — строчная, без знаков, ё → е
    let mut out = vec![];
    let mut start = None;
    let inside = |c: char| c.is_alphanumeric() || c == '\u{301}';
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (k, &(i, c)) in chars.iter().enumerate() {
        let hyphen = c == '-' && start.is_some() && chars.get(k + 1).is_some_and(|&(_, n)| n.is_alphanumeric());
        match (inside(c) || hyphen, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i, norm_word(&text[s..i])));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, text.len(), norm_word(&text[s..])));
    }
    out
}

fn norm_word(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).map(|c| if c == 'ё' { 'е' } else { c }).collect()
}

/// Pairs (i in a, j in b) of a longest common subsequence: which text word the
/// recogniser heard where.
fn lcs_pairs(a: &[String], b: &[String]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            d[i][j] = if a[i] == b[j] { d[i + 1][j + 1] + 1 } else { d[i + 1][j].max(d[i][j + 1]) };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, vec![]);
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if d[i + 1][j] >= d[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// The model sometimes stops before the text does: a fine-tuned voice in
/// experiments/29 left the last 5–42 words of a paragraph unread in a quarter of
/// its takes, and the sound was not short enough for a length rule to tell. Here
/// the finished sound is recognised; when its last TAIL_MISSING or more words
/// were not heard, the rest is synthesised once more, from the start of the
/// sentence the gap begins in (an intonation started mid-phrase is heard), the
/// take is cut there and the two are joined after a short pause.
/// `text` is the prepared text, as the synthesiser got it.
pub async fn read_tail(app: &App, text: &str, o: &Options, audio: Vec<f32>, sr: u32,
                       alive: impl Fn() -> bool) -> Result<Vec<f32>, ApiError> {
    let words = word_spans(text);
    if words.len() < TAIL_MISSING {
        return Ok(audio);
    }
    // распознавание берёт 16 кГц; линейной интерполяции для сверки слов хватает
    let n16 = (audio.len() as f64 * 16000.0 / sr as f64) as usize;
    let mut bytes = Vec::with_capacity(n16 * 4);
    for k in 0..n16 {
        let t = k as f64 * sr as f64 / 16000.0;
        let (i, f) = (t as usize, (t - t.floor()) as f32);
        let s = audio[i.min(audio.len() - 1)] * (1.0 - f) + audio[(i + 1).min(audio.len() - 1)] * f;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let tr = app
        .engines
        .transcribe(json!({"language": o.language, "word_timestamps": true, "temperature": 0.0}), &bytes, |_| alive())
        .await
        .map_err(|s| match s {
            Stop::Cancelled => ApiError::new(409, "cancelled"),
            Stop::Failed(e) => e.into(),
        })?;
    let heard: Vec<(f64, String)> = tr["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|s| s["words"].as_array().cloned().unwrap_or_default())
        .map(|w| (w["start"].as_f64().unwrap_or(0.0), norm_word(w["word"].as_str().unwrap_or_default())))
        .filter(|(_, w)| !w.is_empty())
        .collect();
    let ours: Vec<String> = words.iter().map(|w| w.2.clone()).collect();
    let theirs: Vec<String> = heard.iter().map(|h| h.1.clone()).collect();
    let pairs = lcs_pairs(&ours, &theirs);
    let last = pairs.last().map_or(0, |&(i, _)| i + 1);
    if words.len() - last < TAIL_MISSING {
        return Ok(audio);
    }
    // начало предложения, в котором пропуск: после последнего .!?… перед первым непрочитанным словом
    let boundary = text[..words[last].0]
        .char_indices()
        .filter(|&(_, c)| ".!?…".contains(c))
        .map(|(i, c)| i + c.len_utf8())
        .last()
        .unwrap_or(0);
    let k0 = words.iter().position(|w| w.0 >= boundary).unwrap_or(last).min(last);
    let (k0, cut) = match pairs.iter().find(|&&(i, _)| i == k0) {
        Some(&(_, j)) if k0 < last => (k0, Some(((heard[j].0 - 0.05).max(0.0) * sr as f64) as usize)),
        _ => (last, None),
    };
    log::info!("дочитываю хвост: не услышано {} слов из {}, заново с {}-го", words.len() - last, words.len(), k0 + 1);
    let rest = app.engines.speak(speak_args(&text[words[k0].0..], o), |_| alive()).await.map_err(|s| match s {
        Stop::Cancelled => ApiError::new(409, "cancelled"),
        Stop::Failed(e) => e.into(),
    })?;
    let mut out = audio;
    if let Some(c) = cut {
        out.truncate(c.min(out.len()));
    }
    out.extend(std::iter::repeat_n(0.0, (TAIL_PAUSE * sr as f64) as usize));
    out.extend(rest.audio);
    Ok(out)
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
    require_speech(&out.audio, out.sample_rate, &o.voice.name)?;
    let raw = if o.verify { read_tail(app, &text, o, out.audio, out.sample_rate, &alive).await? } else { out.audio };
    let audio = audio::stretch(raw, out.sample_rate, o.speed).await?;
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
        verify: req.verify,
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
            verify: false, // живой разговор: первый звук важнее
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

#[cfg(test)]
mod tests {
    use super::{lcs_pairs, require_speech, word_spans};

    #[test]
    fn tail_words_match_what_was_heard() {
        let w = word_spans("Её всё не́ было. Фёдор — кто-то, 3 раза.");
        let forms: Vec<&str> = w.iter().map(|x| x.2.as_str()).collect();
        assert_eq!(forms, ["ее", "все", "не", "было", "федор", "кто-то", "3", "раза"].map(|s| s.replace('-', "")));
        let heard: Vec<String> = ["ее", "все", "небыло", "федор"].map(String::from).to_vec();
        let ours: Vec<String> = forms.iter().map(|s| s.to_string()).collect();
        assert_eq!(lcs_pairs(&ours, &heard), [(0, 0), (1, 1), (4, 3)]);
    }

    #[test]
    fn no_speech_is_an_error_not_an_empty_file() {
        let err = require_speech(&[], 24000, "lv-olga10k").unwrap_err();
        assert_eq!(err.status.as_u16(), 500);
        assert!(err.message.contains("lv-olga10k"), "{}", err.message);
        // 40 мс — меньше порога, 60 мс — уже звук.
        assert!(require_speech(&vec![0.0; 960], 24000, "v").is_err());
        assert!(require_speech(&vec![0.0; 1440], 24000, "v").is_ok());
    }
}
