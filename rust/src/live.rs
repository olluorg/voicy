//! Live speech over a WebSocket, shaped for a voice agent — server/live.py.
//!
//! The agent needs, in this order of urgency: the user started talking (stop
//! speaking), the user finished (answer, not a moment earlier), what they said.
//!
//! Speech start and pauses come from the voice detector, frame by frame. End of
//! turn is decided at each pause: the turn model listens and says whether it
//! sounds finished, and unless the text ends on a word a phrase cannot end on,
//! the turn ends there; otherwise only after `max_pause` of silence. Text is read
//! every `interval` seconds of new speech; a word is final once two readings
//! agree on it, and the window is cut at the end of the last agreed sentence.
//!
//! State lives under a mutex that is never held across a call to an engine;
//! after every such call the conditions are checked again, as the Python server
//! does after each await.

use std::sync::{Arc, Mutex};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;

use crate::App;
use crate::audio::Resampler;
use crate::contexts::Resolved;
use crate::errors::ApiError;
use crate::host::{f32_bytes, f32_from};
use crate::jobs::round;

const SR: usize = 16000;
const SPEECH_ON: f32 = 0.5;
const SPEECH_OFF: f32 = 0.35; // гистерезис, как у Silero по умолчанию
const START_SECONDS: f64 = 0.1; // столько речи подряд — речь началась
const PRE_ROLL: f64 = 0.5; // с до начала речи, которые идут в реплику
const MAX_WINDOW: f64 = 15.0; // с непрочитанного окна — дальше закрываем согласованное

// Слова, на которых русская фраза не кончается: предлоги, союзы, заминки.
const DANGLING: &str = "
в во на с со к ко по о об обо от из у за над под про для без до при через между перед
и а но или либо что чтобы как если когда потому поэтому так то ли будто хотя пока
это мой моя мое моё мои твой наш ваш их его её ее какой какая какое какие который
которая которое которые очень самый не ни
э ээ эээ эм мм м ну вот типа короче значит";

fn norm(word: &str) -> String {
    word.to_lowercase().replace('ё', "е").chars().filter(|c| c.is_alphanumeric() || *c == '_').collect()
}

/// Does the text stop where a Russian phrase cannot?
pub fn dangling(text: &str) -> bool {
    let text = text.trim();
    let Some(last) = text.chars().last() else { return false };
    // многоточие Whisper ставит на заминке
    if ",:;-—(«\"".contains(last) || text.ends_with("...") {
        return true;
    }
    let w = norm(text.split_whitespace().last().unwrap_or_default());
    DANGLING.split_whitespace().any(|d| d == w)
}

#[derive(Clone)]
struct Word {
    start: f64,
    end: f64,
    text: String,
}

#[derive(Clone, Default)]
struct Turn {
    seq: u64,
    active: bool,
    start: f64,
    speaking: bool,
    speech_run: u32,
    silence_run: u32,
    last_speech: f64,
    pause_id: u64,
    checked: Option<u64>,
    ending: bool,
    finals: Vec<String>,
    probability: Option<f64>,
}

struct State {
    audio: Vec<f32>, // 16 кГц, начиная с отсчёта `base`
    base: usize,
    total: usize,     // отсчётов с начала сессии
    commit_at: usize, // отсчёт, с которого текст ещё не окончателен
    hyp: Vec<Word>,   // прошлое чтение открытого окна
    last_partial: String,
    said: Vec<String>,
    turn: Turn,
    speech_since_pass: usize,
}

impl State {
    fn slice(&self, start: usize, end: usize) -> Vec<f32> {
        let a = start.saturating_sub(self.base).min(self.audio.len());
        let b = end.saturating_sub(self.base).min(self.audio.len());
        self.audio[a..b.max(a)].to_vec()
    }

    /// Keep only what reading and turn detection may still need.
    fn trim(&mut self) {
        let total = self.total as i64;
        let mut keep = (self.commit_at as i64).min(total - SR as i64);
        if self.turn.active {
            keep = keep.min(((self.turn.start * SR as f64) as i64).max(total - 8 * SR as i64));
        }
        if keep > self.base as i64 {
            let keep = keep as usize;
            self.audio.drain(..keep - self.base);
            self.base = keep;
        }
    }

    fn commit(&mut self, upto: usize, text: &str) {
        self.commit_at = upto;
        self.hyp.clear();
        self.last_partial.clear();
        if !text.is_empty() {
            self.turn.finals.push(text.into());
            self.said.push(text.into());
        }
    }
}

struct Session {
    app: Arc<App>,
    s: Mutex<State>,
    reading: tokio::sync::Mutex<()>, // чтение окна и закрытие реплики — по одному
    wake: Notify,
    out: mpsc::UnboundedSender<Option<(u16, Value)>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    language: Option<String>,
    ctx: Resolved,
    interval: f64,
    threshold: f64,
    min_pause: f64,
    max_pause: f64,
    vad: u64,
    frame: usize,
}

impl Session {
    fn send(&self, v: Value) {
        let _ = self.out.send(Some((0, v)));
    }

    fn spawn(self: &Arc<Self>, f: impl std::future::Future<Output = ()> + Send + 'static) {
        self.tasks.lock().unwrap().push(tokio::spawn(f));
    }

    async fn transcribe(&self, audio: &[f32], words: bool) -> Result<Value, ApiError> {
        let args = json!({"language": self.language, "prompt": self.ctx.prompt, "hotwords": self.ctx.hotwords,
                          "live": true, "draft": words, "word_timestamps": words});
        self.app.engines.transcribe(args, &f32_bytes(audio), |_| true).await.map_err(Into::into)
    }

    // ------------------------------------------------------------- звук

    async fn feed(self: &Arc<Self>, x: Vec<f32>) -> Result<(), ApiError> {
        if x.is_empty() {
            return Ok(());
        }
        {
            let mut s = self.s.lock().unwrap();
            s.audio.extend_from_slice(&x);
            s.total += x.len();
        }
        let (r, bin) = self.app.engines.host.run("vad.feed", json!({"stream": self.vad}), &f32_bytes(&x)).await?;
        self.endpoint(&f32_from(&bin), r["pending"].as_u64().unwrap_or(0) as usize);
        let mut s = self.s.lock().unwrap();
        if !s.turn.active {
            // тишина между репликами: окно чтения идёт следом, с запасом PRE_ROLL
            s.commit_at = s.commit_at.max(s.total.saturating_sub((PRE_ROLL * SR as f64) as usize));
            s.hyp.clear();
        }
        s.trim();
        Ok(())
    }

    fn endpoint(self: &Arc<Self>, probs: &[f32], pending: usize) {
        let frame = self.frame;
        let start_frames = ((START_SECONDS * SR as f64 / frame as f64).round() as u32).max(1);
        let mut s = self.s.lock().unwrap();
        let frames_done = (s.total - pending) / frame; // кадры идут без пропусков
        for (i, &p) in probs.iter().enumerate() {
            let at = (frames_done - probs.len() + i + 1) as f64 * frame as f64 / SR as f64; // конец кадра
            if p >= SPEECH_ON || (s.turn.speaking && p >= SPEECH_OFF) {
                s.turn.speech_run += 1;
                s.turn.silence_run = 0;
                s.turn.last_speech = at;
                s.speech_since_pass += frame;
                if !s.turn.speaking && s.turn.speech_run >= start_frames {
                    s.turn.speaking = true;
                    if !s.turn.active && !s.turn.ending {
                        let began = at - start_frames as f64 * frame as f64 / SR as f64;
                        let t = &mut s.turn;
                        t.active = true;
                        t.finals.clear();
                        t.probability = None;
                        t.start = (began - PRE_ROLL).max(0.0);
                        let start = (t.start * SR as f64) as usize;
                        s.commit_at = s.commit_at.min(start);
                        self.send(json!({"type": "speech_start", "at": round(began, 2)}));
                    }
                }
                continue;
            }
            let t = &mut s.turn;
            t.speech_run = 0;
            t.silence_run += 1;
            if t.speaking {
                t.speaking = false;
                t.pause_id += 1;
            }
            let pause = t.silence_run as f64 * frame as f64 / SR as f64;
            if t.active && !t.ending {
                if pause >= self.min_pause && t.checked != Some(t.pause_id) {
                    t.checked = Some(t.pause_id);
                    let (me, seq, pid, start) = (self.clone(), t.seq, t.pause_id, t.start);
                    self.spawn(async move { me.check_turn(seq, pid, start).await });
                }
                if pause >= self.max_pause {
                    t.ending = true;
                    let (me, seq) = (self.clone(), t.seq);
                    self.spawn(async move { me.end_turn(seq, "silence", None).await });
                }
            }
        }
        if s.speech_since_pass as f64 >= self.interval * SR as f64 {
            self.wake.notify_one();
        }
    }

    // ---------------------------------------------------------- реплика

    async fn check_turn(self: Arc<Self>, seq: u64, pause_id: u64, turn_start: f64) {
        let audio = {
            let s = self.s.lock().unwrap();
            s.slice((turn_start * SR as f64) as usize, s.total)
        };
        let rate = self.app.engines.turn_rate();
        let audio = if rate as usize != SR { Resampler::whole(SR as u32, rate, &audio) } else { audio };
        let Ok((r, _)) = self.app.engines.host.run("turn.probability", json!({}), &f32_bytes(&audio)).await else {
            return;
        };
        let p = r["probability"].as_f64().unwrap_or(0.0);
        let complete = p >= self.threshold;
        let now = self.s.lock().unwrap().total;
        self.send(json!({"type": "turn_check", "at": round(now as f64 / SR as f64, 2),
                         "probability": round(p, 3), "complete": complete}));
        let go = {
            let mut s = self.s.lock().unwrap();
            let t = &mut s.turn;
            let go = complete && t.seq == seq && t.pause_id == pause_id && !t.speaking && t.active && !t.ending;
            if go {
                t.probability = Some(p);
            }
            go
        };
        if go {
            self.end_turn(seq, "model", Some(pause_id)).await;
        }
    }

    /// Read what is left of the turn and close it. With reason "model" the text
    /// may still veto: a turn ending on "и" or "в" is not over.
    async fn end_turn(self: &Arc<Self>, seq: u64, reason: &str, pause_id: Option<u64>) {
        let _reading = self.reading.lock().await;
        let (upto, audio) = {
            let mut s = self.s.lock().unwrap();
            let t = &s.turn;
            if t.seq != seq || !t.active || (reason == "model" && (Some(t.pause_id) != pause_id || t.speaking)) {
                return;
            }
            s.turn.ending = true;
            let upto = s.total;
            (upto, s.slice(s.commit_at, upto))
        };
        let text = self.read(&audio).await;
        let mut s = self.s.lock().unwrap();
        if reason == "model" {
            if Some(s.turn.pause_id) != pause_id || s.turn.speaking {
                s.turn.ending = false; // заговорил, пока читали
                return;
            }
            let whole = s.turn.finals.iter().cloned().chain([text.clone()]).collect::<Vec<_>>().join(" ");
            let whole = whole.trim();
            if dangling(whole) {
                s.turn.ending = false;
                self.send(json!({"type": "turn_check", "at": round(s.total as f64 / SR as f64, 2),
                                 "probability": round(s.turn.probability.unwrap_or(0.0), 3),
                                 "complete": false, "veto": whole.split_whitespace().last()}));
                return;
            }
        }
        let start_s = s.commit_at as f64 / SR as f64;
        s.commit(upto, &text);
        if !text.is_empty() {
            self.send(json!({"type": "final", "text": text, "start": round(start_s, 2),
                             "end": round(upto as f64 / SR as f64, 2)}));
        }
        let t = s.turn.clone();
        self.send(json!({"type": "turn_end", "text": t.finals.join(" ").trim(), "start": round(t.start, 2),
                         "end": round(t.last_speech, 2), "reason": reason,
                         "probability": t.probability.map(|p| round(p, 3))}));
        // старый объект больше не закроют
        s.turn = Turn {
            seq: t.seq + 1,
            speaking: t.speaking,
            speech_run: t.speech_run,
            silence_run: t.silence_run,
            last_speech: t.last_speech,
            pause_id: t.pause_id,
            checked: Some(t.pause_id),
            ..Turn::default()
        };
    }

    // ------------------------------------------------------------- чтение

    async fn read(&self, audio: &[f32]) -> String {
        if audio.len() < (0.3 * SR as f64) as usize {
            return String::new();
        }
        match self.transcribe(audio, false).await {
            Ok(tr) => self.ctx.fix(tr["text"].as_str().unwrap_or_default().trim()),
            Err(_) => String::new(),
        }
    }

    /// One reading of the open window; commit what two readings agree on.
    async fn reader_pass(self: &Arc<Self>) {
        let _reading = self.reading.lock().await;
        let (seq, start, end, audio) = {
            let mut s = self.s.lock().unwrap();
            if !s.turn.active || s.turn.ending {
                return;
            }
            s.speech_since_pass = 0;
            let (start, end) = (s.commit_at, s.total);
            (s.turn.seq, start, end, s.slice(start, end))
        };
        if audio.len() < SR / 2 {
            return;
        }
        let Ok(tr) = self.transcribe(&audio, true).await else { return };
        let mut s = self.s.lock().unwrap();
        if s.commit_at != start || s.turn.seq != seq || !s.turn.active {
            return; // реплика закрылась, пока читали
        }
        let t0 = start as f64 / SR as f64;
        let mut words: Vec<Word> = tr["segments"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|seg| seg["words"].as_array().cloned().unwrap_or_default())
            .map(|w| Word {
                start: t0 + w["start"].as_f64().unwrap_or(0.0),
                end: t0 + w["end"].as_f64().unwrap_or(0.0),
                text: w["word"].as_str().unwrap_or_default().to_string(),
            })
            .collect();
        let agreed = s.hyp.iter().zip(&words).take_while(|(a, b)| norm(&a.text) == norm(&b.text)).count();
        let ends_sentence = |w: &Word| w.text.trim().ends_with(['.', '!', '?', '…']);
        let mut cut = (0..agreed).rev().find(|&i| ends_sentence(&words[i]));
        if cut.is_none() && agreed > 0 && (end - start) as f64 / SR as f64 > MAX_WINDOW {
            cut = Some(agreed - 1); // пауз и точек нет, а окно растёт
        }
        if let Some(cut) = cut {
            let nxt = words.get(cut + 1).map_or(words[cut].end + 0.1, |w| w.start);
            let at = end.min(((words[cut].end + nxt) / 2.0 * SR as f64) as usize);
            let text = self.ctx.fix(words[..=cut].iter().map(|w| w.text.as_str()).collect::<String>().trim());
            s.commit(at, &text);
            self.send(json!({"type": "final", "text": text, "start": round(t0, 2), "end": round(at as f64 / SR as f64, 2)}));
            words.drain(..=cut);
        }
        let partial = self.ctx.fix(words.iter().map(|w| w.text.as_str()).collect::<String>().trim());
        s.hyp = words;
        if !partial.is_empty() && partial != s.last_partial {
            s.last_partial = partial.clone();
            self.send(json!({"type": "partial", "text": partial, "start": round(s.commit_at as f64 / SR as f64, 2)}));
        }
    }

    async fn finish(self: &Arc<Self>) {
        loop {
            let pending: Vec<JoinHandle<()>> = std::mem::take(&mut *self.tasks.lock().unwrap());
            if pending.is_empty() {
                break;
            }
            for h in pending {
                let _ = h.await;
            }
        }
        let (active, seq) = {
            let s = self.s.lock().unwrap();
            (s.turn.active, s.turn.seq)
        };
        if active {
            self.end_turn(seq, "end", None).await;
        }
        let s = self.s.lock().unwrap();
        self.send(json!({"type": "done", "text": s.said.join(" ").trim(),
                         "duration": round(s.total as f64 / SR as f64, 2)}));
    }
}

#[derive(Deserialize)]
pub struct LiveQuery {
    language: Option<String>,
    prompt: Option<String>,
    hotwords: Option<String>,
    context: Option<String>,
    #[serde(default = "sr")]
    sample_rate: u32,
    #[serde(default = "d07")]
    interval: f64,
    #[serde(default = "d05")]
    turn_threshold: f64,
    #[serde(default = "d02")]
    min_pause: f64,
    #[serde(default = "d20")]
    max_pause: f64,
}

fn sr() -> u32 {
    SR as u32
}
fn d07() -> f64 {
    0.7
}
fn d05() -> f64 {
    0.5
}
fn d02() -> f64 {
    0.2
}
fn d20() -> f64 {
    2.0
}

pub async fn socket(app: Arc<App>, q: LiveQuery, ws: WebSocket) {
    use futures::{SinkExt, StreamExt};
    let (mut sink, mut stream) = ws.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Option<(u16, Value)>>();
    let writer = tokio::spawn(async move {
        while let Some(item) = out_rx.recv().await {
            match item {
                Some((0, v)) => {
                    if sink.send(Message::Text(v.to_string().into())).await.is_err() {
                        break;
                    }
                }
                Some((code, _)) => {
                    let _ = sink.send(Message::Close(Some(CloseFrame { code, reason: "".into() }))).await;
                    break;
                }
                None => {
                    let _ = sink.send(Message::Close(None)).await;
                    break;
                }
            }
        }
    });

    let setup = (|| {
        if !(8000..=192000).contains(&q.sample_rate) {
            return Err("sample_rate out of range".to_string());
        }
        let ctx = app
            .contexts
            .resolve(q.context.as_deref(), q.prompt.as_deref(), q.hotwords.as_deref())
            .map_err(|c| format!("unknown context '{c}'"))?;
        // окно читается со временем слов — без них согласовывать нечего
        app.engines
            .require("transcribe", ctx.prompt.is_some(), ctx.hotwords.is_some(), true)
            .map_err(|e| e.message)?;
        Ok(ctx)
    })();
    let ctx = match setup {
        Ok(c) => c,
        Err(e) => {
            let _ = out_tx.send(Some((0, json!({"type": "error", "error": e}))));
            let _ = out_tx.send(Some((1003, Value::Null)));
            let _ = writer.await;
            return;
        }
    };
    let (vad, frame) = match app.engines.host.run("vad.open", json!({}), &[]).await {
        Ok((v, _)) => (v["stream"].as_u64().unwrap_or(0), v["frame"].as_u64().unwrap_or(512) as usize),
        Err(e) => {
            let _ = out_tx.send(Some((0, json!({"type": "error", "error": e.message}))));
            let _ = out_tx.send(Some((1011, Value::Null)));
            let _ = writer.await;
            return;
        }
    };
    let session = Arc::new(Session {
        app: app.clone(),
        s: Mutex::new(State {
            audio: vec![],
            base: 0,
            total: 0,
            commit_at: 0,
            hyp: vec![],
            last_partial: String::new(),
            said: vec![],
            turn: Turn::default(),
            speech_since_pass: 0,
        }),
        reading: tokio::sync::Mutex::new(()),
        wake: Notify::new(),
        out: out_tx.clone(),
        tasks: Mutex::default(),
        language: q.language.filter(|l| !l.is_empty()),
        ctx,
        interval: q.interval.clamp(0.3, 5.0),
        threshold: q.turn_threshold.clamp(0.01, 0.99),
        min_pause: q.min_pause.clamp(0.1, 2.0),
        max_pause: q.max_pause.clamp(0.3, 10.0),
        vad,
        frame,
    });
    session.send(json!({"type": "ready", "sample_rate": q.sample_rate, "context": q.context,
                        "turn": {"threshold": session.threshold, "min_pause": session.min_pause,
                                 "max_pause": session.max_pause}}));
    // первая сессия ждёт загрузки весов; звук тем временем копится в сокете
    let _ = tokio::join!(app.engines.load("stt"), app.engines.load("turn"));

    let reader = {
        let s = session.clone();
        tokio::spawn(async move {
            loop {
                s.wake.notified().await;
                s.reader_pass().await;
            }
        })
    };
    // потоковый пересчёт частоты: кусочки по 100 мс не щёлкают на стыках
    let mut resampler = (q.sample_rate as usize != SR).then(|| Resampler::new(q.sample_rate, SR as u32));
    let mut ended = false;
    let mut failure: Option<String> = None;
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Binary(b)) => {
                let pcm: Vec<f32> =
                    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
                let x = match resampler.as_mut() {
                    Some(r) => r.chunk(&pcm),
                    None => pcm,
                };
                if let Err(e) = session.feed(x).await {
                    failure = Some(e.message);
                    break;
                }
            }
            Ok(Message::Text(t)) => {
                let cmd: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                if matches!(cmd["type"].as_str(), Some("end" | "stop")) {
                    ended = true;
                    break;
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            _ => {}
        }
    }
    reader.abort();
    if let Some(e) = failure {
        session.send(json!({"type": "error", "error": e}));
        let _ = out_tx.send(Some((1011, Value::Null)));
    } else if ended {
        session.finish().await;
        let _ = out_tx.send(None);
    }
    for h in std::mem::take(&mut *session.tasks.lock().unwrap()) {
        h.abort();
    }
    let _ = app.engines.host.run("vad.close", json!({"stream": vad}), &[]).await;
    drop(out_tx);
    drop(session);
    let _ = writer.await;
}
