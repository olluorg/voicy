//! Music: an instrumental, a bed under narration, a song to given lyrics.
//!
//! Several models at once, and the request picks one by `model` (docs/adr/0028):
//! they are good at different things, and choosing is the asker's business.
//! `GET /v1/music/models` lists them with what each can do; a request a model
//! cannot do is refused before the queue, not done badly.
//!
//! The request has the shape of ElevenLabs' `/v1/music` — `prompt`,
//! `music_length_ms`, `force_instrumental`, `model_id`, `output_format` — so a
//! client written for it works by changing the base URL. Lyrics, their
//! language and the seed are voicy's own fields; `composition_plan` is not in
//! the contract and is refused rather than ignored.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::FutureExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::App;
use crate::api::{create, submit, summary, wait};
use crate::audio;
use crate::engines::{MusicAsk, MusicStep, Stop};
use crate::errors::{ApiError, ApiResult, Json};
use crate::jobs::{Job, JobResult, WorkError, now, round};

pub const FORMATS: [&str; 6] = ["opus", "wav", "mp3", "flac", "aac", "pcm"];

#[derive(Deserialize)]
pub struct MusicRequest {
    /// описание: жанр, инструменты, настроение, голос
    pub prompt: String,
    /// текст песни с разметкой частей: [verse], [chorus]…
    pub lyrics: Option<String>,
    /// язык текста, ISO 639-1
    pub language: Option<String>,
    pub music_length_ms: Option<u64>,
    pub duration_seconds: Option<f64>,
    #[serde(default)]
    pub force_instrumental: bool,
    pub model: Option<String>,
    pub model_id: Option<String>,
    pub seed: Option<i64>,
    /// как у синтеза: opus, wav, mp3, flac, aac, pcm; иначе — `output_format`
    pub response_format: Option<String>,
    pub composition_plan: Option<Value>,
    pub webhook_url: Option<String>,
}

/// `?output_format=mp3_44100_128` — как у ElevenLabs: кодек, частота, битрейт.
#[derive(Deserialize, Default)]
pub struct MusicQuery {
    output_format: Option<String>,
}

/// Формат и частота, если её назвали.
fn format(req: &MusicRequest, q: &MusicQuery) -> ApiResult<(String, Option<u32>)> {
    let (fmt, rate) = match (&req.response_format, &q.output_format) {
        (Some(f), _) => (f.to_lowercase(), None),
        (None, Some(of)) => {
            let mut parts = of.split('_');
            let codec = parts.next().unwrap_or_default().to_lowercase();
            let rate = parts.next().map(|r| r.parse::<u32>().map_err(|_| ApiError::bad(format!("output_format: bad sample rate in '{of}'"))));
            (codec, rate.transpose()?)
        }
        // ElevenLabs по умолчанию отдаёт mp3
        (None, None) => ("mp3".into(), None),
    };
    if !FORMATS.contains(&fmt.as_str()) {
        return Err(ApiError::bad(format!("format '{fmt}' is not supported; {} are", FORMATS.join(", "))));
    }
    if rate.is_some_and(|r| !(8000..=48000).contains(&r)) {
        return Err(ApiError::bad("output_format: sample rate must be 8000–48000"));
    }
    Ok((fmt, rate))
}

fn param(mut e: ApiError, name: &str) -> ApiError {
    e.param = Some(name.into());
    e
}

/// The model the request names. `music_v1` and the like are ElevenLabs' own
/// names, which its clients send by default: they mean the default here.
fn model_of<'a>(app: &'a App, req: &MusicRequest) -> ApiResult<&'a crate::engines::MusicModel> {
    let music = &app.engines.music;
    let ids = || music.models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ");
    let Some(default) = music.default.as_deref() else {
        return Err(ApiError::new(501, "no music model on this server; install one: voicy setup music"));
    };
    let asked = req.model.as_deref().or(req.model_id.as_deref()).filter(|m| !m.is_empty() && !m.starts_with("music_v"));
    let id = asked.unwrap_or(default);
    music.get(id).ok_or_else(|| param(ApiError::bad(format!("no music model '{id}'; there are: {}", ids())), "model"))
}

/// What the request asks of the model, checked against its card.
struct Checked {
    model: String,
    ask: MusicAsk,
    /// длина ровно заказанная; иначе заказ — лишь потолок, конец решает модель
    exact: bool,
}

/// The language a text is written in, by its script, when the request did
/// not say: without it ACE-Step's LM picks one itself, and for «Советская
/// музыка про деревню» it wrote Finnish lyrics. Only what the script tells
/// for sure; Latin says nothing — English, Spanish and Finnish share it.
fn language_of(text: &str) -> Option<&'static str> {
    let has = |f: &dyn Fn(char) -> bool| text.chars().any(f);
    if has(&|c| matches!(c, 'і' | 'ї' | 'є' | 'ґ' | 'І' | 'Ї' | 'Є' | 'Ґ')) {
        Some("uk")
    } else if has(&|c| matches!(c, '\u{0400}'..='\u{04FF}')) {
        Some("ru")
    } else if has(&|c| matches!(c, '\u{AC00}'..='\u{D7AF}')) {
        Some("ko")
    } else if has(&|c| matches!(c, '\u{3040}'..='\u{30FF}')) {
        Some("ja")
    } else if has(&|c| matches!(c, '\u{4E00}'..='\u{9FFF}')) {
        Some("zh")
    } else {
        None
    }
}

fn check(app: &App, req: &MusicRequest) -> ApiResult<Checked> {
    let m = model_of(app, req)?;
    let card = &m.card;
    let prompt = req.prompt.trim();
    if prompt.is_empty() {
        return Err(param(ApiError::bad("prompt is empty"), "prompt"));
    }
    if req.composition_plan.is_some() {
        return Err(param(ApiError::bad(format!("{} cannot do: composition_plan; give prompt and lyrics", m.id)),
                         "composition_plan"));
    }
    let lyrics = req.lyrics.as_deref().map(str::trim).filter(|l| !l.is_empty());
    if lyrics.is_some() && req.force_instrumental {
        return Err(ApiError::bad("lyrics and force_instrumental together: an instrumental has no lyrics"));
    }
    if lyrics.is_none() && !req.force_instrumental && card["needs_lyrics"] == json!(true) {
        return Err(param(ApiError::bad(format!("{} sings given lyrics only and writes none; give lyrics", m.id)), "lyrics"));
    }
    if lyrics.is_some() && card["vocals"] != json!(true) {
        return Err(param(ApiError::bad(format!("{} does not sing; it makes instrumentals only", m.id)), "lyrics"));
    }
    let prompt_ok = |l: &str| card["prompt_languages"].as_array().is_some_and(|a| a.iter().any(|x| x == &json!(l)));
    if let Some(l) = language_of(prompt).filter(|l| !prompt_ok(l)) {
        return Err(param(ApiError::bad(format!("{} understands descriptions in English only, not '{l}'; \
                                                describe the music in English", m.id)), "prompt"));
    }
    if req.force_instrumental && card["instrumental"] != json!(true) {
        return Err(param(ApiError::bad(format!("{} cannot make an instrumental: it always sings", m.id)),
                         "force_instrumental"));
    }
    let language = match req.language.as_deref().map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty()) {
        Some(l) => {
            let known = card["languages"].as_array().is_some_and(|a| a.iter().any(|x| x == &json!(l)));
            if !known {
                return Err(param(ApiError::bad(format!("{} does not sing in '{l}'", m.id)), "language"));
            }
            Some(l)
        }
        // текст песни говорит о языке вернее описания
        None => lyrics.and_then(language_of).or_else(|| language_of(prompt)).map(String::from)
            .filter(|l| card["languages"].as_array().is_some_and(|a| a.iter().any(|x| x == &json!(l)))),
    };
    let seconds = match (req.music_length_ms, req.duration_seconds) {
        (Some(ms), _) => Some(ms as f64 / 1000.0),
        (None, s) => s,
    };
    // модель, которая длину сама не выбирает, делает свою длину по умолчанию
    let seconds = seconds.or_else(|| (card["chooses_length"] != json!(true)).then(|| card["default_seconds"].as_f64()).flatten());
    if let Some(s) = seconds {
        let (lo, hi) = (card["min_seconds"].as_f64().unwrap_or(1.0), card["max_seconds"].as_f64().unwrap_or(600.0));
        if !(lo..=hi).contains(&s) {
            let name = if req.music_length_ms.is_some() { "music_length_ms" } else { "duration_seconds" };
            return Err(param(ApiError::bad(format!("{} makes {lo}–{hi} s of music", m.id)), name));
        }
    }
    // модель, которая не поёт, делает инструментал и без просьбы
    let instrumental = req.force_instrumental || card["vocals"] != json!(true);
    Ok(Checked { model: m.id.clone(), exact: card["length"] == json!("exact"), ask: MusicAsk {
        prompt: prompt.into(), lyrics: lyrics.map(String::from), instrumental,
        language: if instrumental { None } else { language }, seconds, seed: req.seed } })
}

/// Interleaved channels to another rate, channel by channel.
fn resample_ch(x: &[f32], ch: usize, from: u32, to: u32) -> Vec<f32> {
    let parts: Vec<Vec<f32>> = (0..ch)
        .map(|c| audio::Resampler::whole(from, to, &x.iter().skip(c).step_by(ch).copied().collect::<Vec<_>>()))
        .collect();
    let n = parts.iter().map(Vec::len).min().unwrap_or(0);
    (0..n).flat_map(|i| parts.iter().map(move |p| p[i])).collect()
}

/// The asked length, exactly. ACE-Step's length is the count of codes its LM
/// writes at 5 Hz, and the LM stops a code or four early: 74.2 s for 75,
/// 9.2 for 10. A short track gets silence at its end — the song ends a moment
/// earlier; a long one is cut with a short fade.
fn fit_length(x: &mut Vec<f32>, sr: u32, ch: usize, seconds: f64) {
    const FADE: f64 = 0.5;
    let frames = (seconds * sr as f64).round() as usize;
    if x.len() > ch * frames {
        x.truncate(ch * frames);
        let n = ((FADE * sr as f64) as usize).min(frames);
        for i in 0..n {
            let g = 1.0 - (i as f32 + 1.0) / n as f32;
            for c in 0..ch {
                x[ch * (frames - n + i) + c] *= g;
            }
        }
    }
    x.resize(ch * frames, 0.0);
}

/// Lossy codecs overshoot the peak of a loud master: a rock song at 0.95
/// came back from opus at 1.21, which clips on playback. So before opus, mp3
/// and aac the whole track is turned down to −1 dBFS when it peaks above it —
/// a change of level only, nothing is limited.
fn headroom(x: &mut [f32], fmt: &str) {
    const CEILING: f32 = 0.891; // −1 дБ
    if !matches!(fmt, "opus" | "mp3" | "aac") {
        return;
    }
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > CEILING {
        let g = CEILING / peak;
        x.iter_mut().for_each(|v| *v *= g);
    }
}

/// Progress as the job's events have it: the stage, and how far into it when
/// the model knows — diffusion steps it does; LM tokens and decoder tiles are
/// only counted.
fn event(s: &MusicStep) -> Value {
    let mut e = json!({"stage": s.stage, "step": s.step});
    if let Some(t) = s.total.filter(|&t| t > 0) {
        e["steps"] = json!(t);
        e["done"] = json!(round((s.step as f64 / t as f64).min(1.0), 3));
    }
    e
}

async fn submit_music(app: &Arc<App>, req: &MusicRequest, q: &MusicQuery, headers: &HeaderMap,
                          webhook_url: Option<&str>) -> ApiResult<(Arc<Job>, Checked)> {
    let c = check(app, req)?;
    let (fmt, rate) = format(req, q)?;
    let model = c.model.clone();
    let ask = c.ask.clone();
    let asked = c.ask.seconds.filter(|_| c.exact);
    let app2 = app.clone();
    let work: crate::jobs::Work = Box::new(move |job: Arc<Job>| {
        async move {
            let j = job.clone();
            let mut last = "";
            let song = app2
                .engines
                .music
                .generate(&model, ask, move |s| {
                    if j.cancelled() {
                        return false;
                    }
                    // токенов LM сотни: событие на каждый двадцать пятый
                    if last != s.stage || s.total.is_some() || s.step % 25 == 0 {
                        j.emit(event(&s));
                        last = s.stage;
                    }
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
            let (sr, ch) = (song.sample_rate, song.channels.max(1) as usize);
            let mut audio = song.audio;
            if let Some(s) = asked {
                fit_length(&mut audio, sr, ch, s);
            }
            let (mut x, sr) = match rate {
                Some(r) if r != sr => {
                    let a = audio;
                    (tokio::task::spawn_blocking(move || resample_ch(&a, ch, sr, r)).await
                        .map_err(|e| ApiError::internal(e.to_string()))?, r)
                }
                _ => (audio, sr),
            };
            let secs = round(x.len() as f64 / ch as f64 / sr as f64, 2);
            job.emit(json!({"stage": "encoding", "produced": secs, "expected": secs, "expected_exact": true}));
            headroom(&mut x, &fmt);
            let (data, ctype) = audio::encode_ch(&x, sr, ch as u16, &fmt).await?;
            let plan = song.plan;
            job.s.lock().unwrap().result = Some(JobResult::Music {
                data: Arc::new(data),
                content_type: ctype.into(),
                format: fmt.clone(),
                seconds: secs,
                model: model.clone(),
                plan: plan.clone(),
            });
            Ok(json!({"produced": secs, "expected": secs, "seconds": secs, "model": model}))
        }
        .boxed()
    });
    let job = create(app, "music", headers, webhook_url)?;
    Ok((submit(app, job, work)?, c))
}

type S = State<Arc<App>>;

fn license_of(app: &App, model: &str) -> String {
    app.engines.music.get(model).and_then(|m| m.card["license"].as_str().map(String::from)).unwrap_or_default()
}

/// `POST /v1/music` — the audio itself, as ElevenLabs answers.
pub async fn compose(State(app): S, Query(q): Query<MusicQuery>, headers: HeaderMap,
                     Json(req): Json<MusicRequest>) -> ApiResult<Response> {
    let (job, _) = submit_music(&app, &req, &q, &headers, None).await?;
    wait(&job).await?;
    let s = job.s.lock().unwrap();
    let Some(JobResult::Music { data, content_type, seconds, model, plan, .. }) = &s.result else {
        return Err(ApiError::internal("no audio"));
    };
    let (started, finished) = (s.started.unwrap_or(job.created), s.finished.unwrap_or(now()));
    let mut r = (
        [
            (header::CONTENT_TYPE, content_type.clone()),
            (header::HeaderName::from_static("x-job-id"), job.id.clone()),
            (header::HeaderName::from_static("x-model"), model.clone()),
            (header::HeaderName::from_static("x-model-license"), license_of(&app, model)),
            (header::HeaderName::from_static("x-audio-seconds"), format!("{seconds:.2}")),
            (header::HeaderName::from_static("x-generation-seconds"), format!("{:.2}", finished - started)),
            (header::HeaderName::from_static("x-queue-seconds"), format!("{:.2}", started - job.created)),
        ],
        Body::from(data.as_ref().clone()),
    )
        .into_response();
    // bpm и тональность, которые выбрала модель; текст — в /v1/jobs/{id}/result
    for (name, key) in [("x-music-bpm", "bpm"), ("x-music-key", "key")] {
        // заголовок — ASCII: «A♭ major» → «Ab major»
        if let Some(v) = plan[key].as_i64().map(|v| v.to_string()).or_else(|| plan[key].as_str().map(String::from))
            .map(|v| v.replace('♭', "b").replace('♯', "#"))
            .and_then(|v| v.parse().ok())
        {
            r.headers_mut().insert(header::HeaderName::from_static(name), v);
        }
    }
    Ok(r)
}

/// `POST /v1/jobs/music` — the same request, queued; the audio is at
/// `/v1/jobs/{id}/audio`, what the model planned at `/v1/jobs/{id}/result`.
pub async fn job(State(app): S, Query(q): Query<MusicQuery>, headers: HeaderMap,
                 Json(req): Json<MusicRequest>) -> ApiResult<Response> {
    let (job, c) = submit_music(&app, &req, &q, &headers, req.webhook_url.as_deref()).await?;
    let mut out = summary(&job, &app.queue);
    out["model"] = json!(c.model);
    if let Some(s) = c.ask.seconds {
        out["expected_seconds"] = json!(round(s, 1));
    }
    Ok((StatusCode::ACCEPTED, axum::Json(out)).into_response())
}

/// `GET /v1/music/models` — every music model here and what it can do.
pub async fn models(State(app): S) -> impl IntoResponse {
    axum::Json(json!({"default": app.engines.music.default, "models": app.engines.music.cards().await}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(f: Option<&str>) -> MusicRequest {
        MusicRequest {
            prompt: "x".into(), lyrics: None, language: None, music_length_ms: None, duration_seconds: None,
            force_instrumental: false, model: None, model_id: None, seed: None,
            response_format: f.map(String::from), composition_plan: None, webhook_url: None,
        }
    }

    #[test]
    fn output_format_in_elevenlabs_shape() {
        let q = |o: &str| MusicQuery { output_format: Some(o.into()) };
        assert_eq!(format(&req(None), &MusicQuery::default()).ok(), Some(("mp3".into(), None)));
        assert_eq!(format(&req(None), &q("mp3_44100_128")).ok(), Some(("mp3".into(), Some(44100))));
        assert_eq!(format(&req(Some("opus")), &q("pcm_16000")).ok(), Some(("opus".into(), None)));
        assert!(format(&req(None), &q("ulaw_8000")).is_err());
    }

    #[test]
    fn stereo_resample_keeps_channels_apart() {
        let x: Vec<f32> = (0..4800).flat_map(|_| [0.5f32, -0.5]).collect();
        let y = resample_ch(&x, 2, 48000, 44100);
        assert!((y.len() as i64 - 4410 * 2).abs() <= 4, "{}", y.len());
        let mid = &y[1000..1010];
        assert!(mid.chunks(2).all(|p| p[0] > 0.4 && p[1] < -0.4), "{mid:?}");
    }

    #[test]
    fn language_by_script() {
        assert_eq!(language_of("Советская музыка про деревню"), Some("ru"));
        assert_eq!(language_of("Їжак і пісня"), Some("uk"));
        assert_eq!(language_of("calm piano, 80 bpm"), None);
        assert_eq!(language_of("夜の歌"), Some("ja"));
        assert_eq!(language_of("사랑 노래"), Some("ko"));
    }

    #[test]
    fn length_is_what_was_asked() {
        for have in [9.2, 10.0, 10.7] {
            let mut x = vec![0.5f32; 2 * (have * 100.0) as usize];
            fit_length(&mut x, 100, 2, 10.0);
            assert_eq!(x.len(), 2000, "{have}");
        }
        let mut x = vec![0.5f32; 2 * 1070];
        fit_length(&mut x, 100, 2, 10.0);
        assert_eq!(x[1998], 0.0, "конец обрезанного затухает");
    }

    #[test]
    fn lossy_formats_get_headroom() {
        let mut x = vec![0.5f32, -0.95, 0.2];
        headroom(&mut x, "opus");
        assert!((x[1] + 0.891).abs() < 1e-6 && (x[0] - 0.5 * 0.891 / 0.95).abs() < 1e-6);
        let mut y = vec![0.5f32, -0.95];
        headroom(&mut y, "wav");
        assert_eq!(y, vec![0.5, -0.95]);
    }

    #[test]
    fn render_progress_is_a_fraction() {
        assert_eq!(event(&MusicStep { stage: "rendering", step: 4, total: Some(8) })["done"], json!(0.5));
        assert!(event(&MusicStep { stage: "planning", step: 100, total: None }).get("done").is_none());
    }
}
