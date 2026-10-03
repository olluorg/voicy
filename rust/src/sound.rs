//! Sounds that are not speech: a creaking door, a stream, wind.
//!
//! The request has the shape of ElevenLabs' `/v1/sound-generation` — `text`,
//! `duration_seconds`, `loop`, `output_format` — so a client written for it
//! works by changing the base URL, as OpenAI's do for speech. `prompt_influence`
//! is not in the engine contract yet and is refused rather than ignored.
//!
//! The engine makes one piece no longer than it can; anything longer, and any
//! loop, is made here from that piece (voicy-core's ambience): its tail is woven
//! into its head and the result repeated. Like every request, it is a job in
//! the one queue.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::FutureExt;
use serde::Deserialize;
use serde_json::{Value, json};
use voicy_core::ambience;

use crate::App;
use crate::api::{create, submit, summary, wait};
use crate::audio;
use crate::engines::Stop;
use crate::errors::{ApiError, ApiResult, Json};
use crate::jobs::{Job, JobResult, WorkError, now, round};

/// с — когда длина не задана. ElevenLabs в этом случае выбирает её сам;
/// модели voicy длину не выбирают.
const DEFAULT_SECONDS: f64 = 5.0;
const MIN_SECONDS: f64 = 0.5;
/// с — час фона: длиннее петля уже не нужна, а память под звук нужна.
const MAX_SECONDS: f64 = 3600.0;

#[derive(Deserialize)]
pub struct SoundRequest {
    pub text: String,
    pub duration_seconds: Option<f64>,
    #[serde(default, rename = "loop")]
    pub looped: bool,
    pub prompt_influence: Option<f64>,
    /// как у синтеза: opus, wav, mp3, flac, aac, pcm; иначе — `output_format`
    pub response_format: Option<String>,
    pub seed: Option<i64>,
    pub webhook_url: Option<String>,
}

/// `?output_format=mp3_44100_128` — как у ElevenLabs: кодек, частота, битрейт.
#[derive(Deserialize, Default)]
pub struct SoundQuery {
    output_format: Option<String>,
}

/// Формат и частота, если её назвали.
fn format(req: &SoundRequest, q: &SoundQuery) -> ApiResult<(String, Option<u32>)> {
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
    if !["opus", "wav", "mp3", "flac", "aac", "pcm"].contains(&fmt.as_str()) {
        return Err(ApiError::bad(format!("format '{fmt}' is not supported; opus, wav, mp3, flac, aac and pcm are")));
    }
    if rate.is_some_and(|r| !(8000..=48000).contains(&r)) {
        return Err(ApiError::bad("output_format: sample rate must be 8000–48000"));
    }
    Ok((fmt, rate))
}

fn cyrillic(text: &str) -> bool {
    text.chars().any(|c| matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё'))
}

/// How the requested length is made from what the engine can do in one call.
#[derive(Debug, PartialEq)]
struct Plan {
    /// с — сколько просить у движка
    piece: f64,
    /// с — сколько вплетать хвостом в начало; 0 — кусок отдаётся как есть
    fade: f64,
    /// сколько раз повторить петлю; 0 — до длины, с затуханием по краям
    repeats: usize,
}

/// A loop must itself loop: it is made of whole repeats of one piece. A long
/// sound that need not loop is a loop cut to length and faded at the edges.
fn plan(seconds: f64, looped: bool, max: f64) -> Plan {
    let fade = ambience::LOOP_FADE.min(max / 4.0);
    if !looped && seconds <= max {
        return Plan { piece: seconds, fade: 0.0, repeats: 1 };
    }
    if !looped {
        return Plan { piece: max, fade, repeats: 0 };
    }
    let longest = max - fade;
    let repeats = (seconds / longest).ceil().max(1.0) as usize;
    Plan { piece: seconds / repeats as f64 + fade, fade, repeats }
}

fn shape(x: Vec<f32>, sr: u32, seconds: f64, p: &Plan) -> Vec<f32> {
    if p.fade == 0.0 {
        return x;
    }
    let lp = ambience::seamless(&x, sr, p.fade);
    if p.repeats > 0 {
        return ambience::tile(&lp, lp.len() * p.repeats);
    }
    let mut y = ambience::tile(&lp, (seconds * sr as f64).round() as usize);
    ambience::fade_edges(&mut y, sr, ambience::EDGE_FADE);
    y
}

pub async fn submit_sound(app: &Arc<App>, req: &SoundRequest, q: &SoundQuery, headers: &HeaderMap,
                          webhook_url: Option<&str>) -> ApiResult<(Arc<Job>, f64)> {
    let Some(info) = app.engines.sound() else {
        return Err(ApiError::new(501, "no sound engine on this server; start it with SOUND_ENGINE"));
    };
    let name = info["engine"].as_str().unwrap_or("sound").to_string();
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return Err(ApiError::bad("text is empty"));
    }
    if req.prompt_influence.is_some() {
        return Err(ApiError::bad(format!("{name} cannot do: prompt_influence")));
    }
    let languages: Vec<&str> = info["languages"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if cyrillic(&text) && !languages.contains(&"ru") {
        return Err(ApiError::bad(format!("{name} understands descriptions in: {}; describe the sound in English",
                                         languages.join(", "))));
    }
    let seconds = req.duration_seconds.unwrap_or(DEFAULT_SECONDS);
    if !(MIN_SECONDS..=MAX_SECONDS).contains(&seconds) {
        let mut e = ApiError::bad(format!("duration_seconds must be {MIN_SECONDS}–{MAX_SECONDS}"));
        e.param = Some("duration_seconds".into());
        return Err(e);
    }
    let max = info["max_seconds"].as_f64().unwrap_or(10.0);
    let p = plan(seconds, req.looped, max);
    let (fmt, rate) = format(req, q)?;
    let args = json!({"prompt": text, "seconds": p.piece, "seed": req.seed});
    let app2 = app.clone();
    let work: crate::jobs::Work = Box::new(move |job: Arc<Job>| {
        async move {
            let out = app2
                .engines
                .sound_generate(args, |e| {
                    if job.cancelled() {
                        return false;
                    }
                    let mut ev = json!({"stage": "generation", "expected": round(p.piece, 1), "expected_exact": true});
                    if let Some(v) = e.get("done").and_then(Value::as_f64) {
                        ev["done"] = json!(round(v, 3));
                    }
                    if let Some(v) = e.get("produced").and_then(Value::as_f64) {
                        ev["produced"] = json!(round(v, 2));
                    }
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
            let sr = out.sample_rate;
            let x = tokio::task::spawn_blocking(move || {
                let y = shape(out.audio, sr, seconds, &p);
                match rate {
                    Some(r) if r != sr => (audio::Resampler::whole(sr, r, &y), r),
                    _ => (y, sr),
                }
            })
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?;
            let (x, sr) = x;
            let secs = round(x.len() as f64 / sr as f64, 2);
            job.emit(json!({"stage": "encoding", "produced": secs, "expected": secs, "expected_exact": true}));
            let (data, ctype) = audio::encode(&x, sr, &fmt).await?;
            job.s.lock().unwrap().result = Some(JobResult::Sound {
                data: Arc::new(data),
                content_type: ctype.into(),
                format: fmt.clone(),
                seconds: secs,
            });
            Ok(json!({"produced": secs, "expected": secs, "seconds": secs}))
        }
        .boxed()
    });
    let job = create(app, "sound", headers, webhook_url)?;
    Ok((submit(app, job, work)?, seconds))
}

type S = State<Arc<App>>;

/// `POST /v1/sound-generation` — the audio itself, as ElevenLabs answers.
pub async fn generate(State(app): S, Query(q): Query<SoundQuery>, headers: HeaderMap,
                      Json(req): Json<SoundRequest>) -> ApiResult<Response> {
    let (job, _) = submit_sound(&app, &req, &q, &headers, None).await?;
    wait(&job).await?;
    let s = job.s.lock().unwrap();
    let Some(JobResult::Sound { data, content_type, seconds, .. }) = &s.result else {
        return Err(ApiError::internal("no audio"));
    };
    let (started, finished) = (s.started.unwrap_or(job.created), s.finished.unwrap_or(now()));
    Ok((
        [
            (header::CONTENT_TYPE, content_type.clone()),
            (header::HeaderName::from_static("x-job-id"), job.id.clone()),
            (header::HeaderName::from_static("x-audio-seconds"), format!("{seconds:.2}")),
            (header::HeaderName::from_static("x-generation-seconds"), format!("{:.2}", finished - started)),
            (header::HeaderName::from_static("x-queue-seconds"), format!("{:.2}", started - job.created)),
        ],
        Body::from(data.as_ref().clone()),
    )
        .into_response())
}

/// `POST /v1/jobs/sound` — the same request, queued; the audio is at
/// `/v1/jobs/{id}/audio`.
pub async fn job(State(app): S, Query(q): Query<SoundQuery>, headers: HeaderMap,
                 Json(req): Json<SoundRequest>) -> ApiResult<Response> {
    let (job, expected) = submit_sound(&app, &req, &q, &headers, req.webhook_url.as_deref()).await?;
    let mut out = summary(&job, &app.queue);
    out["expected_seconds"] = json!(round(expected, 1));
    Ok((StatusCode::ACCEPTED, axum::Json(out)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_sound_is_one_piece() {
        assert_eq!(plan(4.0, false, 10.0), Plan { piece: 4.0, fade: 0.0, repeats: 1 });
    }

    #[test]
    fn long_sound_is_a_loop_cut_to_length() {
        assert_eq!(plan(60.0, false, 10.0), Plan { piece: 10.0, fade: 0.5, repeats: 0 });
    }

    #[test]
    fn loop_is_whole_repeats_of_its_length() {
        // короткая петля — один кусок, на хвост длиннее
        assert_eq!(plan(4.0, true, 10.0), Plan { piece: 4.5, fade: 0.5, repeats: 1 });
        // длинная — целое число повторов одного куска
        let p = plan(60.0, true, 10.0);
        assert_eq!(p.repeats, 7);
        assert!(p.piece <= 10.0 && ((p.piece - p.fade) * 7.0 - 60.0).abs() < 1e-9);
    }

    #[test]
    fn shaped_length_is_what_was_asked() {
        let sr = 1000;
        for (seconds, looped) in [(4.0, true), (60.0, true), (60.0, false)] {
            let p = plan(seconds, looped, 10.0);
            let x = vec![0.1f32; (p.piece * sr as f64).round() as usize];
            let y = shape(x, sr, seconds, &p);
            assert!((y.len() as f64 / sr as f64 - seconds).abs() < 0.01, "{seconds} {looped}: {}", y.len());
        }
    }

    #[test]
    fn russian_is_noticed() {
        assert!(cyrillic("скрип двери"));
        assert!(!cyrillic("door creak"));
    }

    #[test]
    fn output_format_in_elevenlabs_shape() {
        let req = |f: Option<&str>| SoundRequest {
            text: "x".into(), duration_seconds: None, looped: false, prompt_influence: None,
            response_format: f.map(String::from), seed: None, webhook_url: None,
        };
        let q = |o: &str| SoundQuery { output_format: Some(o.into()) };
        assert_eq!(format(&req(None), &SoundQuery::default()).ok(), Some(("mp3".into(), None)));
        assert_eq!(format(&req(None), &q("mp3_44100_128")).ok(), Some(("mp3".into(), Some(44100))));
        assert_eq!(format(&req(None), &q("pcm_16000")).ok(), Some(("pcm".into(), Some(16000))));
        assert_eq!(format(&req(Some("opus")), &q("pcm_16000")).ok(), Some(("opus".into(), None)));
        assert!(format(&req(None), &q("ulaw_8000")).is_err());
    }
}
