//! voicy — a local speech server with an OpenAI-compatible audio API.
//!
//!     voicy serve [--host 127.0.0.1] [--port 8080]
//!
//! One binary: the console, the pronunciation dictionary and the default voices
//! are inside it, and the models run in this process (native/) on prebuilt
//! llama.cpp, whisper.cpp and ONNX Runtime libraries from the cache. What has
//! no native engine yet runs as the Python engine it is (server/engines), in a
//! child process (host.rs). The HTTP API is the Python server's, checked by the
//! same suite (tests/).

mod api;
mod audio;
mod auth;
mod contexts;
mod engines;
mod errors;
mod host;
mod jobs;
mod live;
mod native;
mod speak;
mod textprep;
mod transcripts;
mod voices;
mod webhooks;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use axum::extract::DefaultBodyLimit;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, middleware};
use clap::{Parser, Subcommand};
use tower_http::catch_panic::CatchPanicLayer;

const INDEX_HTML: &str = include_str!("../../server/static/index.html");
const DEFAULT_VOICES: [(&str, &[u8]); 3] = [
    ("turgenev.wav", include_bytes!("../../server/voices/turgenev.wav")),
    ("dostoevsky.wav", include_bytes!("../../server/voices/dostoevsky.wav")),
    ("voices.json", include_bytes!("../../server/voices/voices.json")),
];

pub struct App {
    pub engines: engines::Engines,
    pub registry: jobs::Registry,
    pub queue: Arc<jobs::Queue>,
    pub voices: voices::Voices,
    pub contexts: contexts::Contexts,
    pub public_url: String,
    pub keys: Vec<String>,
    pub static_dir: Option<PathBuf>,
}

#[derive(Parser)]
#[command(name = "voicy", version, about = "Локальный речевой сервер с OpenAI-совместимым API")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Поднять сервер
    Serve {
        #[arg(long, env = "VOICY_HOST", default_value = "127.0.0.1")]
        host: String,
        #[arg(long, env = "VOICY_PORT", default_value_t = 8080)]
        port: u16,
        /// Каталог репозитория (там server/ и data/); по умолчанию ищется от текущего и от бинарника
        #[arg(long, env = "VOICY_HOME")]
        home: Option<PathBuf>,
        /// Python с зависимостями движков; по умолчанию .venv рядом с репозиторием
        #[arg(long, env = "VOICY_PYTHON")]
        python: Option<PathBuf>,
    },
    /// Вероятности детектора голоса и конца реплики для wav 16 кГц — сверка с Python
    #[command(hide = true)]
    ProbeListen { wav: PathBuf },
    /// Замер родного синтеза: фразы из JSON [{"text", "seed", "file"}], голос — wav и расшифровка
    #[command(hide = true)]
    BenchTts {
        jobs: PathBuf,
        voice: PathBuf,
        voice_text: String,
        #[arg(long, default_value = "q5_k")]
        talker: String,
        #[arg(long, default_value = "ru")]
        language: String,
    },
}

/// The repository, if the binary runs from one: then voices, contexts, the
/// console and the dictionary are its files, as for the Python server.
fn find_home(given: Option<PathBuf>) -> anyhow::Result<Option<PathBuf>> {
    let is_home = |p: &Path| p.join("server").join("engines").join("host.py").is_file();
    if let Some(p) = given {
        anyhow::ensure!(is_home(&p), "{} has no server/engines/host.py", p.display());
        return Ok(Some(p));
    }
    let starts = [std::env::current_dir().ok(), std::env::current_exe().ok()];
    for start in starts.into_iter().flatten() {
        for dir in start.ancestors() {
            if is_home(dir) {
                return Ok(Some(dir.to_path_buf()));
            }
        }
    }
    Ok(None)
}

/// Without the repository: voices live in the cache, seeded with the two that ship.
fn seed_voices(dir: &Path) -> std::io::Result<()> {
    if dir.join("voices.json").exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    for (name, data) in DEFAULT_VOICES {
        std::fs::write(dir.join(name), data)?;
    }
    Ok(())
}

fn find_python(home: &Path, given: Option<PathBuf>) -> PathBuf {
    if let Some(p) = given {
        return p;
    }
    let venv = if cfg!(windows) {
        home.join(".venv").join("Scripts").join("python.exe")
    } else {
        home.join(".venv").join("bin").join("python")
    };
    if venv.is_file() {
        return venv;
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn env_or(name: &str, default: impl FnOnce() -> PathBuf) -> PathBuf {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(default)
}

async fn index(axum::extract::State(app): axum::extract::State<Arc<App>>) -> Response {
    let page = match &app.static_dir {
        Some(d) => tokio::fs::read_to_string(d.join("index.html")).await.unwrap_or_else(|_| INDEX_HTML.into()),
        None => INDEX_HTML.into(),
    };
    ([(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")], page).into_response()
}

async fn static_file(axum::extract::State(app): axum::extract::State<Arc<App>>,
                     axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    let found = match &app.static_dir {
        Some(d) if !name.contains("..") => tokio::fs::read(d.join(&name)).await.ok(),
        _ => None,
    };
    match found {
        Some(b) => b.into_response(),
        None if name == "index.html" => INDEX_HTML.into_response(),
        None => errors::ApiError::new(404, "Not Found").into_response(),
    }
}

fn router(app: Arc<App>) -> Router {
    use api::*;
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/translations", post(translations))
        .route("/v1/models", get(models))
        .route("/v1/voices", get(get_voices).post(add_voice))
        .route("/v1/contexts", get(get_contexts))
        .route("/v1/contexts/{name}", get(get_context).put(put_context).delete(delete_context))
        .route("/v1/text/prepare", post(prepare_text))
        .route("/v1/jobs/speech", post(job_speech))
        .route("/v1/jobs/transcribe", post(job_transcribe))
        .route("/v1/jobs", get(list_jobs))
        .route("/v1/jobs/{id}", get(get_job_route).delete(cancel_job))
        .route("/v1/jobs/{id}/events", get(job_events))
        .route("/v1/jobs/{id}/audio", get(job_audio))
        .route("/v1/jobs/{id}/result", get(job_result))
        .route("/v1/audio/speech/stream", get(speech_socket))
        .route("/v1/audio/transcriptions/stream", get(live_socket))
        .route("/health", get(health))
        .route("/", get(index))
        .route("/static/{*name}", get(static_file))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn_with_state(app.clone(), auth::guard))
        .layer(CatchPanicLayer::custom(|_: Box<dyn std::any::Any + Send>| {
            errors::ApiError::internal("internal error").into_response()
        }))
        .layer(DefaultBodyLimit::disable())
        .with_state(app)
}

async fn serve(host: String, port: u16, home: Option<PathBuf>, python: Option<PathBuf>) -> anyhow::Result<()> {
    let home = find_home(home)?;
    let server_dir = home.as_ref().map(|h| h.join("server"));
    let python = find_python(home.as_deref().unwrap_or(Path::new(".")), python);
    textprep::load(home.as_ref().map(|h| h.join("data").join("pronunciation.json")).as_deref());
    let cache = native::cache_dir();
    let voices_dir = env_or("VOICY_VOICES_DIR", || match &server_dir {
        Some(s) => s.join("voices"),
        None => cache.join("voices"),
    });
    if server_dir.is_none() && std::env::var_os("VOICY_VOICES_DIR").is_none() {
        seed_voices(&voices_dir)?;
    }
    let contexts_dir = env_or("VOICY_CONTEXTS_DIR", || match &server_dir {
        Some(s) => s.join("contexts"),
        None => cache.join("contexts"),
    });

    let engines = engines::Engines::start(&python, server_dir.as_deref()).await?;
    let var = |n: &str| std::env::var(n).unwrap_or_default();
    let app = Arc::new(App {
        engines,
        registry: jobs::Registry::new(var("VOICY_JOB_TTL").parse().unwrap_or(1800.0)),
        queue: jobs::Queue::start(var("VOICY_QUEUE_MAX").parse().unwrap_or(32), var("VOICY_WEBHOOK_SECRET")),
        voices: voices::Voices { dir: voices_dir },
        contexts: contexts::Contexts { dir: contexts_dir },
        public_url: var("VOICY_PUBLIC_URL").trim_end_matches('/').to_string(),
        keys: var("VOICY_API_KEY").split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect(),
        static_dir: server_dir.as_ref().map(|s| s.join("static")),
    });
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("cannot listen on {host}:{port}"))?;
    eprintln!("voicy: http://{host}:{port}");
    axum::serve(listener, router(app)).await?;
    Ok(())
}

fn probe_listen(wav: PathBuf) -> anyhow::Result<()> {
    let lib = native::lib_dir()?;
    native::init_onnx(&lib)?;
    let m = native::cache_dir().join("models");
    let vad = native::listen::Vad::load(&m.join("vad").join("silero_vad_v6.onnx"))?;
    let turn = native::listen::Turn::load(&m.join("turn").join("smart-turn-v3.2-cpu.onnx"))?;
    let (audio, _) = audio::read_wav(&std::fs::read(wav)?).ok_or_else(|| anyhow::anyhow!("not a wav"))?;
    let mut stream = native::listen::VadStream::default();
    let mut probs = vec![];
    for chunk in audio.chunks(1600) {
        probs.extend(stream.feed(&vad, chunk)?);
    }
    let t = std::time::Instant::now();
    let p = turn.probability(&audio)?;
    println!("{}", serde_json::json!({"vad": probs, "turn": p, "turn_ms": t.elapsed().as_secs_f64() * 1000.0}));
    Ok(())
}

fn bench_tts(jobs: PathBuf, voice: PathBuf, voice_text: String, talker: String, language: String) -> anyhow::Result<()> {
    use std::time::Instant;
    let files = native::qwen::Files {
        dir: native::cache_dir().join("models").join("qwen3-tts-12hz-1.7b-base-gguf"),
        talker: format!("qwen3_tts_talker.{talker}.gguf"),
        predictor: "qwen3_tts_predictor.q8_0.gguf".into(),
    };
    let t = Instant::now();
    let q = native::qwen::Qwen::load(&files, true)?;
    eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    let lang = native::qwen::language_id(&language);
    q.speak("Прогрев.", &voice, &voice_text, lang, Some(0), |_| true)?;
    let mut rows: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&jobs)?)?;
    let (mut ta, mut tt) = (0.0, 0.0);
    for r in rows.iter_mut() {
        let text = r["text"].as_str().unwrap_or_default().to_string();
        let t = Instant::now();
        let audio = q.speak(&text, &voice, &voice_text, lang, r["seed"].as_u64().map(|s| s as u32), |_| true)?.unwrap_or_default();
        let dt = t.elapsed().as_secs_f64();
        let secs = audio.len() as f64 / 24000.0;
        std::fs::write(r["file"].as_str().unwrap_or("out.wav"), audio::to_wav(&audio, 24000))?;
        r["audio_s"] = serde_json::json!(jobs::round(secs, 2));
        r["synth_s"] = serde_json::json!(jobs::round(dt, 3));
        ta += secs;
        tt += dt;
        eprintln!("{secs:5.2} s за {dt:5.2} s  {}", text.chars().take(40).collect::<String>());
    }
    println!("{}", serde_json::json!({"rows": rows, "speed": jobs::round(ta / tt, 2)}));
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Serve { host, port, home, python } => serve(host, port, home, python).await,
        Command::ProbeListen { wav } => probe_listen(wav),
        Command::BenchTts { jobs, voice, voice_text, talker, language } => bench_tts(jobs, voice, voice_text, talker, language),
    }
}
