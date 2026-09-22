//! voicy — a local speech server with an OpenAI-compatible audio API.
//!
//!     voicy serve [--host 127.0.0.1] [--port 8080]
//!
//! The server is Rust; the models still run as the Python engines they are
//! (server/engines), in a child process it starts and talks to over pipes
//! (host.rs). The HTTP API is the same as the Python server's, checked by the
//! same suite (tests/). One binary is meant to take the CLI's commands too.

mod api;
mod audio;
mod auth;
mod contexts;
mod engines;
mod errors;
mod host;
mod jobs;
mod live;
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
use tower_http::services::ServeDir;

pub struct App {
    pub engines: engines::Engines,
    pub registry: jobs::Registry,
    pub queue: Arc<jobs::Queue>,
    pub voices: voices::Voices,
    pub contexts: contexts::Contexts,
    pub public_url: String,
    pub keys: Vec<String>,
    pub static_dir: PathBuf,
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
}

fn find_home(given: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    let is_home = |p: &Path| p.join("server").join("engines").join("host.py").is_file();
    if let Some(p) = given {
        anyhow::ensure!(is_home(&p), "{} has no server/engines/host.py", p.display());
        return Ok(p);
    }
    let starts = [std::env::current_dir().ok(), std::env::current_exe().ok()];
    for start in starts.into_iter().flatten() {
        for dir in start.ancestors() {
            if is_home(dir) {
                return Ok(dir.to_path_buf());
            }
        }
    }
    anyhow::bail!("cannot find the voicy repository (server/engines/host.py); pass --home or VOICY_HOME")
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
    match tokio::fs::read(app.static_dir.join("index.html")).await {
        Ok(b) => ([(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")], b).into_response(),
        Err(e) => errors::ApiError::internal(e.to_string()).into_response(),
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
        .nest_service("/static", ServeDir::new(app.static_dir.clone()))
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
    let python = find_python(&home, python);
    let server_dir = home.join("server");
    textprep::load(&home.join("data").join("pronunciation.json"));

    eprintln!("voicy: engines via {}", python.display());
    let host_proc = host::Host::spawn(&python, &server_dir).await?;
    let engines = engines::Engines::start(host_proc).await?;
    let var = |n: &str| std::env::var(n).unwrap_or_default();
    let app = Arc::new(App {
        engines,
        registry: jobs::Registry::new(var("VOICY_JOB_TTL").parse().unwrap_or(1800.0)),
        queue: jobs::Queue::start(var("VOICY_QUEUE_MAX").parse().unwrap_or(32), var("VOICY_WEBHOOK_SECRET")),
        voices: voices::Voices { dir: env_or("VOICY_VOICES_DIR", || server_dir.join("voices")) },
        contexts: contexts::Contexts { dir: env_or("VOICY_CONTEXTS_DIR", || server_dir.join("contexts")) },
        public_url: var("VOICY_PUBLIC_URL").trim_end_matches('/').to_string(),
        keys: var("VOICY_API_KEY").split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect(),
        static_dir: server_dir.join("static"),
    });
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("cannot listen on {host}:{port}"))?;
    eprintln!("voicy: http://{host}:{port}");
    axum::serve(listener, router(app)).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Serve { host, port, home, python } => serve(host, port, home, python).await,
    }
}
