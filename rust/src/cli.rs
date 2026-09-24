//! The commands next to `serve`: the same ones the Python `./voicy` has, over
//! the same HTTP API — so one binary is both the server and the way to use it.
//!
//! What the Python CLI decided, this decides too: audio never goes to stdout
//! (a binary in the terminal ruins it), stdout carries the result — a path, a
//! transcript, an id — and everything else goes to stderr, so `$(voicy hear
//! …)` is exactly the text.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde_json::{Value, json};

use crate::native;

const FORMATS: [&str; 6] = ["opus", "wav", "mp3", "flac", "aac", "pcm"];

/// 127.0.0.1, а не localhost: на Windows это имя разрешается сначала в IPv6
/// ::1, где сервер не слушает, и клиент получает отказ вместо ответа.
pub fn default_url() -> String {
    std::env::var("VOICY_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".into())
}

fn key() -> Option<String> {
    // тот же ключ, что задан серверу; серверу можно несколько через запятую
    std::env::var("VOICY_API_KEY").ok().and_then(|k| k.split(',').next().map(|s| s.trim().to_string())).filter(|k| !k.is_empty())
}

/// Свой сервер — мимо прокси: с HTTP_PROXY обращение к себе же уходит в прокси
/// и возвращается 503.
fn client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(3600)).build()?)
}

fn log(msg: impl AsRef<str>) {
    eprintln!("{}", msg.as_ref());
}

struct Api {
    url: String,
    http: reqwest::Client,
}

impl Api {
    fn new(url: &str) -> anyhow::Result<Api> {
        Ok(Api { url: url.trim_end_matches('/').to_string(), http: client()? })
    }

    fn auth(&self, r: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match key() {
            Some(k) => r.bearer_auth(k),
            None => r,
        }
    }

    async fn check(r: reqwest::Response) -> anyhow::Result<reqwest::Response> {
        if r.status().is_success() {
            return Ok(r);
        }
        let status = r.status();
        let body = r.text().await.unwrap_or_default();
        let message = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(String::from))
            .unwrap_or_else(|| body.trim().to_string());
        bail!("{status}: {message}")
    }

    async fn get(&self, path: &str) -> anyhow::Result<reqwest::Response> {
        let r = self.auth(self.http.get(format!("{}{path}", self.url))).send().await
            .with_context(|| format!("сервер не отвечает на {}", self.url))?;
        Api::check(r).await
    }

    async fn get_json(&self, path: &str) -> anyhow::Result<Value> {
        Ok(self.get(path).await?.json().await?)
    }

    async fn post_json(&self, path: &str, body: Value) -> anyhow::Result<reqwest::Response> {
        let r = self.auth(self.http.post(format!("{}{path}", self.url))).json(&body).send().await
            .with_context(|| format!("сервер не отвечает на {}", self.url))?;
        Api::check(r).await
    }

    async fn post_form(&self, path: &str, form: reqwest::multipart::Form) -> anyhow::Result<reqwest::Response> {
        let r = self.auth(self.http.post(format!("{}{path}", self.url))).multipart(form).send().await
            .with_context(|| format!("сервер не отвечает на {}", self.url))?;
        Api::check(r).await
    }

    async fn delete_json(&self, path: &str) -> anyhow::Result<Value> {
        let r = self.auth(self.http.delete(format!("{}{path}", self.url))).send().await?;
        Ok(Api::check(r).await?.json().await?)
    }

    async fn health(&self) -> Option<Value> {
        self.get_json("/health").await.ok()
    }

    /// Ни одна команда не имеет смысла без сервера, и молчаливый отказ хуже
    /// подсказки, как его поднять.
    async fn require(&self) -> anyhow::Result<Value> {
        match self.health().await {
            Some(h) => Ok(h),
            None => bail!("voicy не отвечает на {}\n       поднять: voicy up", self.url),
        }
    }
}

/// `текст`, `@файл` или `-` для потока.
fn read_text(arg: &str) -> anyhow::Result<String> {
    if arg == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        return Ok(s);
    }
    if let Some(path) = arg.strip_prefix('@') {
        return std::fs::read_to_string(path).with_context(|| format!("не читается {path}"));
    }
    Ok(arg.to_string())
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().map(|e| e.to_string_lossy().to_lowercase()).as_deref() {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("opus") | Some("ogg") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("m4a") => "audio/mp4",
        Some("aac") => "audio/aac",
        Some("webm") => "audio/webm",
        _ => "application/octet-stream",
    }
}

async fn file_part(path: &Path) -> anyhow::Result<reqwest::multipart::Part> {
    let bytes = tokio::fs::read(path).await.with_context(|| format!("файл не найден: {}", path.display()))?;
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "file".into());
    Ok(reqwest::multipart::Part::bytes(bytes).file_name(name).mime_str(content_type(path))?)
}

fn write_out(out: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, bytes).with_context(|| format!("не пишется {}", out.display()))
}

/// Задание принято: в stdout — только id, чтобы его можно было подставить.
fn detached(job: &Value, rest: &str) -> anyhow::Result<()> {
    let id = job["id"].as_str().unwrap_or_default();
    let ahead = job["position"].as_u64().unwrap_or(0);
    log(format!("задание {id} принято{}\n       забрать: voicy job {id} {rest} --wait",
                if ahead > 0 { format!(", впереди {ahead}") } else { String::new() }));
    println!("{id}");
    Ok(())
}

// --------------------------------------------------------------------- say

#[allow(clippy::too_many_arguments)]
pub async fn say(url: &str, text: &str, out: Option<PathBuf>, voice: Option<String>, format: String, speed: f64,
                 language: Option<String>, seed: Option<i64>, prepare: bool, legato: bool, stress: bool, detach: bool,
                 webhook: Option<String>) -> anyhow::Result<()> {
    let text = read_text(text)?.trim().to_string();
    anyhow::ensure!(!text.is_empty(), "пустой текст");
    let out = out.unwrap_or_else(|| {
        PathBuf::from(format!("voicy-{}.{format}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0)))
    });
    if out.as_os_str() == "-" {
        bail!("аудио в stdout не пишется — бинарь засорит вывод. Укажите файл: voicy say \"...\" out.{format}");
    }
    // расширение важнее флага
    let fmt = out.extension().map(|e| e.to_string_lossy().to_lowercase()).filter(|e| FORMATS.contains(&e.as_str()))
        .unwrap_or(format);

    let api = Api::new(url)?;
    let health = api.require().await?;
    if health["tts"]["loaded"] == json!(false) {
        log("модель синтеза ещё не в памяти — первый вызов дольше обычного");
    }
    let mut payload = json!({"input": text, "response_format": fmt, "speed": speed, "prepare": prepare,
                             "legato": legato, "stress": stress});
    if let Some(v) = voice {
        payload["voice"] = json!(v);
    }
    if let Some(l) = language {
        payload["language"] = json!(l);
    }
    if let Some(s) = seed {
        payload["seed"] = json!(s);
    }
    if detach || webhook.is_some() {
        if let Some(w) = webhook {
            payload["webhook_url"] = json!(w);
        }
        let job: Value = api.post_json("/v1/jobs/speech", payload).await?.json().await?;
        return detached(&job, &out.display().to_string());
    }
    let started = Instant::now();
    let r = api.post_json("/v1/audio/speech", payload).await?;
    let head = |k: &str| r.headers().get(k).and_then(|v| v.to_str().ok()).unwrap_or("?").to_string();
    let (secs, spent, voice) = (head("x-audio-seconds"), head("x-generation-seconds"), head("x-voice"));
    let audio = r.bytes().await?;
    write_out(&out, &audio)?;
    let spent = if spent == "?" { format!("{:.1}", started.elapsed().as_secs_f64()) } else { spent };
    log(format!("{voice}: {secs} с звука за {spent} с, {:.0} КиБ", audio.len() as f64 / 1024.0));
    println!("{}", out.display());
    Ok(())
}

// -------------------------------------------------------------------- hear

#[allow(clippy::too_many_arguments)]
pub async fn hear(url: &str, file: PathBuf, language: Option<String>, format: String, prompt: Option<String>,
                  temperature: f64, words: bool, context: Option<String>, hotwords: Option<String>, detach: bool,
                  webhook: Option<String>) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let mut form = reqwest::multipart::Form::new().part("file", file_part(&file).await?);
    let mut add = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            form = std::mem::take(&mut form).text(k.to_string(), v);
        }
    };
    add("language", language);
    add("prompt", prompt);
    add("context", context);
    add("hotwords", hotwords);
    add("temperature", Some(temperature.to_string()));

    if detach || webhook.is_some() {
        add("word_timestamps", Some(words.to_string()));
        add("webhook_url", webhook);
        let job: Value = api.post_form("/v1/jobs/transcribe", form).await?.json().await?;
        return detached(&job, &format!("--format {format}"));
    }
    add("model", Some("whisper-1".into()));
    add("response_format", Some(format.clone()));
    if words {
        add("timestamp_granularities", Some("word".into()));
    }
    let body = api.post_form("/v1/audio/transcriptions", form).await?.text().await?;
    let text = if format == "json" {
        serde_json::from_str::<Value>(&body).ok().and_then(|v| v["text"].as_str().map(String::from)).unwrap_or(body)
    } else {
        body
    };
    println!("{}", text.trim());
    Ok(())
}

// -------------------------------------------------------------------- jobs

pub async fn job(url: &str, id: String, out: Option<PathBuf>, wait: bool, format: String) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let mut last = String::new();
    let job = loop {
        let job = api.get_json(&format!("/v1/jobs/{id}")).await?;
        let state = job["state"].as_str().unwrap_or_default().to_string();
        if !wait || ["done", "error", "cancelled"].contains(&state.as_str()) {
            break job;
        }
        let note = if state == "queued" {
            format!("в очереди, впереди {}", job["position"].as_u64().unwrap_or(0))
        } else {
            job["progress"]["stage"].as_str().unwrap_or("").to_string()
        };
        if note != last {
            log(format!("  {note}"));
            last = note;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    };

    let state = job["state"].as_str().unwrap_or_default();
    if state != "done" {
        if state == "error" {
            bail!("задание {id} упало: {}", job["error"]);
        }
        log(format!("задание {id}: {state}"));
        if out.is_none() {
            println!("{}", serde_json::to_string_pretty(&job)?);
        }
        std::process::exit(if ["queued", "running"].contains(&state) { 3 } else { 1 });
    }

    if job["kind"] == json!("speech") {
        let Some(out) = out else {
            println!("{}", serde_json::to_string_pretty(&job)?);
            log(format!("звук готов — укажите файл: voicy job {id} out.{}",
                        job["result"]["format"].as_str().unwrap_or("opus")));
            return Ok(());
        };
        let audio = api.get(&format!("/v1/jobs/{id}/audio")).await?.bytes().await?;
        write_out(&out, &audio)?;
        log(format!("{}: {} с звука", job["result"]["voice"].as_str().unwrap_or("?"), job["result"]["seconds"]));
        println!("{}", out.display());
        return Ok(());
    }
    let body = api.get(&format!("/v1/jobs/{id}/result?format={format}")).await?.text().await?;
    let text = if format == "json" {
        serde_json::from_str::<Value>(&body).ok().and_then(|v| v["text"].as_str().map(String::from)).unwrap_or(body)
    } else {
        body
    };
    println!("{}", text.trim());
    Ok(())
}

pub async fn jobs(url: &str) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let data = api.get_json("/v1/jobs").await?;
    for j in data["data"].as_array().unwrap_or(&vec![]) {
        let state = j["state"].as_str().unwrap_or_default();
        let extra = if state == "queued" {
            format!("впереди {}", j["position"].as_u64().unwrap_or(0))
        } else {
            j["error"].as_str().unwrap_or_default().to_string()
        };
        println!("{}  {:<10} {:<9} {extra}", j["id"].as_str().unwrap_or("?"), j["kind"].as_str().unwrap_or("?"), state);
    }
    let q = &data["queue"];
    log(format!("в очереди {} из {}{}", q["pending"], q["max"],
                q["running"].as_str().map(|r| format!(", работает {r}")).unwrap_or_default()));
    Ok(())
}

pub async fn cancel(url: &str, id: String) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let job = api.delete_json(&format!("/v1/jobs/{id}")).await?;
    log(if job["cancel_requested"] == json!(true) { "остановится на ближайшем шаге" } else { "отменено" });
    println!("{}", job["id"].as_str().unwrap_or(&id));
    Ok(())
}

// ------------------------------------------------------- контексты и голоса

pub async fn contexts(url: &str) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let data = api.get_json("/v1/contexts").await?;
    for c in data["data"].as_array().unwrap_or(&vec![]) {
        let mark = if c["builtin"] == json!(true) { "*" } else { " " };
        println!("{mark} {:<14} {:>3} терм.  {}", c["name"].as_str().unwrap_or("?"),
                 c["hotwords"].as_array().map(|a| a.len()).unwrap_or(0), c["note"].as_str().unwrap_or(""));
    }
    log(format!("свой: curl -X PUT {url}/v1/contexts/имя -H 'Content-Type: application/json' \
                 -d '{{\"hotwords\": [...], \"replacements\": {{...}}}}'"));
    Ok(())
}

pub async fn voices(url: &str) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let data = api.get_json("/v1/voices").await?;
    let list = data["data"].as_array().cloned().unwrap_or_default();
    for v in &list {
        let mark = if v["name"] == data["default"] { "*" } else { " " };
        println!("{mark} {:<14} {}", v["name"].as_str().unwrap_or("?"), v["note"].as_str().unwrap_or(""));
    }
    if list.is_empty() {
        log("голосов нет — добавьте: voicy add-voice образец.wav имя");
    }
    Ok(())
}

pub async fn add_voice(url: &str, file: PathBuf, name: String, text: Option<String>, note: Option<String>,
                       replace: bool) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    // Расшифровку не требуем: сервер распознает образец сам, и это точнее ручного ввода
    let form = reqwest::multipart::Form::new()
        .part("file", file_part(&file).await?)
        .text("name", name)
        .text("text", text.unwrap_or_default())
        .text("note", note.unwrap_or_default())
        .text("replace", replace.to_string());
    let v: Value = api.post_form("/v1/voices", form).await?.json().await?;
    log(format!("расшифровка образца: {:?}", v["reference_text"].as_str().unwrap_or("")));
    for w in v["warnings"].as_array().unwrap_or(&vec![]) {
        log(format!("внимание: {}", w.as_str().unwrap_or_default()));
    }
    println!("{}", v["name"].as_str().unwrap_or_default());
    Ok(())
}

pub async fn prepare(url: &str, text: &str, no_dictionary: bool, legato: bool, stress: bool) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    let body = json!({"text": read_text(text)?, "dictionary": !no_dictionary, "legato": legato, "stress": stress});
    let v: Value = api.post_json("/v1/text/prepare", body).await?.json().await?;
    println!("{}", v["text"].as_str().unwrap_or_default());
    Ok(())
}

// ------------------------------------------------------------ up/down/status

fn run_dir() -> PathBuf {
    native::cache_dir().join("run")
}

fn read_pid() -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(run_dir().join("server.pid")).ok()?.trim().parse().ok()?;
    alive(pid).then_some(pid)
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    unsafe { libc_kill(pid as i32, 0) == 0 }
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

#[cfg(windows)]
fn alive(pid: u32) -> bool {
    // tasklist врать не станет, а тащить winapi ради одной проверки незачем
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

pub async fn status(url: &str) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    let Some(h) = api.health().await else {
        log(format!("voicy не отвечает на {url}"));
        match read_pid() {
            Some(pid) => log(format!("свой процесс {pid} жив, но ещё не отвечает — журнал: {}",
                                     run_dir().join("server.log").display())),
            None => log("поднять: voicy up"),
        }
        std::process::exit(1);
    };
    match read_pid() {
        Some(pid) => log(format!("запущен процессом {pid}")),
        None => log("запущен не этим CLI"),
    }
    println!("{}", serde_json::to_string_pretty(&h)?);
    Ok(())
}

pub async fn warm(url: &str) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    api.require().await?;
    log("прогрев моделей …");
    let started = Instant::now();
    api.post_json("/v1/audio/speech", json!({"input": "Проверка связи.", "response_format": "wav"})).await?;
    log(format!("синтез готов ({:.0} с)", started.elapsed().as_secs_f64()));
    Ok(())
}

/// Поднять сервер тем же бинарником: отдельным процессом, чтобы он пережил
/// эту команду, с pid и журналом в кэше.
pub async fn up(url: &str, wait: f64, do_warm: bool, cpu: bool) -> anyhow::Result<()> {
    let api = Api::new(url)?;
    if api.health().await.is_some() {
        log(format!("voicy уже работает на {url}"));
        return if do_warm { warm(url).await } else { Ok(()) };
    }
    let port = url.rsplit(':').next().and_then(|p| p.trim_end_matches('/').parse::<u16>().ok()).unwrap_or(8080);
    let dir = run_dir();
    std::fs::create_dir_all(&dir)?;
    let log_file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("server.log"))?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.arg("serve").arg("--port").arg(port.to_string())
        .stdin(std::process::Stdio::null())
        .stderr(log_file.try_clone()?)
        .stdout(log_file);
    if cpu {
        cmd.env("FORCE_CPU", "1");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // переживёт эту команду
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000008 | 0x00000200); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    }
    let child = cmd.spawn().context("не запускается сервер")?;
    std::fs::write(dir.join("server.pid"), child.id().to_string())?;
    log(format!("процесс {}, журнал — {}", child.id(), dir.join("server.log").display()));
    log("ждём готовности (первый запуск грузит модели, это долго) …");
    let deadline = Instant::now() + Duration::from_secs_f64(wait);
    while Instant::now() < deadline {
        if api.health().await.is_some() {
            log(format!("voicy на {url}, консоль там же"));
            return if do_warm { warm(url).await } else { Ok(()) };
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    bail!("сервер не ответил за {wait:.0} с — журнал: {}", dir.join("server.log").display())
}

pub async fn down(url: &str) -> anyhow::Result<()> {
    let Some(pid) = read_pid() else {
        let api = Api::new(url)?;
        log(if api.health().await.is_some() {
            "на этом адресе отвечает не наш процесс — остановите его сам"
        } else {
            "нечего останавливать"
        });
        return Ok(());
    };
    log(format!("останавливаю процесс {pid}"));
    #[cfg(unix)]
    unsafe {
        libc_kill(pid as i32, 15);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).output();
    }
    for _ in 0..50 {
        if read_pid().is_none() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let _ = std::fs::remove_file(run_dir().join("server.pid"));
    Ok(())
}
