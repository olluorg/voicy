//! `voicy setup` — everything the in-process engines need, fetched by the binary
//! itself.
//!
//! Prebuilt llama.cpp and whisper.cpp from their GitHub releases, CTranslate2
//! and ONNX Runtime out of the wheels their authors publish on PyPI, the models
//! from Hugging Face — into `~/.cache/voicy` (`VOICY_CACHE`). Versions are
//! pinned: the bindings in native/ are written against these and no others.
//!
//! Nothing is built here except `ct2shim`, a page of C++ over CTranslate2's C++
//! API, and only when a compiler is at hand. The one thing this cannot do is
//! Qwen3-TTS comes already converted; making those files from the official
//! weights is the one step that needs PyTorch (`scripts/convert_qwen.py`).

use std::fs;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use futures::StreamExt;

use crate::native;

const LLAMA: &str = "b11090";
const WHISPER_CPP: &str = "b5130";
const CT2: &str = "4.8.2";
const ORT: &str = "1.30.0";
const CUBLAS12: &str = "12.8.4.1";
const WHISPER_MODEL: &str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo";
/// Qwen3-TTS в GGUF и ONNX: те же официальные веса, переведённые
/// scripts/convert_qwen.py. Свой репозиторий — VOICY_TTS_GGUF_REPO.
const QWEN_MODEL: &str = "sknyazev/qwen3-tts-12hz-1.7b-base-gguf";
/// По умолчанию — та же модель, дообученная слушаться знака ударения (docs/adr/0023).
const QWEN_STRESS_MODEL: &str = "sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf";
const TURN_MODEL: &str = "pipecat-ai/smart-turn-v3";
/// Silero — из колеса faster-whisper: тот же файл, что слышит движок Python.
const FASTER_WHISPER: &str = "1.2.1";

/// What a platform needs, and where it comes from. A platform is here once its
/// archives are known; being here is not the same as being measured — that is
/// what rust/README.md says of each.
struct Platform {
    /// Сборки llama.cpp и whisper.cpp из их релизов: ggml с бэкендом CUDA — оттуда же.
    llama_assets: &'static [&'static str],
    whisper_asset: &'static str,
    /// Колёса PyPI: чем помечены под эту платформу и что из них берётся.
    wheels: &'static [&'static str],
    /// Все эти куски должны быть в имени колеса: одного «manylinux» мало —
    /// под ним лежат и x86-64, и aarch64, и первым в списке PyPI бывает не тот.
    wheel_tag: &'static [&'static str],
    wanted: &'static [&'static str],
}

const WHEELS: &[&str] = &["onnxruntime-gpu=={ort}", "nvidia-cudnn-cu13", "nvidia-curand", "nvidia-cufft",
                          "nvidia-nvjitlink", "nvidia-cuda-nvrtc", "ctranslate2=={ct2}",
                          "nvidia-cublas-cu12=={cublas12}"];

const LINUX_X64_CUDA13: Platform = Platform {
    llama_assets: &["llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz"],
    whisper_asset: "whisper-bin-ubuntu-x64.tar.gz",
    wheels: WHEELS,
    wheel_tag: &["manylinux", "x86_64"],
    wanted: &["libonnxruntime.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so", "libcudnn",
              "libcurand.so", "libcufft.so", "libnvJitLink.so", "libnvrtc", "libctranslate2", "libgomp",
              "libcublas.so.12", "libcublasLt.so.12"],
};

const WINDOWS_X64_CUDA13: Platform = Platform {
    llama_assets: &["llama-{v}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"],
    whisper_asset: "whisper-bin-x64.zip",
    wheels: WHEELS,
    wheel_tag: &["win_amd64"],
    // cudnn64_9.dll есть и у CTranslate2 (под CUDA 12), и в колесе cuDNN под CUDA 13;
    // имя одно, и в процессе останется тот, кто загрузился первым — cuDNN берём у CUDA 13.
    wanted: &["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll", "cudnn",
              "curand64_", "cufft64_", "nvJitLink_", "nvrtc", "ctranslate2.dll", "libiomp5md.dll", "cublas64_12.dll",
              "cublasLt64_12.dll"],
};

fn platform() -> anyhow::Result<&'static Platform> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok(&LINUX_X64_CUDA13),
        ("windows", "x86_64") => Ok(&WINDOWS_X64_CUDA13),
        (os, arch) => bail!(
            "no engine libraries for {os}-{arch} yet: the archives for this platform are the next step\n\
             (docs/adr/0022). Where there is a repository with server/, the Python engines still run: voicy serve."
        ),
    }
}

fn say(msg: impl AsRef<str>) {
    eprintln!("setup: {}", msg.as_ref());
}

fn expand(t: &str) -> String {
    t.replace("{v}", LLAMA).replace("{ort}", ORT).replace("{ct2}", CT2).replace("{cublas12}", CUBLAS12)
}

// ------------------------------------------------------------------ скачивание

/// One file of the plan: where from, where to, and how big, when the server says.
struct Download {
    url: String,
    to: PathBuf,
    /// Каталог, под которым файл показывается в плане: модель целиком, а не её подкаталоги.
    dir: PathBuf,
    size: Option<u64>,
}

/// Everything setup is about to fetch. It is shown whole, sizes and directories,
/// and fetched only once agreed to: gigabytes are not something to download quietly.
#[derive(Default)]
struct Plan {
    items: Vec<Download>,
}

impl Plan {
    /// `to` once it is not there yet; the path either way.
    fn add(&mut self, url: String, to: PathBuf, dir: &Path) -> PathBuf {
        if !to.exists() && !self.items.iter().any(|d| d.to == to) {
            self.items.push(Download { url, to: to.clone(), dir: dir.to_path_buf(), size: None });
        }
        to
    }

    /// An archive in the downloads directory, under the name its URL gives.
    fn archive(&mut self, url: String, dir: &Path) -> PathBuf {
        let name = url.split('?').next().unwrap_or(&url).rsplit('/').next().unwrap_or("file").to_string();
        self.add(url, dir.join(name), dir)
    }

    /// A file of a Hugging Face repository at a revision.
    fn hf(&mut self, repo: &str, revision: &str, file: &str, dir: &Path) -> PathBuf {
        self.add(format!("{}/{repo}/resolve/{revision}/{file}?download=true", hf_endpoint()), dir.join(file), dir)
    }

    /// Sizes, asked for with a one-byte range: HEAD is not answered by the
    /// signed links that GitHub and Hugging Face redirect to.
    async fn measure(&mut self, http: &reqwest::Client) {
        let sizes: Vec<Option<u64>> = futures::stream::iter(self.items.iter().map(|d| size_of(http, &d.url)))
            .buffered(8)
            .collect()
            .await;
        for (d, s) in self.items.iter_mut().zip(sizes) {
            d.size = s;
        }
    }

    fn total(&self) -> u64 {
        self.items.iter().filter_map(|d| d.size).sum()
    }

    /// `lib` — куда распакуются библиотеки, если они в плане.
    fn show(&self, lib: Option<&Path>) {
        let unknown = self.items.iter().filter(|d| d.size.is_none()).count();
        let total = match (unknown, self.total()) {
            (0, t) => human(t),
            (_, 0) => "размер узнать не удалось".into(),
            (_, t) => format!("не меньше {}", human(t)),
        };
        say(format!("будет скачано {} {}, {total}:", self.items.len(), files(self.items.len())));
        let resumed: u64 = self.items.iter().filter_map(|d| fs::metadata(part_of(d)).ok()).map(|m| m.len()).sum();
        let mut dirs: Vec<(&Path, u64, usize)> = vec![];
        for d in &self.items {
            match dirs.iter_mut().find(|(p, ..)| *p == d.dir) {
                Some(g) => (g.1, g.2) = (g.1 + d.size.unwrap_or(0), g.2 + 1),
                None => dirs.push((&d.dir, d.size.unwrap_or(0), 1)),
            }
        }
        for (dir, size, n) in dirs {
            let size = if size == 0 { "?".into() } else { human(size) };
            say(format!("  {size:>8}  {n:>2} {:<6}  {}", files(n), dir.display()));
        }
        if resumed > 0 {
            say(format!("{} из них уже лежит с прошлого раза — докачаю с того же места", human(resumed)));
        }
        if let Some(lib) = lib {
            say(format!("библиотеки из архивов лягут в {}; сами архивы после этого можно удалить", lib.display()));
        }
        say(format!("корневой каталог — {} (меняется переменной VOICY_CACHE)", native::cache_dir().display()));
    }

    async fn fetch_all(&self, http: &reqwest::Client) -> anyhow::Result<()> {
        let mut bar = Bar::new(self.items.len(), self.total());
        for (i, d) in self.items.iter().enumerate() {
            fetch(http, d, i + 1, &mut bar).await?;
        }
        say(format!("скачано {} за {}", human(bar.received), duration(bar.started.elapsed().as_secs())));
        Ok(())
    }
}

/// Hugging Face or a mirror of it: HF_ENDPOINT, as huggingface_hub reads it.
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_string())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into())
}

async fn size_of(http: &reqwest::Client, url: &str) -> Option<u64> {
    let r = http.get(url).header(reqwest::header::RANGE, "bytes=0-0").send().await.ok()?.error_for_status().ok()?;
    let range = r.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok());
    match range.and_then(|v| v.rsplit('/').next()?.parse().ok()) {
        Some(total) => Some(total),
        // диапазон проигнорирован — тогда длина и есть размер; тело не читаем
        None if r.status() == reqwest::StatusCode::OK => r.content_length(),
        None => None,
    }
}

/// Asks before gigabytes go over the wire. Without a terminal there is no one
/// to ask, and agreement has to come with the command itself.
fn confirm(yes: bool) -> anyhow::Result<()> {
    if yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        bail!("скачивание не подтверждено: здесь нет терминала, чтобы спросить; voicy setup --yes");
    }
    eprint!("setup: скачать? [Y/n] ");
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    match answer.trim().to_lowercase().as_str() {
        "" | "y" | "yes" | "д" | "да" => Ok(()),
        _ => bail!("скачивание отменено, ничего не скачано"),
    }
}

/// Progress of the whole download: in a terminal one line redrawn in place,
/// under the name of the file; elsewhere — a line per 50 MB, as a log wants it.
struct Bar {
    tty: bool,
    count: usize,
    total: u64,
    /// Сколько весят уже скачанные файлы — для общего процента.
    finished: u64,
    /// Сколько байт пришло по сети в этом запуске — для скорости: докачанное
    /// из прошлого раза её не завышает.
    received: u64,
    started: Instant,
    drawn: Instant,
    width: usize,
}

impl Bar {
    fn new(count: usize, total: u64) -> Self {
        let now = Instant::now();
        Bar { tty: std::io::stderr().is_terminal(), count, total, finished: 0, received: 0, started: now, drawn: now, width: 0 }
    }

    fn start(&mut self, i: usize, d: &Download) {
        let root = native::cache_dir();
        let path = d.to.strip_prefix(&root).unwrap_or(&d.to);
        let size = d.size.map(|s| format!(" ({})", human(s))).unwrap_or_default();
        say(format!("[{i}/{}] {}{size}", self.count, path.display()));
    }

    fn update(&mut self, done: u64, size: u64, shown: &mut u64) {
        if !self.tty {
            // раз в 50 МБ: качаются гигабайты, и молчание выглядит как зависание
            if done - *shown > 50 << 20 {
                *shown = done;
                match size {
                    0 => say(format!("  {}", human(done))),
                    s => say(format!("  {} из {}", human(done), human(s))),
                }
            }
            return;
        }
        if self.drawn.elapsed() < Duration::from_millis(150) {
            return;
        }
        self.drawn = Instant::now();
        const CELLS: usize = 20;
        let filled = (done.min(size) * CELLS as u64).checked_div(size).unwrap_or(0) as usize;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(CELLS - filled));
        let of = if size == 0 { String::new() } else { format!(" / {}", human(size)) };
        let secs = self.started.elapsed().as_secs_f64();
        let got = self.finished + done;
        let speed = self.received as f64 / secs.max(0.001);
        let mut line = format!("  {bar}  {}{of} · {}/с", human(done), human(speed as u64));
        if self.total > 0 && speed > 0.0 {
            let left = self.total.saturating_sub(got) as f64 / speed;
            line += &format!(" · всего {}%, ещё {}", (got * 100 / self.total).min(100), duration(left as u64));
        }
        self.draw(&line);
    }

    fn finish(&mut self, done: u64) {
        self.finished += done;
        self.clear();
    }

    fn clear(&mut self) {
        if self.tty && self.width > 0 {
            self.draw("");
        }
    }

    /// `\r` и пробелы поверх прежнего: так стирают строку и там, где консоль
    /// не понимает управляющих последовательностей.
    fn draw(&mut self, line: &str) {
        let n = line.chars().count();
        eprint!("\r{line}{}\r", " ".repeat(self.width.saturating_sub(n)));
        if !line.is_empty() {
            eprint!("{line}");
        }
        self.width = n;
    }
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} ГБ", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{} МБ", b >> 20),
        b => format!("{} КБ", b.div_ceil(1 << 10)),
    }
}

fn duration(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s} с"),
        s if s < 3600 => format!("{} мин {} с", s / 60, s % 60),
        s => format!("{} ч {} мин", s / 3600, s % 3600 / 60),
    }
}

fn files(n: usize) -> &'static str {
    match (n % 10, n % 100) {
        (1, r) if r != 11 => "файл",
        (2..=4, r) if !(12..=14).contains(&r) => "файла",
        _ => "файлов",
    }
}

/// Attempts per file. Hugging Face drops long connections now and then, and
/// each retry picks up where the last one stopped rather than starting over.
const ATTEMPTS: u32 = 6;

fn part_of(d: &Download) -> PathBuf {
    let name = d.to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    d.to.with_file_name(format!("{name}.part"))
}

async fn fetch(http: &reqwest::Client, d: &Download, i: usize, bar: &mut Bar) -> anyhow::Result<()> {
    bar.start(i, d);
    if let Some(parent) = d.to.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = part_of(d);
    let mut attempt = 1;
    loop {
        match fetch_once(http, d, &part, bar).await {
            Ok(size) => {
                bar.finish(size);
                fs::rename(&part, &d.to)?;
                return Ok(());
            }
            Err(e) if attempt < ATTEMPTS => {
                bar.clear();
                let have = fs::metadata(&part).map_or(0, |m| m.len());
                let wait = 2u64 << attempt; // 4, 8 … 64 с
                say(format!("  оборвалось на {}: {}; попытка {} из {ATTEMPTS} через {wait} с",
                            human(have), e.root_cause(), attempt + 1));
                tokio::time::sleep(Duration::from_secs(wait)).await;
                attempt += 1;
            }
            Err(e) => {
                bar.clear();
                let kept = match fs::metadata(&part).map_or(0, |m| m.len()) {
                    0 => String::new(),
                    n => format!("\nСкачанные {} лежат в {}; повторный voicy setup продолжит с этого места.",
                                 human(n), part.display()),
                };
                if unreachable(&e) {
                    bail!("{}{kept}\nПричина: {}", no_route(&d.url), e.root_cause());
                }
                return Err(e).with_context(|| format!("{} не скачан за {ATTEMPTS} попыток{kept}", d.url));
            }
        }
    }
}

/// Whether the failure is the connection itself: refused, reset during the
/// handshake, timed out. Retrying that does not help; the network has to change.
fn unreachable(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<reqwest::Error>().is_some_and(|r| r.is_connect() || r.is_timeout()))
}

/// What to say when a host cannot be reached at all: it is the network, and
/// here is what can be changed about it.
fn no_route(url: &str) -> String {
    let host = url.split("://").nth(1).and_then(|r| r.split('/').next()).unwrap_or(url);
    let set = if cfg!(windows) { "setx" } else { "export" };
    let eq = if cfg!(windows) { " " } else { "=" };
    let system = match std::env::consts::OS {
        "windows" => "системный прокси Windows voicy берёт сам, иначе — ",
        "macos" => "системный прокси macOS voicy берёт сам, иначе — ",
        _ => "",
    };
    let mut s = format!(
        "{host} недоступен с этой машины: соединение обрывается ещё при подключении, {ATTEMPTS} попыток подряд.\n\
         Это сеть, а не voicy — откройте https://{host} в браузере.\n\
         Если браузер ходит через прокси или VPN, voicy нужно пустить тем же путём:\n  \
         {system}{set} HTTPS_PROXY{eq}http://адрес:порт");
    if host.contains("huggingface.co") {
        s += &format!("\nЗеркало Hugging Face вместо него: {set} HF_ENDPOINT{eq}https://адрес-зеркала");
    }
    if cfg!(windows) {
        s += "\n(после setx — новое окно: переменная видна только новым процессам)";
    }
    s
}

/// One attempt: from the start, or from the end of what `.part` already holds.
/// Returns the size of the whole file once it is all there.
async fn fetch_once(http: &reqwest::Client, d: &Download, part: &Path, bar: &mut Bar) -> anyhow::Result<u64> {
    use reqwest::StatusCode;
    let have = fs::metadata(part).map_or(0, |m| m.len());
    let mut req = http.get(&d.url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let r = req.send().await.with_context(|| format!("cannot reach {}", d.url))?;
    // в .part уже весь файл: процесс оборвался между последним байтом и переименованием
    if r.status() == StatusCode::RANGE_NOT_SATISFIABLE && d.size == Some(have) {
        return Ok(have);
    }
    let r = r.error_for_status().with_context(|| format!("cannot download {}", d.url))?;
    let resumed = r.status() == StatusCode::PARTIAL_CONTENT;
    let total = if resumed {
        let range = r.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok());
        range.and_then(|v| v.rsplit('/').next()?.parse().ok()).or(d.size)
    } else {
        r.content_length().or(d.size)
    }
    .unwrap_or(0);
    let mut file = if resumed {
        tokio::fs::OpenOptions::new().append(true).open(part).await?
    } else {
        tokio::fs::File::create(part).await? // сервер докачку не умеет — с начала
    };
    let mut done = if resumed { have } else { 0 };
    let mut shown = done;
    let mut stream = r.bytes_stream();
    let mut failed = None;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                failed = Some(e);
                break;
            }
        };
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        done += chunk.len() as u64;
        bar.received += chunk.len() as u64;
        bar.update(done, total, &mut shown);
    }
    // что пришло — на диск в любом случае: с этого места пойдёт следующая попытка
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    drop(file);
    if let Some(e) = failed {
        return Err(e.into());
    }
    if total != 0 && done < total {
        bail!("соединение закрыто на {} из {}", human(done), human(total));
    }
    if total != 0 && done > total {
        fs::remove_file(part)?; // не тот файл — начать заново
        bail!("пришло {done} байт вместо {total}");
    }
    Ok(done)
}

/// The manylinux x86-64 wheel of `name[==version]`, as PyPI's JSON index gives it.
async fn wheel_url(http: &reqwest::Client, spec: &str, tag: &[&str]) -> anyhow::Result<String> {
    let (name, version) = spec.split_once("==").map_or((spec, ""), |(n, v)| (n, v));
    let url = if version.is_empty() {
        format!("https://pypi.org/pypi/{name}/json")
    } else {
        format!("https://pypi.org/pypi/{name}/{version}/json")
    };
    let v: serde_json::Value = http.get(&url).send().await?.error_for_status()?.json().await?;
    let files = v["urls"].as_array().context("pypi: no files")?;
    files
        .iter()
        .filter_map(|f| {
            let n = f["filename"].as_str()?;
            let ok = n.ends_with(".whl") && tag.iter().all(|t| n.contains(t)) && (n.contains("cp312") || n.contains("py3-none"));
            ok.then(|| f["url"].as_str().unwrap_or_default().to_string())
        })
        .next()
        .with_context(|| format!("no {} wheel for {spec}", tag.join("+")))
}

// ------------------------------------------------------------------ распаковка

/// Flattens out of a .tar.gz the files whose name `keep` accepts.
fn untar(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    let file = fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let mut n = 0;
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let Some(name) = path.file_name().and_then(|s| s.to_str()).map(String::from) else { continue };
        if !keep(&name) {
            continue;
        }
        let out = dest.join(&name);
        let _ = fs::remove_file(&out);
        // libggml.so и прочие — ссылки на версию рядом; загрузчик ищет их по короткому имени
        if entry.header().entry_type().is_symlink() {
            let link = entry.link_name()?.context("symlink without target")?;
            let target = link.file_name().context("odd symlink")?.to_owned();
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &out)?;
            #[cfg(not(unix))]
            fs::copy(dest.join(&target), &out)?;
            n += 1;
            continue;
        }
        entry.unpack(&out)?;
        n += 1;
    }
    Ok(n)
}

/// The same for a wheel, which is a zip.
fn unzip(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    let mut zip = zip::ZipArchive::new(fs::File::open(archive)?)?;
    let mut n = 0;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        let Some(name) = f.enclosed_name().and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        else {
            continue;
        };
        if !f.is_file() || !keep(&name) {
            continue;
        }
        let mut bytes = vec![];
        f.read_to_end(&mut bytes)?;
        fs::write(dest.join(&name), bytes)?;
        n += 1;
    }
    Ok(n)
}

/// Unpacks the whole tree of a source tarball; only the headers are used.
fn untar_all(archive: &Path, dest: &Path) -> anyhow::Result<PathBuf> {
    let file = fs::File::open(archive)?;
    tar::Archive::new(flate2::read::GzDecoder::new(file)).unpack(dest)?;
    let root = fs::read_dir(dest)?.next().context("empty archive")??.path();
    Ok(root)
}

// ------------------------------------------------------------------- обёртка

/// ct2shim: a C face for CTranslate2's C++ API, and the one thing built here —
/// on Unix. On Windows it is compiled into the binary instead (build.rs), since
/// a compiler is not something to ask of whoever just runs the server.
/// Without it recognition falls back to whisper.cpp or to the Python engine.
fn build_shim(lib: &Path, src_archive: &Path, tmp: &Path) -> anyhow::Result<()> {
    let root = untar_all(src_archive, &tmp.join("ct2-src"))?;
    let cpp = tmp.join("ct2shim.cpp");
    fs::write(&cpp, include_str!("../ct2shim/ct2shim.cpp"))?;
    let ct2 = fs::read_dir(lib)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libctranslate2") || n == "ctranslate2.dll"))
        .context("no CTranslate2 library in the library directory")?;
    let out = lib.join(native::lib_file("ct2shim"));
    if cfg!(windows) { msvc_shim(&cpp, &root.join("include"), &ct2, &out, tmp) } else { unix_shim(&cpp, &root.join("include"), &ct2, &out) }
}

fn unix_shim(cpp: &Path, include: &Path, ct2: &Path, out: &Path) -> anyhow::Result<()> {
    let Some(cxx) = std::env::var("CXX").ok().or_else(|| ["g++", "clang++", "c++"].into_iter().find(|c| which(c)).map(String::from))
    else {
        return no_compiler("sudo apt install build-essential (или xcode-select --install)");
    };
    say(format!("собираю обёртку над CTranslate2 ({cxx})"));
    let r = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-shared"])
        .arg(cpp)
        .arg("-I")
        .arg(include)
        .arg(ct2)
        // RPATH, а не RUNPATH: он же покрывает зависимости самой libctranslate2
        .args(["-Wl,--disable-new-dtags,-rpath,$ORIGIN", "-o"])
        .arg(out)
        .output()
        .with_context(|| format!("cannot run {cxx}"))?;
    finish(r)
}

/// The wheels ship DLLs without import libraries, and MSVC links against those.
/// One is made from the DLL's own export table; the `LIBRARY` line matters,
/// since otherwise the name comes from the .def file and whoever links against
/// it looks for a DLL that does not exist.
fn import_lib(dll: &Path, stem: &str) -> String {
    let name = dll.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    format!(
        "dumpbin /exports \"{dll}\" > {stem}-exports.txt || exit /b 1\r\n\
         powershell -NoProfile -Command \"$n = Select-String -Path {stem}-exports.txt -Pattern          '^\\s+\\d+\\s+[0-9A-F]+\\s+[0-9A-F]{{8}}\\s+(\\S+)' -AllMatches | ForEach-Object          {{ $_.Matches[0].Groups[1].Value }}; @('LIBRARY {name}', 'EXPORTS') + $n |          Set-Content -Encoding ascii {stem}.def\" || exit /b 1\r\n\
         lib /nologo /def:{stem}.def /machine:x64 /out:{stem}.lib >nul || exit /b 1\r\n",
        dll = dll.display())
}

/// On Windows CTranslate2 exports C++ names mangled the Microsoft way, so the
/// wrapper is built by MSVC — and linked against an import library made from
/// the DLL's own export table, since the wheel ships no .lib. The `LIBRARY`
/// line matters: without it the import library takes its name from the .def
/// file, and the wrapper then looks for a DLL that does not exist.
fn msvc_shim(cpp: &Path, include: &Path, ct2: &Path, out: &Path, tmp: &Path) -> anyhow::Result<()> {
    let Some(vcvars) = vcvars() else {
        return no_compiler("поставить «Build Tools for Visual Studio» с компонентом C++                             (winget install Microsoft.VisualStudio.2022.BuildTools --override                             \"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\")");
    };
    say("собираю обёртку над CTranslate2 (MSVC)");
    let script = tmp.join("build_shim.bat");
    // Всё одной командой cmd: переменные MSVC живут только внутри своего процесса
    fs::write(&script, format!(
        "@echo off\r\n\
         call \"{vcvars}\" >nul || exit /b 1\r\n\
         cd /d \"{tmp}\" || exit /b 1\r\n\
         {imports}\
         cl /nologo /LD /EHsc /std:c++17 /O2 /I \"{include}\" \"{cpp}\" /Fe:\"{out}\" /link ctranslate2.lib || exit /b 1\r\n",
        vcvars = vcvars.display(), tmp = tmp.display(), imports = import_lib(ct2, "ctranslate2"),
        include = include.display(), cpp = cpp.display(), out = out.display()))?;
    let r = Command::new("cmd").args(["/c"]).arg(&script).output().context("cannot run cmd")?;
    finish(r)
}

/// Two OpenMP runtimes in one process is what Intel's refuses to live with:
/// `libiomp5md.dll` (CTranslate2 brings it) aborts the program when it finds
/// `libomp.dll` (ggml brings it) already up, and a copy of one under the other's
/// name is still a second instance. They are the same runtime by origin and each
/// side uses only its standard entry points, so `libomp.dll` is replaced by a
/// stub that forwards them to `libiomp5md.dll`: one runtime, two names.
fn openmp_bridge(lib: &Path, tmp: &Path) -> anyhow::Result<()> {
    let (intel, llvm) = (lib.join("libiomp5md.dll"), lib.join("libomp.dll"));
    if !(intel.is_file() && llvm.is_file()) {
        return Ok(());
    }
    let Some(vcvars) = vcvars() else {
        // без компилятора остаётся ключ, который понимают оба рантайма (main::allow_two_openmp):
        // по замерам скорость та же, но Intel называет это неподдерживаемым (experiments/24)
        say("нет компилятора C++ — OpenMP останется в двух экземплярах, их разводит ключ KMP_DUPLICATE_LIB_OK");
        return Ok(());
    };
    let script = tmp.join("build_omp.bat");
    // Переадресацию линковщик делает только для символов, которые видит: отсюда
    // импортная библиотека Intel-рантайма в строке сборки.
    fs::write(&script, format!(
        "@echo off\r\n\
         call \"{vcvars}\" >nul || exit /b 1\r\n\
         cd /d \"{tmp}\" || exit /b 1\r\n\
         {imports}\
         del /q omp-imports.txt 2>nul\r\n\
         for %%F in (\"{lib}\\*.dll\") do dumpbin /imports:libomp.dll \"%%F\" >> omp-imports.txt\r\n\
         powershell -NoProfile -Command \"$n = Select-String -Path omp-imports.txt -Pattern          '\\b(?:omp_|__kmpc_|kmp_)\\w+' -AllMatches | ForEach-Object {{ $_.Matches }} | ForEach-Object          {{ $_.Value }} | Sort-Object -Unique; @('LIBRARY libomp.dll', 'EXPORTS') + ($n | ForEach-Object          {{ \\\"  $_=libiomp5md.$_\\\" }}) | Set-Content -Encoding ascii omp.def\" || exit /b 1\r\n\
         echo // forwarder > omp_stub.cpp\r\n\
         cl /nologo /LD omp_stub.cpp /link libiomp5md.lib /DEF:omp.def /OUT:\"{llvm}\" || exit /b 1\r\n",
        vcvars = vcvars.display(), tmp = tmp.display(), imports = import_lib(&intel, "libiomp5md"),
        lib = lib.display(), llvm = llvm.display()))?;
    let r = Command::new("cmd").args(["/c"]).arg(&script).output().context("cannot run cmd")?;
    finish(r)?;
    let n = fs::read_to_string(tmp.join("omp.def")).map(|d| d.lines().count().saturating_sub(2)).unwrap_or(0);
    say(format!("OpenMP один на процесс: libomp.dll переадресует {n} символов в libiomp5md.dll"));
    Ok(())
}

/// vcvars64.bat of the newest Visual Studio, as vswhere reports it.
fn vcvars() -> Option<PathBuf> {
    let pf = std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| "C:\\Program Files (x86)".into());
    let vswhere = PathBuf::from(&pf).join("Microsoft Visual Studio").join("Installer").join("vswhere.exe");
    let out = Command::new(&vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "-property", "installationPath"])
        .output()
        .ok()?;
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let bat = PathBuf::from(root).join("VC").join("Auxiliary").join("Build").join("vcvars64.bat");
    bat.is_file().then_some(bat)
}

fn no_compiler(how: &str) -> anyhow::Result<()> {
    say("нет компилятора C++ — обёртка над CTranslate2 не собрана; распознавание пойдёт мимо неё.");
    say(format!("  {how}, потом voicy setup снова"));
    Ok(())
}

fn finish(r: std::process::Output) -> anyhow::Result<()> {
    if !r.status.success() {
        bail!("обёртка не собралась:\n{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
    }
    Ok(())
}

fn which(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|d| d.join(cmd).is_file()))
        .unwrap_or(false)
}

// -------------------------------------------------------------------- шаги

/// Unpacks by what the archive is: the same releases come as .tar.gz and .zip.
fn extract(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    if archive.extension().is_some_and(|e| e == "zip") {
        unzip(archive, dest, keep)
    } else {
        untar(archive, dest, keep)
    }
}

/// The extension shared libraries have here, as the archives name them.
fn dyn_ext() -> &'static str {
    if cfg!(windows) { ".dll" } else if cfg!(target_os = "macos") { ".dylib" } else { ".so" }
}

/// The archives the engine libraries come out of, all in the downloads directory.
struct Libs {
    llama: Vec<PathBuf>,
    whisper: PathBuf,
    wheels: Vec<PathBuf>,
    ct2_src: Option<PathBuf>,
}

async fn plan_libs(http: &reqwest::Client, plan: &mut Plan, tmp: &Path) -> anyhow::Result<Libs> {
    let p = platform()?;
    let llama = p
        .llama_assets
        .iter()
        .map(|a| plan.archive(format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/{}", expand(a)), tmp))
        .collect();
    // whisper.cpp — из сборки для процессора: CUDA ей даёт ggml llama.cpp (experiments/21)
    let whisper =
        plan.archive(format!("https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_CPP}/{}", p.whisper_asset), tmp);
    let mut wheels = vec![];
    for spec in p.wheels {
        wheels.push(plan.archive(wheel_url(http, &expand(spec), p.wheel_tag).await?, tmp));
    }
    // заголовки CTranslate2 для обёртки; на Windows она вкомпилирована в бинарник (build.rs)
    let ct2_src = (!cfg!(all(windows, target_env = "msvc")))
        .then(|| plan.archive(format!("https://github.com/OpenNMT/CTranslate2/archive/refs/tags/v{CT2}.tar.gz"), tmp));
    Ok(Libs { llama, whisper, wheels, ct2_src })
}

async fn libs(archives: Libs, tmp: &Path) -> anyhow::Result<()> {
    let p = platform()?;
    let lib = native::lib_dir_path();
    fs::create_dir_all(&lib)?;
    for archive in &archives.llama {
        extract(archive, &lib, |n| n.contains(dyn_ext()))?;
        // квантование весов Qwen3-TTS делает эта же сборка (scripts/convert_qwen.py)
        extract(archive, lib.parent().unwrap_or(&lib), |n| n == "llama-quantize" || n == "llama-quantize.exe")?;
    }
    let whisper = native::lib_file("whisper");
    extract(&archives.whisper, &lib, |n| n.starts_with(&whisper))?;
    for wheel in &archives.wheels {
        unzip(wheel, &lib, |n| n.contains(dyn_ext()) && p.wanted.iter().any(|w| n.starts_with(w)))?;
    }
    if cfg!(windows) {
        openmp_bridge(&lib, tmp)?;
    }
    // В колесе лежит libonnxruntime.so.1.30.0, а ищется короткое имя
    #[cfg(unix)]
    if !lib.join(native::lib_file("onnxruntime")).exists() {
        let so = lib.join(native::lib_file("onnxruntime"));
        let mut versioned: Vec<PathBuf> = fs::read_dir(&lib)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libonnxruntime.so.")))
            .collect();
        versioned.sort();
        if let Some(v) = versioned.last() {
            std::os::unix::fs::symlink(v.file_name().unwrap_or_default(), &so)?;
        }
    }
    if let Some(src) = &archives.ct2_src {
        build_shim(&lib, src, tmp)?;
    }
    say(format!("библиотеки — {} ({} файлов)", lib.display(), fs::read_dir(&lib)?.count()));
    Ok(())
}

/// What is left to do with the models once they are here.
struct Models {
    root: PathBuf,
    /// Колесо faster-whisper, из которого берётся Silero, и куда его положить.
    vad: Option<(PathBuf, PathBuf)>,
    stress: bool,
}

const WHISPER_FILES: [&str; 5] = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"];
const TURN_FILE: &str = "smart-turn-v3.2-cpu.onnx";
const SILERO: &str = "silero_vad_v6.onnx";

fn whisper_dir(root: &Path) -> PathBuf {
    root.join("whisper").join("faster-whisper-large-v3-turbo")
}

/// Говорящая часть — в том варианте, который попросили (q5_k по умолчанию).
fn talker_variant() -> String {
    std::env::var("TTS_GGUF_TALKER").unwrap_or_else(|_| "q5_k".into())
}

/// What Qwen3-TTS is read from: every one of these, or synthesis fails at its
/// first request.
fn qwen_files(variant: &str, stress: bool) -> Vec<String> {
    let mut files = vec![
        "tokenizer.json".to_string(),
        format!("qwen3_tts_talker.{variant}.gguf"),
        "qwen3_tts_predictor.q8_0.gguf".into(),
        "qwen3_tts_decoder.fp16.onnx".into(),
        "qwen3_tts_codec_encoder.fp32.onnx".into(),
        "qwen3_tts_codec_encoder.fp32.onnx.data".into(),
        "qwen3_tts_speaker_encoder.fp32.onnx".into(),
        "qwen3_tts_speaker_encoder.fp32.onnx.data".into(),
        "embeddings/proj_bias.npy".into(),
        "embeddings/proj_weight.npy".into(),
        "embeddings/text_embedding_projected.npy".into(),
    ];
    files.extend((0..16).map(|i| format!("embeddings/codec_embedding_{i}.npy")));
    if stress {
        files.push(crate::engines::STRESS_MARKER.into());
    }
    files
}

/// The Qwen3-TTS directory the server reads: `VOICY_TTS_GGUF_DIR`, else the
/// model with stress marks once setup has begun to put it here, else the
/// official weights if they are what is installed.
pub fn tts_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("VOICY_TTS_GGUF_DIR") {
        return PathBuf::from(dir);
    }
    let root = native::cache_dir().join("models");
    let (stress, base) = (root.join(crate::engines::TTS_STRESS_DIR), root.join(crate::engines::TTS_BASE_DIR));
    if !stress.is_dir() && base.is_dir() { base } else { stress }
}

/// What the in-process engines need and the cache does not have — checked
/// without the network. A download cut off halfway leaves a cache that looks
/// installed, and that is better said at start than at the first request.
pub fn missing() -> Vec<PathBuf> {
    let root = native::cache_dir().join("models");
    let whisper = whisper_dir(&root);
    let mut want: Vec<PathBuf> = WHISPER_FILES.iter().map(|f| whisper.join(f)).collect();
    want.push(root.join("turn").join(TURN_FILE));
    want.push(root.join("vad").join(SILERO));
    let tts = tts_dir();
    let stress = tts.file_name().is_some_and(|n| n == crate::engines::TTS_STRESS_DIR);
    want.extend(qwen_files(&talker_variant(), stress).iter().map(|f| tts.join(f)));
    let accent = native::accent::dir();
    want.extend(native::accent::FILES.iter().map(|f| accent.join(f)));
    want.push(accent.join(native::accent::DICTIONARY_FST));
    let mut out: Vec<PathBuf> = want.into_iter().filter(|p| !p.is_file()).collect();
    // библиотеки: по файлу на каждое имя, которое ищут движки
    if let Ok(p) = platform() {
        let lib = native::lib_dir_path();
        let names: Vec<String> = fs::read_dir(&lib)
            .map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        let own = ["llama", "ggml", "whisper", "onnxruntime"].map(native::lib_file);
        for prefix in p.wanted.iter().copied().chain(own.iter().map(String::as_str)) {
            if !names.iter().any(|n| n.starts_with(prefix)) {
                out.push(lib.join(prefix));
            }
        }
    }
    out
}

/// Whether setup has been run here at all: then what is missing is a download
/// cut short, not a machine that runs the Python engines instead.
pub fn started() -> bool {
    native::lib_dir_path().is_dir() || native::cache_dir().join("models").is_dir()
}

/// The missing files, for a person: a few by name and how to get the rest.
pub fn describe(missing: &[PathBuf]) -> String {
    let mut s = format!("установка не докачана: не хватает {} {}", missing.len(), files(missing.len()));
    for p in missing.iter().take(5) {
        s += &format!("\n  {}", p.display());
    }
    if missing.len() > 5 {
        s += &format!("\n  … и ещё {}", missing.len() - 5);
    }
    s + "\nДокачать: voicy setup (скачанное раньше не пропадёт)"
}

async fn plan_models(http: &reqwest::Client, plan: &mut Plan, tmp: &Path) -> anyhow::Result<Models> {
    let root = native::cache_dir().join("models");
    let whisper = whisper_dir(&root);
    for f in WHISPER_FILES {
        plan.hf(WHISPER_MODEL, "main", f, &whisper);
    }
    plan.hf(TURN_MODEL, "main", TURN_FILE, &root.join("turn"));

    let vad_dir = root.join("vad");
    let vad = if vad_dir.join(SILERO).exists() {
        None
    } else {
        let url = wheel_url(http, &format!("faster-whisper=={FASTER_WHISPER}"), &["py3-none"]).await.unwrap_or_else(|_| {
            format!("https://files.pythonhosted.org/packages/py3/f/faster-whisper/faster_whisper-{FASTER_WHISPER}-py3-none-any.whl")
        });
        Some((plan.archive(url, tmp), vad_dir))
    };

    // Qwen3-TTS: по умолчанию — модель, понимающая знаки ударения; исходные веса —
    // VOICY_TTS_GGUF_REPO=sknyazev/qwen3-tts-12hz-1.7b-base-gguf
    let repo = std::env::var("VOICY_TTS_GGUF_REPO").unwrap_or_else(|_| QWEN_STRESS_MODEL.into());
    let stress = repo == QWEN_STRESS_MODEL;
    let gguf = std::env::var_os("VOICY_TTS_GGUF_DIR").map(PathBuf::from).unwrap_or_else(|| {
        root.join(if stress { crate::engines::TTS_STRESS_DIR } else { crate::engines::TTS_BASE_DIR })
    });
    for f in qwen_files(&talker_variant(), stress) {
        plan.hf(&repo, "main", &f, &gguf);
    }
    // RUAccent: словари и четыре модели на закреплённой ревизии
    let accent = native::accent::dir();
    for f in native::accent::FILES {
        plan.hf(native::accent::REPO, native::accent::REVISION, f, &accent);
    }
    Ok(Models { root, vad, stress })
}

async fn models(m: Models) -> anyhow::Result<()> {
    if let Some((wheel, dir)) = &m.vad {
        fs::create_dir_all(dir)?;
        unzip(wheel, dir, |n| n == SILERO)?;
    }
    // из словаря ударений — FST
    let accent = native::accent::dir();
    if !accent.join(native::accent::DICTIONARY_FST).exists() {
        say("собираю словарь ударений (3.2 млн словоформ, один раз)");
        let dir = accent.clone();
        tokio::task::spawn_blocking(move || native::accent::build_dictionary(&dir)).await??;
    }
    let old = m.root.join(crate::engines::TTS_BASE_DIR);
    if m.stress && old.is_dir() && std::env::var_os("VOICY_TTS_GGUF_DIR").is_none() {
        say(format!("прежняя модель без ударений больше не используется, её можно удалить: {}", old.display()));
        say(format!("  вернуться к ней: VOICY_TTS_GGUF_REPO={QWEN_MODEL} voicy setup models"));
    }
    say(format!("модели — {}", m.root.display()));
    Ok(())
}

/// `yes` — согласие на скачивание дано заранее; без него план показывается и
/// ждёт ответа.
pub async fn run(what: &str, yes: bool) -> anyhow::Result<()> {
    let tmp = native::cache_dir().join("downloads");
    fs::create_dir_all(&tmp)?;
    // read_timeout: зависшее соединение должно стать ошибкой и докачкой, а не вечным ожиданием;
    // прокси — из переменных среды и системных настроек, как у браузера
    let http = reqwest::Client::builder()
        .user_agent("voicy-setup")
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .build()?;
    say("смотрю, чего не хватает");
    let mut plan = Plan::default();
    let libs_plan = if what != "models" { Some(plan_libs(&http, &mut plan, &tmp).await?) } else { None };
    let models_plan = if what != "libs" { Some(plan_models(&http, &mut plan, &tmp).await?) } else { None };
    if plan.items.is_empty() {
        say("всё нужное уже скачано");
    } else {
        plan.measure(&http).await;
        plan.show(libs_plan.as_ref().map(|_| native::lib_dir_path()).as_deref());
        confirm(yes)?;
        plan.fetch_all(&http).await?;
    }

    if let Some(archives) = libs_plan {
        libs(archives, &tmp).await?;
    }
    if let Some(m) = models_plan {
        models(m).await?;
    }
    let left = missing();
    if what == "all" && !left.is_empty() {
        bail!("{}", describe(&left));
    }
    say(format!("скачанные архивы можно удалить: {}", tmp.display()));
    Ok(())
}
