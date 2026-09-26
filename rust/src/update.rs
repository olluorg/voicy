//! `voicy update` — the binary replaces itself with the latest release from
//! GitHub.
//!
//! What is new, how big it is and which file will be replaced is shown first,
//! and nothing is fetched without agreement — as with setup. The file is checked
//! against the checksum published beside it in the same release.
//!
//! A running program cannot be overwritten on Windows, but it can be renamed:
//! the old binary steps aside as `voicy.exe.old` and is removed on the next
//! start. On Unix the new file simply takes the old one's name; whoever runs
//! the old one keeps it until they stop.

use std::path::Path;

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use crate::setup;

const REPO: &str = "olluorg/voicy";

fn say(msg: impl AsRef<str>) {
    eprintln!("update: {}", msg.as_ref());
}

/// The file a release carries for this platform, as release.yml names it.
fn asset_name() -> anyhow::Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("voicy-linux-x64"),
        ("windows", "x86_64") => Ok("voicy-windows-x64.exe"),
        (os, arch) => bail!("готовой сборки под {os}-{arch} в выпусках нет — обновление из исходников: git pull и cargo build"),
    }
}

type Version = (u32, u32, u32);

fn parse(v: &str) -> Option<Version> {
    let mut it = v.trim().trim_start_matches('v').split('.').map(|p| p.parse::<u32>().ok());
    Some((it.next()??, it.next()??, it.next()??))
}

fn current() -> Version {
    parse(env!("CARGO_PKG_VERSION")).expect("version")
}

pub struct Release {
    pub tag: String,
    version: Version,
    url: String,
    checksum: String,
    size: Option<u64>,
    /// Первый абзац описания выпуска — из описания тега.
    pub summary: String,
}

impl Release {
    pub fn is_newer(&self) -> bool {
        self.version > current()
    }
}

/// The latest release and its file for this platform.
pub async fn latest(http: &reqwest::Client) -> anyhow::Result<Release> {
    let v: serde_json::Value = http
        .get(format!("https://api.github.com/repos/{REPO}/releases/latest"))
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .context("не достучаться до GitHub")?
        .error_for_status()
        .context("GitHub не отдал последний выпуск")?
        .json()
        .await?;
    let tag = v["tag_name"].as_str().context("у выпуска нет тега")?.to_string();
    let version = parse(&tag).with_context(|| format!("тег {tag} — не номер версии"))?;
    let name = asset_name()?;
    let asset = |n: &str| v["assets"].as_array().and_then(|a| a.iter().find(|x| x["name"] == n)).cloned();
    let file = asset(name).with_context(|| format!("в выпуске {tag} нет {name}"))?;
    let sum = asset(&format!("{name}.sha256")).with_context(|| format!("в выпуске {tag} нет контрольной суммы {name}"))?;
    let body = v["body"].as_str().unwrap_or_default();
    let summary = body.split("\n\n").map(str::trim).find(|p| !p.is_empty() && !p.starts_with('#')).unwrap_or_default();
    Ok(Release {
        tag,
        version,
        url: file["browser_download_url"].as_str().unwrap_or_default().into(),
        checksum: sum["browser_download_url"].as_str().unwrap_or_default().into(),
        size: file["size"].as_u64(),
        summary: summary.replace('\n', " "),
    })
}

/// A binary from `cargo build` is updated by git pull, not by a release: a
/// release would quietly replace what was built from the working tree.
fn built_here(exe: &Path) -> bool {
    let parts: Vec<String> = exe.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    parts.windows(2).any(|w| w[0] == "target" && (w[1] == "release" || w[1] == "debug"))
}

/// `voicy update`: show what is new, ask, replace.
pub async fn run(yes: bool, check: bool) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let http = setup::client()?;
    let rel = latest(&http).await?;
    if !rel.is_newer() {
        say(format!("у вас последняя версия — {}", env!("CARGO_PKG_VERSION")));
        return Ok(());
    }
    describe(&rel, &exe);
    if check {
        return Ok(());
    }
    if built_here(&exe) {
        bail!("{} собран из исходников — обновляйте его через git pull и cargo build", exe.display());
    }
    setup::confirm(yes, "voicy update")?;
    install(&http, &rel, &exe).await
}

pub fn describe(rel: &Release, exe: &Path) {
    say(format!("вышла версия {} (у вас {})", rel.tag.trim_start_matches('v'), env!("CARGO_PKG_VERSION")));
    if !rel.summary.is_empty() {
        say(format!("  {}", rel.summary));
    }
    let size = rel.size.map(|s| format!("{} МБ", s >> 20)).unwrap_or_else(|| "?".into());
    say(format!("скачать {size} с github.com/{REPO} и заменить {}", exe.display()));
}

/// Downloads, checks and puts the new binary in place of `exe`.
pub async fn install(http: &reqwest::Client, rel: &Release, exe: &Path) -> anyhow::Result<()> {
    setup::speak_as("update");
    let dir = exe.parent().context("у бинарника нет каталога")?;
    let name = exe.file_name().context("у бинарника нет имени")?.to_string_lossy().into_owned();
    let new = dir.join(format!("{name}.new"));
    // сначала проверить, что сюда вообще можно писать, — а не после скачивания
    std::fs::write(&new, b"").map_err(|e| no_write(dir, e))?;
    std::fs::remove_file(&new)?;
    setup::download(http, &rel.url, &new, rel.size).await?;

    let expected = http.get(&rel.checksum).send().await?.error_for_status()?.text().await?;
    let expected = expected.split_whitespace().next().unwrap_or_default().to_lowercase();
    let actual = hex::encode(Sha256::digest(std::fs::read(&new)?));
    if actual != expected {
        let _ = std::fs::remove_file(&new);
        bail!("скачанный файл не сходится с контрольной суммой выпуска {} — он не поставлен", rel.tag);
    }
    #[cfg(unix)]
    std::fs::set_permissions(&new, std::fs::metadata(exe)?.permissions())?;
    replace(exe, &new)?;
    say(format!("обновлено до {}: {}", rel.tag.trim_start_matches('v'), exe.display()));
    if let Some(h) = crate::instance::running() {
        say(format!("{h} — он пока на прежней версии; перезапустите его: voicy down и voicy up, или закройте окно и откройте снова"));
    }
    Ok(())
}

fn no_write(dir: &Path, e: std::io::Error) -> anyhow::Error {
    anyhow::anyhow!(
        "нет права писать в {} ({e}). Перенесите voicy в свою папку или обновите его от имени администратора",
        dir.display())
}

#[cfg(unix)]
fn replace(exe: &Path, new: &Path) -> anyhow::Result<()> {
    // rename атомарен: запущенная копия продолжает работать со старым файлом
    std::fs::rename(new, exe).with_context(|| format!("не заменить {}", exe.display()))
}

#[cfg(windows)]
fn replace(exe: &Path, new: &Path) -> anyhow::Result<()> {
    // запущенный exe нельзя перезаписать, но можно переименовать
    let mut old = old_name(exe, 0);
    let mut n = 0;
    while old.exists() && std::fs::remove_file(&old).is_err() {
        n += 1; // прежний .old ещё кем-то запущен — рядом, под другим номером
        old = old_name(exe, n);
    }
    std::fs::rename(exe, &old).with_context(|| format!("не отодвинуть {}", exe.display()))?;
    if let Err(e) = std::fs::rename(new, exe) {
        let _ = std::fs::rename(&old, exe);
        return Err(e).with_context(|| format!("не поставить новую версию на место {}", exe.display()));
    }
    Ok(())
}

#[cfg(windows)]
fn old_name(exe: &Path, n: u32) -> std::path::PathBuf {
    let name = exe.file_name().unwrap_or_default().to_string_lossy();
    exe.with_file_name(if n == 0 { format!("{name}.old") } else { format!("{name}.old{n}") })
}

/// What the last update left behind: the previous binary, once nothing runs it.
pub fn cleanup() {
    let Ok(exe) = std::env::current_exe() else { return };
    let Some(dir) = exe.parent() else { return };
    let Some(name) = exe.file_name().map(|n| n.to_string_lossy().into_owned()) else { return };
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.starts_with(&format!("{name}.old")) || n == format!("{name}.new") {
            let _ = std::fs::remove_file(e.path());
        }
    }
}
