//! One voicy per machine — per cache, to be exact (`VOICY_CACHE`).
//!
//! The server, setup and the double-click launch hold a lock on a file in the
//! cache for as long as they run; the system drops it when the process ends,
//! however it ends. Two servers would fight over the port and the GPU, and setup
//! beside a running server fails on Windows: it cannot overwrite a DLL that the
//! server has loaded. Client commands (say, hear, status…) take no lock: they
//! talk to that one server.
//!
//! Who holds the lock is written next to it: on Windows the locked file itself
//! cannot be read by anyone else.

use std::fs::{self, File, TryLockError};
use std::path::PathBuf;

use crate::native;

fn dir() -> PathBuf {
    native::cache_dir().join("run")
}

/// The lock, held until this value is dropped or the process ends.
pub struct Instance {
    _file: File,
}

/// Whoever holds the lock, as they described themselves.
pub struct Holder {
    pub pid: u32,
    pub what: String,
    pub port: Option<u16>,
}

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.port {
            Some(port) => write!(f, "voicy уже запущен: {} на порту {port}", self.what)?,
            None => write!(f, "voicy уже запущен: {}", self.what)?,
        }
        if self.pid != 0 {
            write!(f, " (процесс {})", self.pid)?;
        }
        Ok(())
    }
}

impl Instance {
    /// Takes the lock, or says who has it.
    pub fn acquire(what: &str, port: Option<u16>) -> anyhow::Result<Result<Instance, Holder>> {
        Ok(lock()?.map(|file| {
            let me = Instance { _file: file };
            me.describe(what, port);
            me
        }))
    }

    /// What this process is doing now: a launch that installed first becomes a server.
    pub fn describe(&self, what: &str, port: Option<u16>) {
        let port = port.map(|p| p.to_string()).unwrap_or_default();
        let _ = fs::write(dir().join("voicy.owner"), format!("{}\n{what}\n{port}\n", std::process::id()));
    }
}

fn lock() -> anyhow::Result<Result<File, Holder>> {
    fs::create_dir_all(dir())?;
    let file = File::options().read(true).write(true).create(true).truncate(false).open(dir().join("voicy.lock"))?;
    match file.try_lock() {
        Ok(()) => Ok(Ok(file)),
        Err(TryLockError::WouldBlock) => Ok(Err(holder())),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

fn holder() -> Holder {
    let text = fs::read_to_string(dir().join("voicy.owner")).unwrap_or_default();
    let mut lines = text.lines();
    let pid = lines.next().and_then(|l| l.parse().ok()).unwrap_or(0);
    let what = lines.next().filter(|l| !l.is_empty()).unwrap_or("другой процесс").to_string();
    let port = lines.next().and_then(|l| l.parse().ok());
    Holder { pid, what, port }
}

/// Whoever holds the lock, if anyone does — without taking it.
pub fn running() -> Option<Holder> {
    // взятая на миг блокировка отпускается сразу, вместе с файлом
    lock().ok()?.err()
}
