//! The engines, in a child process: `python -m engines.host` (server/engines/host.py).
//!
//! Frames both ways: u32 LE header length, a JSON header, and if the header has
//! `"bin": n`, n bytes of payload. Every request carries an id; replies with that
//! id are events, then one result or one error. Dropping a call that has not
//! ended cancels it in the child — a client that went away stops the work.

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct HostError {
    pub kind: String,
    pub message: String,
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl HostError {
    fn internal(message: impl Into<String>) -> Self {
        HostError { kind: "internal".into(), message: message.into() }
    }
}

pub enum Msg {
    Event { data: Value },
    Result { body: Value, bin: Bytes },
    Error(HostError),
}

type Pending = Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<Msg>>>>;

pub struct Host {
    next: AtomicU64,
    writer: tokio::sync::Mutex<ChildStdin>,
    pending: Pending,
    alive: Arc<AtomicBool>,
    _child: Mutex<Child>,
}

impl Host {
    /// Start the child. `python` runs `-m engines.host` inside `server_dir`.
    pub async fn spawn(python: &Path, server_dir: &Path) -> anyhow::Result<Arc<Host>> {
        let mut child = Command::new(python)
            .args(["-m", "engines.host"])
            .current_dir(server_dir)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("cannot start the engine host with {}", python.display()))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let pending: Pending = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        tokio::spawn(read_loop(stdout, pending.clone(), alive.clone()));
        let host = Arc::new(Host {
            next: AtomicU64::new(1),
            writer: tokio::sync::Mutex::new(stdin),
            pending,
            alive,
            _child: Mutex::new(child),
        });
        // первый ответ — знак, что движки импортировались
        host.call("info", json!({}), &[]).await.result().await.map_err(|e| {
            anyhow::anyhow!("the engine host did not start: {}", e.message)
        })?;
        Ok(host)
    }

    pub async fn call(self: &Arc<Self>, op: &str, args: Value, bin: &[u8]) -> Call {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        let mut call = Call { id, rx, host: self.clone(), ended: false };
        if !self.alive.load(Ordering::Relaxed) {
            let _ = tx.send(Msg::Error(HostError::internal("the engine host has exited")));
            call.ended = false;
            self.pending.lock().unwrap().insert(id, tx);
            return call;
        }
        self.pending.lock().unwrap().insert(id, tx);
        let mut header = json!({"id": id, "op": op, "args": args});
        if !bin.is_empty() {
            header["bin"] = json!(bin.len());
        }
        if let Err(e) = self.write(&header, bin).await {
            if let Some(tx) = self.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(Msg::Error(HostError::internal(format!("engine host: {e}"))));
            }
        }
        call
    }

    /// Shorthand for a call whose events do not matter.
    pub async fn run(self: &Arc<Self>, op: &str, args: Value, bin: &[u8]) -> Result<(Value, Bytes), HostError> {
        self.call(op, args, bin).await.result().await
    }

    async fn write(&self, header: &Value, bin: &[u8]) -> std::io::Result<()> {
        let raw = serde_json::to_vec(header).expect("json");
        let mut frame = Vec::with_capacity(4 + raw.len() + bin.len());
        frame.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        frame.extend_from_slice(&raw);
        frame.extend_from_slice(bin);
        let mut w = self.writer.lock().await;
        w.write_all(&frame).await?;
        w.flush().await
    }

    fn cancel_later(self: &Arc<Self>, target: u64) {
        let host = self.clone();
        tokio::spawn(async move {
            let id = host.next.fetch_add(1, Ordering::Relaxed);
            let _ = host.write(&json!({"id": id, "op": "cancel", "args": {"target": target}}), &[]).await;
        });
    }
}

async fn read_loop(mut out: ChildStdout, pending: Pending, alive: Arc<AtomicBool>) {
    let result: anyhow::Result<()> = async {
        loop {
            let mut len = [0u8; 4];
            out.read_exact(&mut len).await?;
            let mut raw = vec![0u8; u32::from_le_bytes(len) as usize];
            out.read_exact(&mut raw).await?;
            let header: Value = serde_json::from_slice(&raw)?;
            let bin = match header.get("bin").and_then(Value::as_u64) {
                Some(n) if n > 0 => {
                    let mut b = vec![0u8; n as usize];
                    out.read_exact(&mut b).await?;
                    Bytes::from(b)
                }
                _ => Bytes::new(),
            };
            let Some(id) = header.get("id").and_then(Value::as_u64) else { bail!("frame without id") };
            let msg = if let Some(data) = header.get("event").and(header.get("data")) {
                Msg::Event { data: data.clone() }
            } else if let Some(body) = header.get("result") {
                Msg::Result { body: body.clone(), bin }
            } else {
                let err = header.get("error").cloned().unwrap_or(Value::Null);
                Msg::Error(HostError {
                    kind: err["kind"].as_str().unwrap_or("internal").into(),
                    message: err["message"].as_str().unwrap_or("engine host error").into(),
                })
            };
            let last = !matches!(msg, Msg::Event { .. });
            let mut p = pending.lock().unwrap();
            if let Some(tx) = p.get(&id) {
                let _ = tx.send(msg);
            }
            if last {
                p.remove(&id);
            }
        }
    }
    .await;
    alive.store(false, Ordering::Relaxed);
    eprintln!("voicy: engine host stopped: {}", result.err().map(|e| e.to_string()).unwrap_or_default());
    for (_, tx) in pending.lock().unwrap().drain() {
        let _ = tx.send(Msg::Error(HostError::internal("the engine host has exited")));
    }
}

pub struct Call {
    id: u64,
    rx: mpsc::UnboundedReceiver<Msg>,
    host: Arc<Host>,
    ended: bool,
}

impl Call {
    /// The next message; after a result or an error there are no more.
    pub async fn next(&mut self) -> Msg {
        let msg = self
            .rx
            .recv()
            .await
            .unwrap_or_else(|| Msg::Error(HostError::internal("the engine host has exited")));
        if !matches!(msg, Msg::Event { .. }) {
            self.ended = true;
        }
        msg
    }

    pub async fn result(mut self) -> Result<(Value, Bytes), HostError> {
        loop {
            match self.next().await {
                Msg::Event { .. } => continue,
                Msg::Result { body, bin } => return Ok((body, bin)),
                Msg::Error(e) => return Err(e),
            }
        }
    }

    /// Ask the child to stop this call; its end still arrives through `next`.
    pub fn cancel(&self) {
        self.host.cancel_later(self.id);
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        if !self.ended {
            self.host.pending.lock().unwrap().remove(&self.id);
            self.host.cancel_later(self.id);
        }
    }
}

// ---------------------------------------------------------------- аудио

pub fn f32_bytes(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * 4);
    for v in x {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn f32_from(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
