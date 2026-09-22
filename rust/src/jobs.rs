//! Every request is a job with an id and a place in the queue — server/progress.py
//! and server/workqueue.py.
//!
//! One worker, one line, in submission order: the models share one card, and
//! running two at once would only split its memory. The OpenAI-compatible routes
//! submit a job and wait for it on the caller's behalf. Every number an event
//! carries is measured — or labelled as an estimate.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};
use tokio::sync::{Notify, broadcast};

use crate::errors::ApiError;
use crate::webhooks;

pub const TERMINAL: [&str; 3] = ["done", "error", "cancelled"];

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

pub fn round(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

#[derive(Clone)]
pub enum JobResult {
    Speech { data: Arc<Vec<u8>>, content_type: String, format: String, voice: String, seconds: f64 },
    Transcript(Value),
}

#[derive(Clone, Default)]
pub struct Webhook {
    pub url: String,
    pub attempts: u32,
    pub delivered: Option<bool>,
    pub error: Option<String>,
    pub started: bool,
}

pub struct JobState {
    pub state: String,
    pub started: Option<f64>,
    pub finished: Option<f64>,
    pub result: Option<JobResult>,
    pub error: Option<String>,
    pub error_status: u16,
    pub last: Value,
    pub webhook: Option<Webhook>,
}

pub struct Job {
    pub id: String,
    pub kind: String,
    pub created: f64,
    pub base_url: String,
    pub cancel_requested: AtomicBool,
    pub s: Mutex<JobState>,
    events: broadcast::Sender<Value>,
}

impl Job {
    fn stamp(&self, payload: &mut Map<String, Value>) {
        if !payload.contains_key("elapsed") {
            payload.insert("elapsed".into(), json!(round(now() - self.created, 2)));
        }
    }

    pub fn emit(&self, payload: Value) {
        let mut p = payload.as_object().cloned().unwrap_or_default();
        self.stamp(&mut p);
        let v = Value::Object(p);
        self.s.lock().unwrap().last = v.clone();
        let _ = self.events.send(v);
    }

    /// Terminal event. `last` is set together with `state`, so anyone who sees
    /// the final state also sees the final event.
    pub fn finish(&self, state: &str, payload: Value) {
        let mut p = payload.as_object().cloned().unwrap_or_default();
        p.insert("stage".into(), json!(state));
        self.stamp(&mut p);
        let v = Value::Object(p);
        {
            let mut s = self.s.lock().unwrap();
            s.finished = Some(now());
            s.last = v.clone();
            s.state = state.into();
        }
        let _ = self.events.send(v);
    }

    /// Subscribe first, then look at `state`.
    pub fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.events.subscribe()
    }

    pub fn state(&self) -> String {
        self.s.lock().unwrap().state.clone()
    }

    pub fn done(&self) -> bool {
        TERMINAL.contains(&self.state().as_str())
    }

    pub fn cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Relaxed)
    }

    pub async fn wait(&self) {
        let mut rx = self.subscribe();
        while !self.done() {
            match rx.recv().await {
                Ok(e) if e["stage"].as_str().is_some_and(|s| TERMINAL.contains(&s)) => break,
                Err(broadcast::error::RecvError::Closed) => break,
                _ => {}
            }
        }
    }
}

pub struct Registry {
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    ttl: f64,
}

impl Registry {
    pub fn new(ttl: f64) -> Self {
        Registry { jobs: Mutex::default(), ttl }
    }

    pub fn create(&self, kind: &str, base_url: &str) -> Arc<Job> {
        self.sweep();
        let (tx, _) = broadcast::channel(4096);
        let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
        let job = Arc::new(Job {
            id: id.clone(),
            kind: kind.into(),
            created: now(),
            base_url: base_url.trim_end_matches('/').into(),
            cancel_requested: AtomicBool::new(false),
            s: Mutex::new(JobState {
                state: "queued".into(),
                started: None,
                finished: None,
                result: None,
                error: None,
                error_status: 500,
                last: json!({}),
                webhook: None,
            }),
            events: tx,
        });
        self.jobs.lock().unwrap().insert(id, job.clone());
        job
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    pub fn discard(&self, id: &str) {
        self.jobs.lock().unwrap().remove(id);
    }

    pub fn all(&self) -> Vec<Arc<Job>> {
        let mut v: Vec<_> = self.jobs.lock().unwrap().values().cloned().collect();
        v.sort_by(|a, b| a.created.total_cmp(&b.created));
        v
    }

    /// Forget finished jobs after the TTL. Queued and running ones stay.
    fn sweep(&self) {
        let cutoff = now() - self.ttl;
        self.jobs.lock().unwrap().retain(|_, j| j.s.lock().unwrap().finished.is_none_or(|f| f >= cutoff));
    }
}

// ------------------------------------------------------------------ очередь

/// Why a piece of work stopped short.
pub enum WorkError {
    Cancelled,
    Failed(ApiError),
}

impl From<ApiError> for WorkError {
    fn from(e: ApiError) -> Self {
        WorkError::Failed(e)
    }
}

pub type Work = Box<dyn FnOnce(Arc<Job>) -> BoxFuture<'static, Result<Value, WorkError>> + Send>;

struct Line {
    pending: VecDeque<(Arc<Job>, Work)>,
    current: Option<Arc<Job>>,
}

pub struct Queue {
    line: Mutex<Line>,
    wake: Notify,
    pub max_pending: usize,
    http: reqwest::Client,
    secret: String,
}

impl Queue {
    pub fn start(max_pending: usize, secret: String) -> Arc<Queue> {
        let q = Arc::new(Queue {
            line: Mutex::new(Line { pending: VecDeque::new(), current: None }),
            wake: Notify::new(),
            max_pending,
            http: reqwest::Client::builder().timeout(Duration::from_secs(10)).build().expect("http client"),
            secret,
        });
        tokio::spawn(q.clone().worker());
        q
    }

    pub fn submit(&self, job: Arc<Job>, work: Work) -> Result<usize, String> {
        let mut line = self.line.lock().unwrap();
        if line.pending.len() >= self.max_pending {
            return Err(format!("queue is full ({} jobs waiting)", self.max_pending));
        }
        line.pending.push_back((job.clone(), work));
        let position = line.pending.len() - 1 + line.current.is_some() as usize;
        job.emit(json!({"stage": "queued", "queue_position": position}));
        drop(line);
        self.wake.notify_one();
        Ok(position)
    }

    /// How many jobs run before this one; None once it has started.
    pub fn position(&self, job: &Job) -> Option<usize> {
        let line = self.line.lock().unwrap();
        line.pending
            .iter()
            .position(|(j, _)| j.id == job.id)
            .map(|i| i + line.current.is_some() as usize)
    }

    pub fn pending(&self) -> usize {
        self.line.lock().unwrap().pending.len()
    }

    pub fn running(&self) -> Option<String> {
        self.line.lock().unwrap().current.as_ref().map(|j| j.id.clone())
    }

    /// Queued: removed at once. Running: stopped at the next step. Done: false.
    pub fn cancel(&self, job: &Arc<Job>) -> bool {
        let mut line = self.line.lock().unwrap();
        if let Some(i) = line.pending.iter().position(|(j, _)| j.id == job.id) {
            line.pending.remove(i);
            job.finish("cancelled", json!({}));
            Self::announce(&line);
            drop(line);
            self.notify(job.clone());
            return true;
        }
        if job.done() {
            return false;
        }
        // webhook отправит воркер, когда работа действительно остановится
        job.cancel_requested.store(true, Ordering::Relaxed);
        true
    }

    fn announce(line: &Line) {
        let base = line.current.is_some() as usize;
        for (i, (j, _)) in line.pending.iter().enumerate() {
            j.emit(json!({"stage": "queued", "queue_position": i + base}));
        }
    }

    fn notify(&self, job: Arc<Job>) {
        let summary = crate::api::summary(&job, self);
        webhooks::notify(job, self.http.clone(), self.secret.clone(), summary);
    }

    async fn worker(self: Arc<Self>) {
        loop {
            let next = {
                let mut line = self.line.lock().unwrap();
                let item = line.pending.pop_front();
                if let Some((job, _)) = &item {
                    line.current = Some(job.clone());
                    Self::announce(&line);
                }
                item
            };
            let Some((job, work)) = next else {
                self.wake.notified().await;
                continue;
            };
            {
                let mut s = job.s.lock().unwrap();
                s.state = "running".into();
                s.started = Some(now());
            }
            job.emit(json!({"stage": "running"}));
            let outcome = if job.cancelled() { Err(WorkError::Cancelled) } else { work(job.clone()).await };
            match outcome {
                Ok(extra) => job.finish("done", extra),
                Err(WorkError::Cancelled) => {
                    job.s.lock().unwrap().result = None;
                    job.finish("cancelled", json!({}));
                }
                Err(WorkError::Failed(e)) => {
                    {
                        let mut s = job.s.lock().unwrap();
                        s.error = Some(e.message.clone());
                        s.error_status = if e.status.as_u16() == 400 { 400 } else { 500 };
                    }
                    job.finish("error", json!({"error": e.message}));
                }
            }
            self.line.lock().unwrap().current = None;
            self.notify(job);
        }
    }
}
