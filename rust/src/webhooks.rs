//! Calling back when a job ends — server/webhooks.py.
//!
//! The summary of the job is POSTed to the URL the job was given, with a few
//! retries spread over minutes. With VOICY_WEBHOOK_SECRET set, the body is
//! signed: `X-Voicy-Signature: sha256=<hex HMAC-SHA256 of the raw body>`.

use std::sync::Arc;
use std::time::Duration;

use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;

use crate::jobs::Job;

const PAUSES: [u64; 5] = [0, 2, 10, 30, 120]; // перед каждой попыткой, секунд
const LOCAL: [&str; 4] = ["localhost", "127.0.0.1", "::1", "0.0.0.0"];

pub fn validate(url: &str) -> Result<(), String> {
    let ok = reqwest::Url::parse(url)
        .ok()
        .filter(|u| matches!(u.scheme(), "http" | "https") && u.host_str().is_some_and(|h| !h.is_empty()));
    ok.map(|_| ()).ok_or_else(|| "webhook_url must be an absolute http(s) URL".into())
}

pub fn notify(job: Arc<Job>, http: reqwest::Client, secret: String, summary: Value) {
    {
        let mut s = job.s.lock().unwrap();
        match s.webhook.as_mut() {
            Some(h) if !h.started => h.started = true,
            _ => return,
        }
    }
    let mut body = json!({"event": format!("job.{}", job.state())});
    if let (Some(b), Some(s)) = (body.as_object_mut(), summary.as_object()) {
        b.extend(s.clone());
    }
    let body = serde_json::to_vec(&body).expect("json");
    tokio::spawn(deliver(job, http, secret, body));
}

async fn deliver(job: Arc<Job>, http: reqwest::Client, secret: String, body: Vec<u8>) {
    let url = job.s.lock().unwrap().webhook.as_ref().map(|h| h.url.clone()).unwrap_or_default();
    let local = reqwest::Url::parse(&url).ok().and_then(|u| u.host_str().map(String::from))
        .is_some_and(|h| LOCAL.contains(&h.trim_matches(['[', ']'])));
    // Локальный получатель — мимо прокси, иначе HTTP_PROXY превратит вызов в 503.
    let client = if local {
        reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(10)).build().unwrap_or(http)
    } else {
        http
    };
    let state = job.state();
    for pause in PAUSES {
        tokio::time::sleep(Duration::from_secs(pause)).await;
        let mut req = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("User-Agent", "voicy-webhook")
            .header("X-Voicy-Event", format!("job.{state}"))
            .header("X-Voicy-Job", &job.id)
            .body(body.clone());
        if !secret.is_empty() {
            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac");
            mac.update(&body);
            req = req.header("X-Voicy-Signature", format!("sha256={}", hex::encode(mac.finalize().into_bytes())));
        }
        if let Some(h) = job.s.lock().unwrap().webhook.as_mut() {
            h.attempts += 1;
        }
        let outcome = match req.send().await {
            Ok(r) if r.status().is_success() => None,
            Ok(r) => Some(format!("HTTP {}", r.status().as_u16())),
            Err(e) => Some(e.to_string()),
        };
        let mut s = job.s.lock().unwrap();
        let Some(h) = s.webhook.as_mut() else { return };
        match outcome {
            None => {
                h.delivered = Some(true);
                h.error = None;
                return;
            }
            Some(err) => h.error = Some(err),
        }
    }
    if let Some(h) = job.s.lock().unwrap().webhook.as_mut() {
        h.delivered = Some(false);
    }
}
