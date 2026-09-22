//! Optional API key, the way OpenAI clients already send one — server/auth.py.
//!
//! Off unless VOICY_API_KEY is set (one key or several, comma-separated). Then
//! every `/v1/` route, WebSocket included, wants `Authorization: Bearer <key>`.
//! Browsers cannot put headers on an EventSource or a WebSocket, so
//! `?api_key=<key>` works too. `/health` and the console stay open.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{Method, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::App;
use crate::errors::ApiError;

fn key(req: &Request) -> String {
    let h = req.headers();
    if let Some(v) = h.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        return match v.get(..7) {
            Some(p) if p.eq_ignore_ascii_case("bearer ") => v[7..].trim().to_string(),
            _ => v.trim().to_string(),
        };
    }
    if let Some(v) = h.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return v.trim().to_string();
    }
    req.uri()
        .query()
        .and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("api_key=")))
        .map(urlencoding_decode)
        .unwrap_or_default()
}

fn urlencoding_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = (b[i] == b'%' && i + 2 < b.len())
            .then(|| std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()))
            .flatten();
        match (b[i], hex) {
            (_, Some(v)) => {
                out.push(v);
                i += 3;
            }
            (b'+', None) => {
                out.push(b' ');
                i += 1;
            }
            (c, None) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Сравнение за постоянное время: по нему нельзя подобрать ключ побайтово.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub async fn guard(State(app): State<Arc<App>>, req: Request, next: Next) -> Response {
    if app.keys.is_empty() || !req.uri().path().starts_with("/v1/") || req.method() == Method::OPTIONS {
        return next.run(req).await;
    }
    let k = key(&req);
    if !k.is_empty() && app.keys.iter().any(|x| same(x.as_bytes(), k.as_bytes())) {
        return next.run(req).await;
    }
    let message = if k.is_empty() { "Missing API key: pass 'Authorization: Bearer <key>'" } else { "Incorrect API key provided" };
    let websocket = req.headers().get(header::UPGRADE).is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"websocket"));
    if websocket {
        // до рукопожатия — отказ по HTTP, клиент WebSocket увидит 403
        return ApiError::new(403, message).into_response();
    }
    ApiError::new(401, message).with_header("www-authenticate", "Bearer").into_response()
}
