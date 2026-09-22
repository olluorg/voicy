//! A transcript as plain data, and the formats OpenAI returns it in —
//! server/transcripts.py.

use axum::http::header;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::contexts::Resolved;

pub const FORMATS: [&str; 5] = ["json", "text", "srt", "vtt", "verbose_json"];

/// The engine's transcript with a context's replacements applied to the text.
/// Words keep what was heard: their timestamps belong to those words.
pub fn to_dict(tr: &Value, ctx: &Resolved) -> Value {
    let segments: Vec<Value> = tr["segments"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|s| {
                    let words: Vec<Value> = s["words"]
                        .as_array()
                        .map(|w| {
                            w.iter()
                                .map(|w| json!({"word": w["word"], "start": w["start"], "end": w["end"]}))
                                .collect()
                        })
                        .unwrap_or_default();
                    json!({"start": s["start"], "end": s["end"],
                           "text": ctx.fix(s["text"].as_str().unwrap_or_default()), "words": words})
                })
                .collect()
        })
        .unwrap_or_default();
    json!({"text": ctx.fix(tr["text"].as_str().unwrap_or_default()), "language": tr["language"],
           "duration": tr["duration"], "segments": segments})
}

fn srt_time(t: f64) -> String {
    let (h, rem) = ((t / 3600.0).floor(), t % 3600.0);
    let (m, s) = ((rem / 60.0).floor(), rem % 60.0);
    format!("{:02}:{:02}:{:02},{:03}", h as i64, m as i64, s as i64, ((s % 1.0) * 1000.0) as i64)
}

fn text(body: String, ctype: &'static str) -> Response {
    ([(header::CONTENT_TYPE, ctype)], body).into_response()
}

pub fn render(tr: &Value, response_format: Option<&str>) -> Response {
    let fmt = response_format.unwrap_or("json").to_lowercase();
    let empty = vec![];
    let segments = tr["segments"].as_array().unwrap_or(&empty);
    let f = |s: &Value, k: &str| s[k].as_f64().unwrap_or(0.0);
    match fmt.as_str() {
        "text" => text(tr["text"].as_str().unwrap_or_default().into(), "text/plain; charset=utf-8"),
        "srt" => {
            let mut lines = vec![];
            for (i, s) in segments.iter().enumerate() {
                lines.push((i + 1).to_string());
                lines.push(format!("{} --> {}", srt_time(f(s, "start")), srt_time(f(s, "end"))));
                lines.push(s["text"].as_str().unwrap_or_default().into());
                lines.push(String::new());
            }
            text(lines.join("\n"), "text/plain; charset=utf-8")
        }
        "vtt" => {
            let mut lines = vec!["WEBVTT".to_string(), String::new()];
            for s in segments {
                lines.push(format!("{} --> {}", srt_time(f(s, "start")).replace(',', "."),
                                   srt_time(f(s, "end")).replace(',', ".")));
                lines.push(s["text"].as_str().unwrap_or_default().into());
                lines.push(String::new());
            }
            text(lines.join("\n"), "text/vtt; charset=utf-8")
        }
        "verbose_json" => {
            let segs: Vec<Value> = segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut o = serde_json::Map::new();
                    o.insert("id".into(), json!(i));
                    if let Some(m) = s.as_object() {
                        o.extend(m.clone());
                    }
                    Value::Object(o)
                })
                .collect();
            axum::Json(json!({"task": tr.get("task").cloned().unwrap_or(json!("transcribe")),
                              "language": tr["language"], "duration": tr["duration"],
                              "text": tr["text"], "segments": segs}))
            .into_response()
        }
        _ => axum::Json(json!({"text": tr["text"]})).into_response(),
    }
}
