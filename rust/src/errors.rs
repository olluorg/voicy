//! Errors in the shape OpenAI clients expect.
//!
//! The official clients look for `{"error": {"message", "type", "param", "code"}}`
//! and, not finding it, show the caller a bare status line. So every error —
//! raised on purpose, a validation failure, an unexpected one — leaves the server
//! in OpenAI's shape. Validation failures are 400, as OpenAI returns them.

use axum::extract::{FromRequest, Request};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::host::HostError;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub param: Option<String>,
    pub headers: Vec<(&'static str, String)>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message: message.into(),
            param: None,
            headers: vec![],
        }
    }

    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(500, message)
    }

    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// status → (type, code), как у OpenAI
fn kind(status: u16) -> (&'static str, Option<&'static str>) {
    match status {
        400 => ("invalid_request_error", None),
        401 => ("invalid_request_error", Some("invalid_api_key")),
        404 => ("invalid_request_error", Some("not_found")),
        405 => ("invalid_request_error", Some("method_not_allowed")),
        409 => ("invalid_request_error", Some("conflict")),
        410 => ("invalid_request_error", Some("gone")),
        413 => ("invalid_request_error", Some("too_large")),
        429 => ("requests", Some("rate_limit_exceeded")),
        s if s >= 500 => ("server_error", None),
        _ => ("invalid_request_error", None),
    }
}

pub fn body(status: u16, message: &str, param: Option<&str>) -> Value {
    let (t, code) = kind(status);
    json!({"error": {"message": message, "type": t, "param": param, "code": code}})
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status.as_u16();
        let mut resp = (self.status, axum::Json(body(status, &self.message, self.param.as_deref())))
            .into_response();
        let h: &mut HeaderMap = resp.headers_mut();
        for (name, value) in self.headers {
            if let Ok(v) = HeaderValue::from_str(&value) {
                h.insert(name, v);
            }
        }
        resp
    }
}

impl From<HostError> for ApiError {
    fn from(e: HostError) -> Self {
        match e.kind.as_str() {
            "unsupported" | "bad_audio" => ApiError::bad(e.message),
            "cancelled" => ApiError::new(409, "cancelled"),
            _ => ApiError::internal(e.message),
        }
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::internal(e.to_string())
    }
}

/// A JSON body whose failures read like FastAPI's: `input: Field required`, with
/// the field as `param` — but with status 400, as OpenAI answers.
pub struct Json<T>(pub T);

impl<S, T> FromRequest<S> for Json<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = match axum::body::Bytes::from_request(req, state).await {
            Ok(b) => b,
            Err(e) => return Err(ApiError::bad(e.body_text())),
        };
        let de = &mut serde_json::Deserializer::from_slice(&bytes);
        match serde_path_to_error::deserialize::<_, T>(de) {
            Ok(v) => Ok(Json(v)),
            Err(e) => {
                let inner = e.inner().to_string();
                let mut param = e.path().to_string();
                let mut message = inner.clone();
                // «missing field `input`» — поле в тексте ошибки, а не в пути
                if let Some(rest) = inner.strip_prefix("missing field `") {
                    if let Some(name) = rest.split('`').next() {
                        param = if param == "." { name.to_string() } else { format!("{param}.{name}") };
                        message = "Field required".into();
                    }
                }
                let message = message.split(" at line ").next().unwrap_or(&message).to_string();
                let param = (param != "." && !param.is_empty()).then_some(param);
                let mut err = ApiError::bad(match &param {
                    Some(p) => format!("{p}: {message}"),
                    None => message,
                });
                err.param = param;
                Err(err)
            }
        }
    }
}
