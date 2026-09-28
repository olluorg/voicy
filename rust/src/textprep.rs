//! Text preparation before synthesis — voicy_core::textprep, with stress
//! marks off the async runtime: the first call loads RUAccent.

pub use voicy_core::textprep::*;

/// Stress marks on `text` (voicy_core::textprep::stress), off the runtime.
pub async fn stress(text: &str) -> String {
    let owned = text.to_string();
    tokio::task::spawn_blocking(move || voicy_core::textprep::stress(&owned))
        .await
        .unwrap_or_else(|_| text.to_string())
}

/// Everything before synthesis, in order: dictionary, legato, stress marks.
pub async fn for_speech(text: &str, use_dictionary: bool, use_legato: bool, use_stress: bool) -> String {
    let t = if use_dictionary || use_legato { prepared_text(text, use_dictionary, use_legato) } else { text.to_string() };
    if use_stress { stress(&t).await } else { t }
}
