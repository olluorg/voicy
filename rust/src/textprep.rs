//! Text preparation before synthesis — server/textprep.py, one to one, plus
//! stress marks.
//!
//! Terms are replaced with the spelling the engine reads correctly, from
//! data/pronunciation.json; "legato" drops commas inside short sentences, since
//! a comma is what makes the model stop mid-phrase. Both off by default.
//!
//! Stress marks are on by default where the model understands them: RUAccent
//! (native/accent.rs) puts U+0301 after the stressed vowel and restores ё.
//! A spelling from the dictionary that already carries a mark is kept — that is
//! how a term overrides RUAccent («дже́нерики»). docs/adr/0023.

use std::path::Path;
use std::sync::OnceLock;

use fancy_regex::Regex;
use serde_json::{Value, json};

pub struct Term {
    pub term: String,
    pattern: Regex,
    say: String,
}

static TERMS: OnceLock<Vec<Term>> = OnceLock::new();

/// The dictionary shipped inside the binary; data/pronunciation.json when the
/// repository is at hand — that is where it is edited.
pub const BUILTIN: &str = include_str!("../../data/pronunciation.json");

pub fn load(dict: Option<&Path>) {
    let bytes = dict.and_then(|d| std::fs::read(d).ok()).unwrap_or_else(|| BUILTIN.as_bytes().to_vec());
    let raw: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let mut items: Vec<(String, String)> = raw["terms"]
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(t, meta)| {
                    meta["say"].as_str().filter(|s| !s.is_empty()).map(|s| (t.clone(), s.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    // длинные термины первыми, иначе "SQL" съест часть "NoSQL"; порядок равных — как в файле
    items.sort_by_key(|(t, _)| std::cmp::Reverse(t.chars().count()));
    let terms = items
        .into_iter()
        .map(|(term, say)| Term {
            pattern: Regex::new(&format!(r"(?<!\w){}(?!\w)", fancy_regex::escape(&term))).expect("term"),
            term,
            say,
        })
        .collect();
    let _ = TERMS.set(terms);
}

pub fn terms() -> &'static [Term] {
    TERMS.get().map(Vec::as_slice).unwrap_or(&[])
}

pub fn apply_dictionary(text: &str) -> (String, Vec<String>) {
    let mut text = text.to_string();
    let mut hits = vec![];
    for t in terms() {
        let n = t.pattern.find_iter(&text).filter(|m| m.is_ok()).count();
        if n > 0 {
            text = t.pattern.replace_all(&text, fancy_regex::NoExpand(&t.say)).into_owned();
            hits.push(format!("{} → {}{}", t.term, t.say, if n > 1 { format!(" ×{n}") } else { String::new() }));
        }
    }
    (text, hits)
}

fn re(pattern: &'static str, cell: &'static OnceLock<Regex>) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("regex"))
}

/// Drop commas inside short sentences so the phrase is read in one breath.
pub fn legato(text: &str) -> String {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    static WORDS: OnceLock<Regex> = OnceLock::new();
    static COMMA: OnceLock<Regex> = OnceLock::new();
    let split = re(r"(?<=[.!?])\s+", &SPLIT);
    let words = re(r"[^\W\d_]+", &WORDS);
    let comma = re(r",\s+", &COMMA);
    let mut out = vec![];
    let mut last = 0;
    let mut pieces = vec![];
    for m in split.find_iter(text).flatten() {
        pieces.push(&text[last..m.start()]);
        last = m.end();
    }
    pieces.push(&text[last..]);
    for sentence in pieces {
        let n = words.find_iter(sentence).flatten().count();
        let s = if n <= 12 { comma.replace_all(sentence, " ").into_owned() } else { sentence.to_string() };
        let s = s.trim().to_string();
        if !s.is_empty() {
            out.push(s);
        }
    }
    out.join(" ")
}

pub fn prepare(text: &str, use_dictionary: bool, use_legato: bool) -> Value {
    let mut out = text.to_string();
    let mut hits = vec![];
    if use_dictionary {
        (out, hits) = apply_dictionary(&out);
    }
    if use_legato {
        out = legato(&out);
    }
    json!({"text": out, "changed": out != text, "replacements": hits})
}

pub fn prepared_text(text: &str, use_dictionary: bool, use_legato: bool) -> String {
    prepare(text, use_dictionary, use_legato)["text"].as_str().unwrap_or(text).to_string()
}

// ------------------------------------------------------------ ударения

static ACCENT: tokio::sync::OnceCell<Option<std::sync::Arc<crate::native::accent::Accentuator>>> =
    tokio::sync::OnceCell::const_new();

/// RUAccent, loaded once on first use; None when `voicy setup` has not put it
/// in place — then speech goes without marks, and that is said once.
async fn accentuator() -> Option<std::sync::Arc<crate::native::accent::Accentuator>> {
    ACCENT
        .get_or_init(|| async {
            let dir = crate::native::accent::dir();
            if !crate::native::accent::Accentuator::ready(&dir) {
                eprintln!("voicy: нет моделей RUAccent в {} — синтез без знаков ударения; voicy setup models", dir.display());
                return None;
            }
            let loaded = tokio::task::spawn_blocking(move || {
                crate::native::init_onnx(&crate::native::lib_dir()?)?;
                crate::native::accent::Accentuator::load(&dir)
            })
            .await;
            match loaded {
                Ok(Ok(a)) => Some(std::sync::Arc::new(a)),
                Ok(Err(e)) => {
                    eprintln!("voicy: RUAccent не загрузился ({e:#}) — синтез без знаков ударения");
                    None
                }
                Err(e) => {
                    eprintln!("voicy: RUAccent не загрузился ({e}) — синтез без знаков ударения");
                    None
                }
            }
        })
        .await
        .clone()
}

fn has_cyrillic(text: &str) -> bool {
    text.chars().any(|c| ('а'..='я').contains(&c) || ('А'..='Я').contains(&c) || c == 'ё' || c == 'Ё')
}

/// Stress marks on `text`, or `text` as it is when there is nothing to mark
/// or no RUAccent to mark it with.
pub async fn stress(text: &str) -> String {
    if !has_cyrillic(text) {
        return text.to_string();
    }
    let Some(a) = accentuator().await else {
        return text.to_string();
    };
    let owned = text.to_string();
    match tokio::task::spawn_blocking(move || a.mark(&owned)).await {
        Ok(Ok(marked)) => marked,
        Ok(Err(e)) => {
            eprintln!("voicy: разметка ударений не удалась ({e:#}) — фраза без знаков");
            text.to_string()
        }
        Err(_) => text.to_string(),
    }
}

/// Everything before synthesis, in order: dictionary, legato, stress marks.
pub async fn for_speech(text: &str, use_dictionary: bool, use_legato: bool, use_stress: bool) -> String {
    let t = if use_dictionary || use_legato { prepared_text(text, use_dictionary, use_legato) } else { text.to_string() };
    if use_stress { stress(&t).await } else { t }
}
