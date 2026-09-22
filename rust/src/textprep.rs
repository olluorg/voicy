//! Optional text preparation before synthesis — server/textprep.py, one to one.
//!
//! Terms are replaced with the spelling the engine reads correctly, from
//! data/pronunciation.json; "legato" drops commas inside short sentences, since
//! a comma is what makes the model stop mid-phrase. Both off by default.

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

pub fn load(dict: &Path) {
    let raw: Value = std::fs::read(dict)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
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
