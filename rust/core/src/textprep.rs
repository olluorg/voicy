//! Text preparation before synthesis, plus stress marks.
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
use std::sync::{Arc, OnceLock};

use fancy_regex::Regex;
use serde_json::{Value, json};

use crate::native::accent::{self, Accentuator};

pub struct Term {
    pub term: String,
    pattern: Regex,
    say: String,
}

static TERMS: OnceLock<Vec<Term>> = OnceLock::new();

/// The dictionary shipped inside the binary: assets/pronunciation.json of this
/// crate, which is where it is edited; the server reads that file when the
/// repository is at hand, so an edit needs no rebuild.
pub const BUILTIN: &str = include_str!("../assets/pronunciation.json");

/// The dictionary to use from now on: `dict`, or the built-in one when it is
/// None or unreadable. Only the first call counts; without one, the built-in
/// dictionary is loaded on first use.
pub fn load(dict: Option<&Path>) {
    let _ = TERMS.set(parse(dict));
}

fn parse(dict: Option<&Path>) -> Vec<Term> {
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
    items
        .into_iter()
        .map(|(term, say)| Term {
            pattern: Regex::new(&format!(r"(?<!\w){}(?!\w)", fancy_regex::escape(&term))).expect("term"),
            term,
            say,
        })
        .collect()
}

pub fn terms() -> &'static [Term] {
    TERMS.get_or_init(|| parse(None))
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

static ACCENT: OnceLock<Option<Arc<Accentuator>>> = OnceLock::new();

/// RUAccent, loaded once on first use; None when `voicy setup` has not put it
/// in place — then speech goes without marks, and that is said once.
fn accentuator() -> Option<Arc<Accentuator>> {
    ACCENT
        .get_or_init(|| {
            let dir = accent::dir();
            if !Accentuator::ready(&dir) {
                log::warn!("нет моделей RUAccent в {} — синтез без знаков ударения; voicy setup models", dir.display());
                return None;
            }
            let loaded = crate::native::lib_dir()
                .and_then(|lib| crate::native::init_onnx(&lib))
                .and_then(|_| Accentuator::load(&dir));
            match loaded {
                Ok(a) => Some(Arc::new(a)),
                Err(e) => {
                    log::warn!("RUAccent не загрузился ({e:#}) — синтез без знаков ударения");
                    None
                }
            }
        })
        .clone()
}

fn has_cyrillic(text: &str) -> bool {
    text.chars().any(|c| ('а'..='я').contains(&c) || ('А'..='Я').contains(&c) || c == 'ё' || c == 'Ё')
}

/// Stress marks on `text`, or `text` as it is when there is nothing to mark
/// or no RUAccent to mark it with. Blocks: the first call loads RUAccent.
pub fn stress(text: &str) -> String {
    if !has_cyrillic(text) {
        return text.to_string();
    }
    let Some(a) = accentuator() else {
        return text.to_string();
    };
    match a.mark(text) {
        Ok(marked) => particle_stress(&marked),
        Err(e) => {
            log::warn!("разметка ударений не удалась ({e:#}) — фраза без знаков");
            text.to_string()
        }
    }
}

/// «не бы́ло» → «не́ было»: with «был, было, были» the stress moves to the particle
/// (but «не была́»), and RUAccent marks the verb. A one-vowel word otherwise
/// carries no mark; this one does, since the norm puts the stress there. Whether
/// the model follows a mark on «не» could not be measured: neither meter2 nor
/// loudness tells «не́ было» from «не бы́ло» (experiments/29, speaker_ft_nebylo.py).
fn particle_stress(marked: &str) -> String {
    static NE_BYLO: OnceLock<Regex> = OnceLock::new();
    let re = re(r"(?<![А-Яа-яЁё])([Нн]е)(\s+)(б\x{301}?ы\x{301}?л(?:[ои]\x{301}?)?)(?![А-Яа-яЁё\x{301}])", &NE_BYLO);
    re.replace_all(marked, |c: &fancy_regex::Captures<str>| format!("{}\u{301}{}{}", &c[1], &c[2], c[3].replace('\u{301}', "")))
        .into_owned()
}

/// Everything before synthesis, in order: dictionary, legato, ё, stress marks.
pub fn for_speech(text: &str, use_dictionary: bool, use_legato: bool, use_stress: bool) -> String {
    let t = if use_dictionary || use_legato { prepared_text(text, use_dictionary, use_legato) } else { text.to_string() };
    let t = yo(&t);
    if use_stress { stress(&t) } else { t }
}

#[cfg(test)]
mod yo_tests {
    #[test]
    fn yo_safe_names_and_case() {
        assert_eq!(super::yo("Федор Иваныч, ее нет. ЕЩЕ"), "Фёдор Иваныч, её нет. ЕЩЁ");
        // имена из небезопасной части — только с заглавной; все/всё решает RUAccent
        assert_eq!(super::yo("федор и все"), "федор и все");
        assert_eq!(super::yo("Хрущев и Фёдор"), "Хрущёв и Фёдор");
    }

    #[test]
    fn particle_takes_the_stress() {
        assert_eq!(super::particle_stress("Её всё не бы́ло. Не бы́л, не бы́ли, не была́."),
                   "Её всё не́ было. Не́ был, не́ были, не была́.");
        assert_eq!(super::particle_stress("не бы́ло́ва"), "не бы́ло́ва");
    }
}

// ------------------------------------------------------------ ё

/// Words where е is read ё: assets/yo.txt, from eyo-kernel (MIT).
pub const YO: &str = include_str!("../assets/yo.txt");

/// е-form in lower case → (ё-form in lower case, only when capitalized in the text).
fn yo_words() -> &'static std::collections::HashMap<String, (String, bool)> {
    static WORDS: OnceLock<std::collections::HashMap<String, (String, bool)>> = OnceLock::new();
    WORDS.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for line in YO.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let (name, line) = match line.strip_prefix('@') { Some(l) => (true, l), None => (false, line) };
            let (stem, ends) = match line.split_once('(') {
                Some((s, e)) => (s, e.trim_end_matches(')').split('|').collect::<Vec<_>>()),
                None => (line, vec![""]),
            };
            for end in ends {
                let word = format!("{stem}{end}").to_lowercase();
                m.insert(word.replace('ё', "е"), (word, name));
            }
        }
        m
    })
}

/// ё where the text has е and the word is always read with ё: «Федор» → «Фёдор»,
/// «ее» → «её». Books and much of the web are typed without ё, and the model reads
/// what it sees: a mark on «Фе́дор» makes it say е. RUAccent restores ё only from
/// its own dictionary, which lacks names like Фёдор and surnames in -ёв; words
/// where both readings exist (все/всё) are left to it, since it decides by context.
pub fn yo(text: &str) -> String {
    let cyr = |c: char| ('а'..='я').contains(&c) || ('А'..='Я').contains(&c) || c == 'ё' || c == 'Ё' || c == '-';
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if !cyr(chars[i]) || chars[i] == '-' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let end = (i..chars.len()).find(|&k| !cyr(chars[k])).unwrap_or(chars.len());
        let word = &chars[i..end];
        let lower: String = word.iter().collect::<String>().to_lowercase();
        let fix = yo_words().get(&lower).filter(|(_, name)| !name || word[0].is_uppercase());
        match fix {
            Some((with_yo, _)) if !lower.contains('ё') => {
                for (c, y) in word.iter().zip(with_yo.chars()) {
                    out.push(if y == 'ё' { if c.is_uppercase() { 'Ё' } else { 'ё' } } else { *c });
                }
            }
            _ => out.extend(word),
        }
        i = end;
    }
    out
}
