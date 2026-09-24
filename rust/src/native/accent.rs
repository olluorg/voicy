//! Stress marks without Python: RUAccent 1.5.8 (ruaccent/accentuator, MIT),
//! one to one — the same dictionaries, the same four ONNX models, the same
//! decisions, down to the quirks. docs/adr/0023.
//!
//! RUAccent answers with its own normalised text: it drops every character
//! outside a short list («CI/CD» comes back as «CICD», «…» and «%» vanish).
//! That text is never spoken. `mark` keeps the caller's text and only adds to
//! it: U+0301 after the stressed vowel and ё where RUAccent restored it.
//!
//! The stages, per sentence, as in ruaccent.py `process_all_internal`:
//! ё from the dictionary and the ё-homograph model; stress homographs by the
//! pair classifier; which words carry stress at all by the token classifier;
//! stress from the 3.2-million-form dictionary, and from the character model
//! for words it lacks. The dictionary is an FST built once by `voicy setup`
//! (`build_dictionary`): ~70 MB of keys become a map read straight from disk.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde_json::Value;

pub const ACUTE: char = '\u{301}';
/// The pinned revision of ruaccent/accentuator: what `voicy setup` downloads.
pub const REPO: &str = "ruaccent/accentuator";
pub const REVISION: &str = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855";
pub const FILES: &[&str] = &[
    "nn/nn_accent/model.onnx", "nn/nn_accent/config.json", "nn/nn_accent/vocab.txt",
    "nn/nn_stress_usage_predictor/model.onnx", "nn/nn_stress_usage_predictor/config.json",
    "nn/nn_stress_usage_predictor/tokenizer.json",
    "nn/nn_yo_homograph_resolver/model.onnx", "nn/nn_yo_homograph_resolver/config.json",
    "nn/nn_yo_homograph_resolver/tokenizer.json",
    "nn/nn_omograph/turbo3.1/model.onnx", "nn/nn_omograph/turbo3.1/config.json",
    "nn/nn_omograph/turbo3.1/tokenizer.json",
    "dictionary/accents.json.gz", "dictionary/omographs.json.gz", "dictionary/yo_words.json.gz",
    "dictionary/yo_homographs.json.gz",
];
pub const DICTIONARY_FST: &str = "accents.fst";

const VOWELS: &str = "аеёиоуыэюяАЕЁИОУЫЭЮЯ";
const PUNCT: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

pub fn dir() -> PathBuf {
    super::cache_dir().join("models").join("ruaccent")
}

// ------------------------------------------------------------ текст

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// ruaccent.py `self.normalize`: keep only these characters.
fn kept(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || c.is_whitespace()
        || ('а'..='я').contains(&c)
        || ('А'..='Я').contains(&c)
        || "ёЁ—.,!?:;''(){}[]«»„“”-".contains(c)
}

fn count_vowels(w: &str) -> usize {
    w.chars().filter(|c| VOWELS.contains(*c)).count()
}

fn has_punctuation(w: &str) -> bool {
    w.chars().any(|c| PUNCT.contains(c))
}

/// text_postprocessor.fix_capital: the case of `source` onto `target` when the lengths agree.
fn fix_capital(source: &str, target: &str) -> String {
    let s: Vec<char> = source.chars().collect();
    let t: Vec<char> = target.chars().collect();
    if s.len() != t.len() {
        return target.to_string();
    }
    s.iter()
        .zip(t)
        .map(|(a, b)| if a.is_uppercase() { b.to_uppercase().collect::<String>() } else { b.to_lowercase().collect() })
        .collect()
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// ruaccent.py `delete_spaces_before_punc`, in its order.
fn delete_spaces_before_punc(text: &str) -> String {
    let mut t = text.to_string();
    for c in "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-".chars() {
        if c == '-' {
            t = t.replace(" -", "-").replace("- ", "-");
        }
        t = t.replace(&format!(" {c}"), &c.to_string());
    }
    t.replace('~', "-")
}

/// A word of TextPreprocessor.split_by_words: its char span in the sentence.
#[derive(Clone, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// TextPreprocessor.split_by_words. The pattern `\w*(?:\+\w+)*|[^\w\s]+` gives,
/// once its empty matches are set aside, runs of word characters (joined by
/// «+») and runs of what is neither a word character nor space.
fn split_words(chars: &[char]) -> Vec<Span> {
    let mut out = vec![];
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        let mut j = i;
        while j < chars.len() && is_word_char(chars[j]) {
            j += 1;
        }
        loop {
            // (?:\+\w+)*
            if j < chars.len() && chars[j] == '+' && j + 1 < chars.len() && is_word_char(chars[j + 1]) {
                j += 1;
                while j < chars.len() && is_word_char(chars[j]) {
                    j += 1;
                }
            } else {
                break;
            }
        }
        if j > start {
            out.push(Span { start, end: j });
            i = j;
            continue;
        }
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        while j < chars.len() && !is_word_char(chars[j]) && !chars[j].is_whitespace() {
            j += 1;
        }
        out.push(Span { start, end: j });
        i = j;
    }
    out
}

// ------------------------------------------------------------ модели

fn cpu_session(path: &Path, threads: usize) -> anyhow::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", path.display()))
}

fn labels(config: &Path) -> anyhow::Result<Vec<String>> {
    let v: Value = serde_json::from_slice(&std::fs::read(config)?)?;
    let m = v["id2label"].as_object().context("id2label")?;
    let mut out = vec![String::new(); m.len()];
    for (k, l) in m {
        out[k.parse::<usize>()?] = l.as_str().unwrap_or_default().to_string();
    }
    Ok(out)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn argmax(x: &[f32]) -> usize {
    // первый максимум, как numpy.argmax
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best
}

fn ids_tensor(v: &[i64]) -> anyhow::Result<Tensor<i64>> {
    Ok(Tensor::from_array((vec![1i64, v.len() as i64], v.to_vec()))?)
}

/// A token classifier with the HF pipeline's word grouping: stress_usage_model.py
/// and yo_homograph_model.py are the same code over different models.
struct TokenModel {
    session: Mutex<Session>,
    tok: tokenizers::Tokenizer,
    labels: Vec<String>,
    unk: Option<u32>,
    type_ids: bool,
}

impl TokenModel {
    fn load(dir: &Path, threads: usize) -> anyhow::Result<TokenModel> {
        let session = cpu_session(&dir.join("model.onnx"), threads)?;
        let type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        // transformers зовёт токенизатор без truncation, и усечение из tokenizer.json снимается
        tok.with_truncation(None).map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.with_padding(None);
        let unk = tok.token_to_id("[UNK]");
        Ok(TokenModel { session: Mutex::new(session), tok, labels: labels(&dir.join("config.json"))?, unk, type_ids })
    }

    /// One label per word group, AVERAGE aggregation.
    fn predict(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let enc = self.tok.encode_char_offsets(text, true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        let n = ids.len();
        let logits = {
            let mut s = self.session.lock().unwrap();
            let out = if self.type_ids {
                let tt: Vec<i64> = enc.get_type_ids().iter().map(|&i| i as i64).collect();
                s.run(ort::inputs!["input_ids" => ids_tensor(&ids)?, "attention_mask" => ids_tensor(&mask)?,
                                   "token_type_ids" => ids_tensor(&tt)?])?
            } else {
                s.run(ort::inputs!["input_ids" => ids_tensor(&ids)?, "attention_mask" => ids_tensor(&mask)?])?
            };
            out["logits"].try_extract_tensor::<f32>()?.1.to_vec()
        };
        let k = logits.len() / n.max(1);
        let chars: Vec<char> = text.chars().collect();
        let special = enc.get_special_tokens_mask();
        let tokens = enc.get_tokens();
        let offsets = enc.get_offsets();
        // (scores, is_subword) по несловарным токенам, в порядке следования
        let mut groups: Vec<Vec<Vec<f32>>> = vec![];
        for idx in 0..n {
            if special[idx] == 1 {
                continue;
            }
            let scores = softmax(&logits[idx * k..(idx + 1) * k]);
            let (s, e) = offsets[idx];
            let word_ref: String = chars[s.min(chars.len())..e.min(chars.len())].iter().collect();
            let mut is_subword = tokens[idx].chars().count() != word_ref.chars().count();
            if Some(enc.get_ids()[idx]) == self.unk {
                is_subword = false;
            }
            match groups.last_mut() {
                Some(g) if is_subword => g.push(scores),
                _ => groups.push(vec![scores]),
            }
        }
        Ok(groups
            .iter()
            .map(|g| {
                let mut avg = vec![0f32; k];
                for s in g {
                    for (a, v) in avg.iter_mut().zip(s) {
                        *a += v;
                    }
                }
                let avg: Vec<f32> = avg.into_iter().map(|a| a / g.len() as f32).collect();
                self.labels[argmax(&avg)].clone()
            })
            .collect())
    }
}

/// omograph_model.py: text with the word in <w></w>, one hypothesis; logits of two classes.
struct PairModel {
    session: Mutex<Session>,
    tok: tokenizers::Tokenizer,
}

impl PairModel {
    fn load(dir: &Path, threads: usize) -> anyhow::Result<PairModel> {
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: 512, ..Default::default() }))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tok.with_padding(None);
        Ok(PairModel { session: Mutex::new(cpu_session(&dir.join("model.onnx"), threads)?), tok })
    }

    fn logits(&self, text: &str, hypothesis: &str) -> anyhow::Result<[f32; 2]> {
        let enc = self.tok.encode((text, hypothesis), true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        let mut s = self.session.lock().unwrap();
        let out = s.run(ort::inputs!["input_ids" => ids_tensor(&ids)?, "attention_mask" => ids_tensor(&mask)?])?;
        let l = out["logits"].try_extract_tensor::<f32>()?.1.to_vec();
        Ok([l[0], l[1]])
    }
}

/// accent_model.py over char_tokenizer.py: [bos] + characters + [eos].
struct CharModel {
    session: Mutex<Session>,
    vocab: HashMap<String, i64>,
    labels: Vec<String>,
}

impl CharModel {
    fn load(dir: &Path) -> anyhow::Result<CharModel> {
        let vocab = std::fs::read_to_string(dir.join("vocab.txt"))?
            .lines()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as i64))
            .collect();
        Ok(CharModel { session: Mutex::new(cpu_session(&dir.join("model.onnx"), 1)?), vocab,
                       labels: labels(&dir.join("config.json"))? })
    }

    fn put_accent(&self, word: &str) -> anyhow::Result<String> {
        let id = |t: &str| self.vocab.get(t).copied().unwrap_or_else(|| self.vocab["[unk]"]);
        let mut ids = vec![id("[bos]")];
        ids.extend(lower(word).chars().map(|c| id(&c.to_string())));
        ids.push(id("[eos]"));
        let n = ids.len();
        let logits = {
            let mut s = self.session.lock().unwrap();
            let out = s.run(ort::inputs!["input_ids" => ids_tensor(&ids)?, "attention_mask" => ids_tensor(&vec![1; n])?,
                                         "token_type_ids" => ids_tensor(&vec![0; n])?])?;
            out["logits"].try_extract_tensor::<f32>()?.1.to_vec()
        };
        let k = logits.len() / n;
        let mut text: Vec<String> = word.chars().map(String::from).collect();
        for i in 0..n {
            let p = softmax(&logits[i * k..(i + 1) * k]);
            let label = &self.labels[argmax(&p)];
            let score = p.iter().cloned().fold(f32::MIN, f32::max);
            if label != "NO" && label != "STRESS_SECONDARY" && score >= 0.55 {
                // text[i - 1]: у [bos] это последний символ, как в Python; у [eos] там IndexError — пропускаем
                let at = if i == 0 { text.len().checked_sub(1) } else { Some(i - 1) };
                if let Some(at) = at.filter(|&a| a < text.len()) {
                    text[at] = format!("+{}", text[at]);
                }
            }
        }
        Ok(text.concat())
    }
}

// ------------------------------------------------------------ словари

fn read_gz_json(path: &Path) -> anyhow::Result<Value> {
    let mut s = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path)?).read_to_string(&mut s)?;
    Ok(serde_json::from_str(&s)?)
}

fn string_map(v: &Value) -> HashMap<String, String> {
    v.as_object()
        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
        .unwrap_or_default()
}

/// accents.json.gz → accents.fst: word → 1 + the index the «+» sits at (every
/// value is its key with one «+»; the 63 without one are left out, which is what
/// RUAccent does with them anyway — they go to the character model).
pub fn build_dictionary(dir: &Path) -> anyhow::Result<()> {
    let out = dir.join(DICTIONARY_FST);
    if out.exists() {
        return Ok(());
    }
    let v = read_gz_json(&dir.join("dictionary").join("accents.json.gz"))?;
    let m = v.as_object().context("accents.json.gz: not an object")?;
    let mut rows: Vec<(String, u64)> = m
        .iter()
        .filter_map(|(k, v)| {
            let v = v.as_str()?;
            v.chars().position(|c| c == '+').map(|p| (k.clone(), p as u64 + 1))
        })
        .collect();
    rows.push(("о".into(), 1)); // letters_accent: {'о': '+о'}
    rows.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    rows.dedup_by(|b, a| a.0 == b.0);
    let tmp = dir.join(format!("{DICTIONARY_FST}.tmp"));
    let mut w = fst::MapBuilder::new(std::io::BufWriter::new(std::fs::File::create(&tmp)?))?;
    for (k, p) in rows {
        w.insert(k.as_bytes(), p)?;
    }
    w.finish()?;
    std::fs::rename(tmp, out)?;
    Ok(())
}

// ------------------------------------------------------------ RUAccent

pub struct Accentuator {
    accents: fst::Map<memmap2::Mmap>,
    omographs: HashMap<String, Vec<String>>,
    yo_words: HashMap<String, String>,
    yo_homographs: HashMap<String, String>,
    stress: TokenModel,
    yo: TokenModel,
    omograph: PairModel,
    accent: CharModel,
}

/// Special words of omograph_model.py `group_words`.
const SPECIAL: &[&str] = &["балчуга", "вертела", "волоки", "волоку", "воронью", "выбродите", "вывозите", "выносите",
    "выноситесь", "выходите", "железы", "начала", "округа", "перепела", "развитая", "развитого", "развитое", "развитой",
    "развитом", "развитому", "развитою", "развитую", "развитые", "развитым", "развитыми", "развитых", "сторожа",
    "сторожи", "сторожу", "удало", "начался", "началась", "началось", "бутиках", "ожила", "создало", "коротки",
    "проклята", "роженица", "роженицы", "рожениц", "роженице", "роженицам", "роженицу", "роженицей", "роженицею",
    "роженицами", "роженицах", "пристава", "приставов", "приставам", "приставами", "приставах", "пережитое",
    "пережитого", "пережитые", "пережитых", "пережитому", "пережитым", "пережитыми", "пережитом", "нипоняла"];

impl Accentuator {
    pub fn ready(dir: &Path) -> bool {
        dir.join(DICTIONARY_FST).is_file() && FILES.iter().all(|f| dir.join(f).is_file())
    }

    pub fn load(dir: &Path) -> anyhow::Result<Accentuator> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(4);
        let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(dir.join(DICTIONARY_FST))?)? };
        let mut omographs: HashMap<String, Vec<String>> = read_gz_json(&dir.join("dictionary/omographs.json.gz"))?
            .as_object()
            .context("omographs")?
            .iter()
            .map(|(k, v)| {
                (k.clone(), v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default())
            })
            .collect();
        omographs.insert("коса".into(), vec!["к+оса".into(), "кос+а".into()]);
        Ok(Accentuator {
            accents: fst::Map::new(mmap)?,
            omographs,
            yo_words: string_map(&read_gz_json(&dir.join("dictionary/yo_words.json.gz"))?),
            yo_homographs: string_map(&read_gz_json(&dir.join("dictionary/yo_homographs.json.gz"))?),
            stress: TokenModel::load(&dir.join("nn/nn_stress_usage_predictor"), threads)?,
            yo: TokenModel::load(&dir.join("nn/nn_yo_homograph_resolver"), threads)?,
            omograph: PairModel::load(&dir.join("nn/nn_omograph/turbo3.1"), threads)?,
            accent: CharModel::load(&dir.join("nn/nn_accent"))?,
        })
    }

    /// One sentence, already normalised: the words with their spans and what
    /// RUAccent made of each («+» before the stressed vowel, ё restored).
    pub fn sentence(&self, sentence: &str) -> anyhow::Result<Vec<(Span, String)>> {
        // " - " → " ~ ": длина та же, поэтому и позиции слов те же
        let replaced = sentence.replace(" - ", " ~ ");
        let chars: Vec<char> = replaced.chars().collect();
        let spans = split_words(&chars);
        if spans.is_empty() {
            return Ok(vec![]);
        }
        let mut words: Vec<String> = spans.iter().map(|s| chars[s.start..s.end].iter().collect()).collect();
        let stress_usages = self.stress.predict(sentence)?;

        // _process_yo
        let lower_sentence = lower(sentence);
        let yo = if lower_sentence.contains('е') { Some(self.yo.predict(&lower_sentence)?) } else { None };
        for (i, w) in words.iter_mut().enumerate() {
            let word = w.clone();
            let lw = lower(&word);
            *w = fix_capital(&word, self.yo_words.get(&lw).map_or(word.as_str(), |s| s.as_str()));
            if yo.as_ref().and_then(|y| y.get(i)).is_some_and(|l| l == "YO") {
                *w = fix_capital(&word, self.yo_homographs.get(&lw).map_or(word.as_str(), |s| s.as_str()));
            }
        }

        self.omographs_step(&mut words)?;

        // _process_accent
        for (i, w) in words.iter_mut().enumerate() {
            if w.contains('+') {
                continue;
            }
            // у Python за пределами списка — IndexError; здесь слово считается ударным
            if stress_usages.get(i).is_some_and(|l| l != "STRESS") {
                continue;
            }
            let lw = lower(w);
            match self.accents.get(lw.as_bytes()) {
                Some(p) => {
                    let p = (p - 1) as usize;
                    let c: Vec<char> = w.chars().collect();
                    if p <= c.len() {
                        *w = c[..p].iter().chain(['+'].iter()).chain(c[p..].iter()).collect();
                    }
                }
                None if !has_punctuation(&lw) && count_vowels(&lw) > 1 => *w = self.accent.put_accent(w)?,
                None => {}
            }
        }
        Ok(spans.into_iter().zip(words).collect())
    }

    /// ruaccent.py `_process_omographs` with omograph_model.py `classify`.
    fn omographs_step(&self, words: &mut [String]) -> anyhow::Result<()> {
        let mut found = vec![]; // (позиция, варианты)
        for (i, w) in words.iter().enumerate() {
            if let Some(v) = self.omographs.get(w) {
                found.push((i, v.clone()));
            }
        }
        if found.is_empty() {
            return Ok(());
        }
        static PRE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let pre = PRE.get_or_init(|| regex::Regex::new(r"\s+([,.?!:;…])").unwrap());
        let mut texts = vec![];
        let mut hyps = vec![];
        for (pos, variants) in &found {
            let mut t: Vec<String> = words.to_vec();
            t[*pos] = format!(" <w>{}</w> ", t[*pos]);
            let joined = delete_spaces_before_punc(&t.join(" "));
            for v in variants {
                texts.push(pre.replace_all(&joined, "$1").into_owned());
                hyps.push(v.clone());
            }
        }
        let outs: Vec<String> = if found.iter().all(|(_, v)| v.len() % 2 == 0) {
            // пачкой: softmax по всей матрице, т.е. сравниваются сырые логиты класса 1, парами подряд
            let mut l1 = vec![];
            for (t, h) in texts.iter().zip(&hyps) {
                l1.push(self.omograph.logits(t, h)?[1]);
            }
            (0..hyps.len() / 2).map(|i| if l1[2 * i + 1] > l1[2 * i] { hyps[2 * i + 1].clone() } else { hyps[2 * i].clone() }).collect()
        } else {
            let groups = group_words(&hyps);
            let mut outs = vec![];
            let mut at = 0;
            for g in groups {
                let t = &texts[at];
                let mut best = (f32::NEG_INFINITY, 0);
                for (j, h) in g.iter().enumerate() {
                    let p = softmax(&self.omograph.logits(t, h)?)[1];
                    if p > best.0 {
                        best = (p, j);
                    }
                }
                outs.push(g[best.1].clone());
                at += g.len();
            }
            outs
        };
        for (k, (pos, _)) in found.iter().enumerate() {
            if let Some(o) = outs.get(k) {
                words[*pos] = o.clone();
            }
        }
        Ok(())
    }

    /// What `process_all` returns for one sentence — for the comparison with
    /// Python, never for speech.
    pub fn process_all(&self, text: &str) -> anyhow::Result<String> {
        let normal: String = text.chars().filter(|c| kept(*c)).collect();
        let replaced = normal.replace(" - ", " ~ ");
        let chars: Vec<char> = replaced.chars().collect();
        let words = self.sentence(&normal)?;
        if words.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::new();
        let mut last = 0;
        for (span, w) in &words {
            out.extend(chars[last..span.start].iter());
            out.push_str(w);
            last = span.end;
        }
        out.extend(chars[last..].iter());
        Ok(delete_spaces_before_punc(&out))
    }

    /// The caller's text with U+0301 after each stressed vowel and ё restored.
    /// Words that already carry a mark, words with ё and one-vowel words are
    /// left as they are: the model was trained on exactly this convention.
    pub fn mark(&self, text: &str) -> anyhow::Result<String> {
        let orig: Vec<char> = text.chars().collect();
        // нормализованный текст и откуда каждый его символ
        let mut normal = String::new();
        let mut from = vec![];
        for (i, c) in orig.iter().enumerate() {
            if kept(*c) {
                normal.push(*c);
                from.push(i);
            }
        }
        let mut replace: HashMap<usize, char> = HashMap::new();
        let mut acute_after: Vec<usize> = vec![];
        let nchars: Vec<char> = normal.chars().collect();
        for sentence in split_sentences(&nchars) {
            let offset = sentence.start;
            let s: String = nchars[sentence.clone()].iter().collect();
            for (span, w) in self.sentence(&s)? {
                let base: Vec<char> = w.chars().filter(|&c| c != '+').collect();
                let (a, b) = (offset + span.start, offset + span.end);
                if base.len() != b - a {
                    continue; // омограф или ё изменили длину — не трогаем
                }
                let src: Vec<usize> = (a..b).map(|k| from[k]).collect();
                // слово в исходном тексте цельное, без выброшенных нормализацией символов
                if src.windows(2).any(|p| p[1] != p[0] + 1) {
                    continue;
                }
                let end = src[src.len() - 1] + 1;
                if end < orig.len() && orig[end] == ACUTE {
                    continue;
                }
                for (j, &k) in src.iter().enumerate() {
                    // гипотезы омографов строчные: регистр — от исходной буквы
                    if (base[j] == 'ё' || base[j] == 'Ё') && (orig[k] == 'е' || orig[k] == 'Е') {
                        replace.insert(k, if orig[k] == 'Е' { 'Ё' } else { 'ё' });
                    }
                }
                let word: String = base.iter().collect();
                if word.to_lowercase().contains('ё') || count_vowels(&word) < 2 {
                    continue;
                }
                let mut j = 0;
                for c in w.chars() {
                    if c == '+' {
                        if j < base.len() && VOWELS.contains(base[j]) {
                            acute_after.push(src[j]);
                        }
                    } else {
                        j += 1;
                    }
                }
            }
        }
        let mut out = String::with_capacity(text.len() + acute_after.len() * 2);
        for (i, c) in orig.iter().enumerate() {
            out.push(*replace.get(&i).unwrap_or(c));
            if acute_after.contains(&i) {
                out.push(ACUTE);
            }
        }
        Ok(out)
    }
}

/// omograph_model.py `group_words`.
fn group_words(words: &[String]) -> Vec<Vec<String>> {
    let mut result = vec![];
    if words.is_empty() {
        return result;
    }
    let base = |w: &str| w.replace('+', "");
    let flush = |group: Vec<String>, b: &str, result: &mut Vec<Vec<String>>| {
        if SPECIAL.contains(&b) && group.len() > 3 {
            result.extend(group.chunks(3).map(|c| c.to_vec()));
        } else if group.len() > 3 && group.len() % 2 == 0 {
            result.extend(group.chunks(2).map(|c| c.to_vec()));
        } else {
            result.push(group);
        }
    };
    let mut group = vec![words[0].clone()];
    let mut current = base(&words[0]);
    for w in &words[1..] {
        let b = base(w);
        if b == current {
            group.push(w.clone());
        } else {
            flush(std::mem::take(&mut group), &current, &mut result);
            group = vec![w.clone()];
            current = b;
        }
    }
    flush(group, &current, &mut result);
    result
}

/// Sentences of the normalised text: after . ! ? and space. RUAccent cuts with
/// razdel; the models only look inside one sentence, so where exactly the cut
/// falls changes little, and the speech itself is not cut here at all.
fn split_sentences(chars: &[char]) -> Vec<std::ops::Range<usize>> {
    let mut out = vec![];
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if ".!?".contains(chars[i]) {
            let mut j = i + 1;
            while j < chars.len() && ".!?".contains(chars[j]) {
                j += 1;
            }
            if j < chars.len() && chars[j].is_whitespace() {
                out.push(start..j);
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if start < chars.len() {
        out.push(start..chars.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        let c: Vec<char> = s.chars().collect();
        split_words(&c).iter().map(|sp| c[sp.start..sp.end].iter().collect()).collect()
    }

    #[test]
    fn splits_like_python() {
        // сверено с TextPreprocessor.split_by_words
        assert_eq!(words("Привет, мир! Это «тест» ~ и всё..."),
                   ["Привет", ",", "мир", "!", "Это", "«", "тест", "»", "~", "и", "всё", "..."]);
        assert_eq!(words("CI/CD — 5 т.е. 3,5 кг"), ["CI", "/", "CD", "—", "5", "т", ".", "е", ".", "3", ",", "5", "кг"]);
        assert_eq!(words("  начало с пробела"), ["начало", "с", "пробела"]);
    }

    #[test]
    fn capital_and_punctuation() {
        assert_eq!(fix_capital("Все", "всё"), "Всё");
        assert_eq!(fix_capital("ВСЕ", "всё"), "ВСЁ");
        assert_eq!(delete_spaces_before_punc("а , б ~ в - г"), "а, б - в-г");
    }

    #[test]
    fn groups_like_python() {
        let g = group_words(&["з+амок".into(), "зам+ок".into(), "к+оса".into(), "кос+а".into(), "кос+а".into()]);
        assert_eq!(g.len(), 2);
        assert_eq!(g[1].len(), 3);
    }
}
