//! Recognition contexts — server/contexts.py: what the speaker is likely to say
//! (a prompt, hotwords) and how to write what the model still gets wrong
//! (replacements, exact and never guessed). `engineering` is built in: its
//! hotwords are the terms of the pronunciation dictionary.

use std::path::PathBuf;

use fancy_regex::{Regex, RegexBuilder};
use serde_json::{Map, Value, json};

use crate::{textprep, voices};

pub const BUILTIN: &str = "engineering";

#[derive(Clone, Debug)]
pub struct Context {
    pub name: String,
    pub note: String,
    pub prompt: String,
    pub hotwords: Vec<String>,
    pub replacements: Vec<(String, String)>,
    pub builtin: bool,
}

impl Context {
    pub fn as_json(&self) -> Value {
        let rep: Map<String, Value> =
            self.replacements.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        json!({"name": self.name, "note": self.note, "prompt": self.prompt,
               "hotwords": self.hotwords, "replacements": rep, "builtin": self.builtin})
    }
}

fn builtin() -> Context {
    let mut terms: Vec<String> = textprep::terms().iter().map(|t| t.term.clone()).collect();
    terms.sort_by_key(|t| t.to_lowercase());
    Context {
        name: BUILTIN.into(),
        note: "термины словаря произношений: Java, Kafka, Spring, SQL…".into(),
        prompt: String::new(),
        hotwords: terms,
        replacements: vec![],
        builtin: true,
    }
}

pub struct Contexts {
    pub dir: PathBuf,
}

impl Contexts {
    fn index_path(&self) -> PathBuf {
        self.dir.join("contexts.json")
    }

    fn load(&self) -> Map<String, Value> {
        std::fs::read(self.index_path())
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    }

    fn save(&self, idx: &Map<String, Value>) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        std::fs::write(self.index_path(), voices::dump(&Value::Object(idx.clone())))
    }

    pub fn list(&self) -> Vec<Context> {
        let mut names: Vec<String> = self.load().keys().cloned().collect();
        names.sort();
        std::iter::once(builtin()).chain(names.iter().filter_map(|n| self.get(n))).collect()
    }

    pub fn get(&self, name: &str) -> Option<Context> {
        if name == BUILTIN {
            return Some(builtin());
        }
        let e = self.load().get(name)?.clone();
        let strs = |v: &Value| -> Vec<String> {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
        };
        Some(Context {
            name: name.into(),
            note: e["note"].as_str().unwrap_or_default().into(),
            prompt: e["prompt"].as_str().unwrap_or_default().into(),
            hotwords: strs(&e["hotwords"]),
            replacements: e["replacements"]
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().into())).collect())
                .unwrap_or_default(),
            builtin: false,
        })
    }

    pub fn put(&self, name: &str, note: &str, prompt: &str, hotwords: Vec<String>,
               replacements: Vec<(String, String)>) -> Result<Context, String> {
        if !voices::name_ok(name) {
            return Err(voices::NAME_RULE.into());
        }
        if name == BUILTIN {
            return Err(format!("'{BUILTIN}' is built in and read-only"));
        }
        let hot: Vec<String> = hotwords.iter().map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect();
        let rep: Map<String, Value> = replacements
            .into_iter()
            .filter(|(k, _)| !k.trim().is_empty())
            .map(|(k, v)| (k.trim().to_string(), Value::String(v)))
            .collect();
        let mut idx = self.load();
        idx.insert(name.into(), json!({"note": note, "prompt": prompt, "hotwords": hot, "replacements": rep}));
        self.save(&idx).map_err(|e| e.to_string())?;
        Ok(self.get(name).expect("just saved"))
    }

    pub fn delete(&self, name: &str) -> bool {
        let mut idx = self.load();
        if idx.remove(name).is_none() {
            return false;
        }
        self.save(&idx).is_ok()
    }

    /// Merge a named context with per-request `prompt` and `hotwords`.
    /// `Err` carries the unknown context's name.
    pub fn resolve(&self, context: Option<&str>, prompt: Option<&str>,
                   hotwords: Option<&str>) -> Result<Resolved, String> {
        let ctx = match context.filter(|c| !c.is_empty()) {
            Some(c) => Some(self.get(c).ok_or_else(|| c.to_string())?),
            None => None,
        };
        let mut words: Vec<String> = vec![];
        let extra = hotwords.map(|h| h.split([',', '\n']).map(|w| w.trim().to_string()).collect::<Vec<_>>());
        for h in ctx.as_ref().map(|c| c.hotwords.clone()).unwrap_or_default().into_iter().chain(extra.unwrap_or_default()) {
            if !h.is_empty() && !words.contains(&h) {
                words.push(h);
            }
        }
        let prompts: Vec<String> = [ctx.as_ref().map(|c| c.prompt.clone()).unwrap_or_default(),
                                    prompt.unwrap_or_default().to_string()]
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        let mut pairs = ctx.map(|c| c.replacements).unwrap_or_default();
        // длинные первыми, чтобы «эс-ку-эль» не съел часть «ноу эс-ку-эль»
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs.sort_by_key(|(src, _)| std::cmp::Reverse(src.chars().count()));
        let replacements = pairs
            .into_iter()
            .filter_map(|(src, dst)| {
                RegexBuilder::new(&format!(r"(?<!\w){}(?!\w)", fancy_regex::escape(&src)))
                    .case_insensitive(true)
                    .build()
                    .ok()
                    .map(|re| (re, dst))
            })
            .collect();
        let prompt = prompts.join(" ");
        Ok(Resolved {
            prompt: (!prompt.is_empty()).then_some(prompt),
            hotwords: (!words.is_empty()).then_some(words),
            replacements,
        })
    }
}

/// What one recognition call uses. Prompt and terms stay apart: how to hand them
/// to the model is the engine's business.
#[derive(Clone, Default)]
pub struct Resolved {
    pub prompt: Option<String>,
    pub hotwords: Option<Vec<String>>,
    replacements: Vec<(Regex, String)>,
}

impl Resolved {
    pub fn fix(&self, text: &str) -> String {
        let mut text = text.to_string();
        for (re, dst) in &self.replacements {
            text = re.replace_all(&text, fancy_regex::NoExpand(dst)).into_owned();
        }
        text
    }
}
