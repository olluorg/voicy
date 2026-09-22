//! Named voices — server/voices.py: a reference clip `<name>.wav` plus its
//! transcript in `voices.json`, in the same directory and the same format, so
//! both servers read each other's voices.

use std::path::PathBuf;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value, json};

pub const SAMPLE_RATE: u32 = 24000;

pub fn name_ok(name: &str) -> bool {
    // буквы любого алфавита, цифры, дефис и подчёркивание — но не в начале:
    // ключи индекса на «_» служебные (`_default`)
    static NAME: OnceLock<Regex> = OnceLock::new();
    NAME.get_or_init(|| Regex::new(r"^[^\W_][\w-]{0,39}$").unwrap()).is_match(name)
}

pub const NAME_RULE: &str = "name: 1–40 letters, digits, '-' or '_', not starting with '_'";

#[derive(Clone, Debug)]
pub struct Voice {
    pub name: String,
    pub path: PathBuf,
    pub text: String,
    pub note: String,
}

impl Voice {
    pub fn as_json(&self) -> Value {
        json!({"name": self.name, "note": self.note, "reference_text": self.text,
               "file": self.path.file_name().and_then(|f| f.to_str()).unwrap_or_default()})
    }
}

pub struct Voices {
    pub dir: PathBuf,
}

impl Voices {
    fn index_path(&self) -> PathBuf {
        self.dir.join("voices.json")
    }

    fn index(&self) -> Map<String, Value> {
        std::fs::read(self.index_path())
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    }

    fn save(&self, idx: &Map<String, Value>) -> std::io::Result<()> {
        std::fs::write(self.index_path(), dump(&Value::Object(idx.clone())))
    }

    pub fn list(&self) -> Vec<Voice> {
        let idx = self.index();
        let mut wavs: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map(|d| {
                d.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                    .collect()
            })
            .unwrap_or_default();
        wavs.sort();
        wavs.into_iter()
            .map(|path| {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
                let meta = idx.get(&name).cloned().unwrap_or(Value::Null);
                Voice {
                    text: meta["text"].as_str().unwrap_or_default().to_string(),
                    note: meta["note"].as_str().unwrap_or_default().to_string(),
                    name,
                    path,
                }
            })
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Voice> {
        self.list().into_iter().find(|v| v.name == name)
    }

    pub fn default(&self) -> Option<Voice> {
        if let Some(wanted) = self.index().get("_default").and_then(Value::as_str) {
            if let Some(v) = self.get(wanted) {
                return Some(v);
            }
        }
        self.list().into_iter().next()
    }

    /// Register a clip already encoded as wav. The caller checks the name.
    pub fn add(&self, name: &str, wav: &[u8], text: &str, note: &str) -> std::io::Result<Voice> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{name}.wav"));
        std::fs::write(&path, wav)?;
        let mut idx = self.index();
        idx.insert(name.to_string(), json!({"text": text, "note": note}));
        self.save(&idx)?;
        Ok(Voice { name: name.into(), path, text: text.into(), note: note.into() })
    }
}

/// JSON как у Python `json.dumps(ensure_ascii=False, indent=1)`.
pub fn dump(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    serde::Serialize::serialize(v, &mut ser).expect("json");
    out
}
