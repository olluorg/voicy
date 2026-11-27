//! Named voices: a reference clip `<name>.wav` plus its
//! transcript in `voices.json`, in the same directory and the same format the
//! Python server kept, so its voices carry over as they are.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value, json};

pub const SAMPLE_RATE: u32 = 24000;

/// The voices that ship, inside the binary. Все — записи LibriVox
/// (общественное достояние); чтец и книга — в `note` индекса.
pub const SHIPPED_VOICES: [(&str, &[u8]); 7] = [
    ("turgenev.wav", include_bytes!("../assets/voices/turgenev.wav")),
    ("dostoevsky.wav", include_bytes!("../assets/voices/dostoevsky.wav")),
    ("akhmatova.wav", include_bytes!("../assets/voices/akhmatova.wav")),
    ("aleichem.wav", include_bytes!("../assets/voices/aleichem.wav")),
    ("rogachevsky.wav", include_bytes!("../assets/voices/rogachevsky.wav")),
    ("vekhi.wav", include_bytes!("../assets/voices/vekhi.wav")),
    ("ohotnik.wav", include_bytes!("../assets/voices/ohotnik.wav")),
];
pub const SHIPPED_INDEX: &[u8] = include_bytes!("../assets/voices/voices.json");

/// Голоса, которые раздавали до появления `_shipped` в индексе.
const SHIPPED_BEFORE_TRACKING: [&str; 2] = ["turgenev", "dostoevsky"];

/// Put the voices that ship into `dir`. Новые голоса выпуска добавляются и в
/// уже заведённый каталог: прежде он заполнялся один раз, и обновившийся
/// пользователь новых голосов не видел. Свои голоса пользователя не трогаются,
/// а поставляемый голос, который уже предлагался (`_shipped`) и был удалён,
/// не возвращается.
pub fn seed(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let shipped: Map<String, Value> = serde_json::from_slice(SHIPPED_INDEX).expect("voices.json в бинарнике");
    let path = dir.join("voices.json");
    let fresh = !path.exists();
    let mut idx = if fresh {
        Map::new()
    } else {
        match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
            Some(Value::Object(m)) => m,
            _ => return Ok(()), // индекс испорчен — не перезаписываем чужие голоса
        }
    };
    let mut offered: Vec<String> = match idx.get("_shipped").and_then(Value::as_array) {
        Some(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        None if fresh => Vec::new(),
        None => SHIPPED_BEFORE_TRACKING.iter().map(|s| s.to_string()).collect(),
    };
    let before = (idx.len(), offered.len());
    for (file, data) in SHIPPED_VOICES {
        let name = file.trim_end_matches(".wav");
        if offered.iter().any(|n| n == name) {
            continue;
        }
        let wav = dir.join(file);
        if !wav.exists() && !idx.contains_key(name) {
            std::fs::write(&wav, data)?;
            if let Some(meta) = shipped.get(name) {
                idx.insert(name.to_string(), meta.clone());
            }
        }
        offered.push(name.to_string());
    }
    if fresh {
        if let Some(d) = shipped.get("_default") {
            idx.insert("_default".into(), d.clone());
        }
    }
    if fresh || (idx.len(), offered.len()) != before || !idx.contains_key("_shipped") {
        idx.insert("_shipped".into(), json!(offered));
        std::fs::write(&path, dump(&Value::Object(idx)))?;
    }
    Ok(())
}

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
    /// A voice from a clip of 8–14 s and its exact transcript: the model
    /// continues the reference, so a transcript cut mid-word costs accuracy
    /// (7.3% errors against 1.0%).
    pub fn from_file(path: impl Into<PathBuf>, text: impl Into<String>) -> Voice {
        let path = path.into();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        Voice { name, path, text: text.into(), note: String::new() }
    }

    pub fn as_json(&self) -> Value {
        json!({"name": self.name, "note": self.note, "reference_text": self.text,
               "file": self.path.file_name().and_then(|f| f.to_str()).unwrap_or_default()})
    }
}

pub struct Voices {
    pub dir: PathBuf,
}

impl Voices {
    /// The voices in the cache (`<cache>/voices`), seeded with the two that ship.
    pub fn in_cache() -> std::io::Result<Voices> {
        let dir = crate::native::cache_dir().join("voices");
        seed(&dir)?;
        Ok(Voices { dir })
    }

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

#[doc(hidden)]
/// JSON как у Python `json.dumps(ensure_ascii=False, indent=1)`.
pub fn dump(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    serde::Serialize::serialize(v, &mut ser).expect("json");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("voicy-seed-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn index(d: &Path) -> Map<String, Value> {
        serde_json::from_slice::<Value>(&std::fs::read(d.join("voices.json")).unwrap()).unwrap().as_object().unwrap().clone()
    }

    #[test]
    fn fresh_dir_gets_every_shipped_voice() {
        let d = dir("fresh");
        seed(&d).unwrap();
        let v = Voices { dir: d.clone() };
        assert_eq!(v.list().len(), SHIPPED_VOICES.len());
        assert_eq!(v.default().unwrap().name, "turgenev");
        assert!(v.get("akhmatova").is_some_and(|x| !x.text.is_empty()));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn old_dir_gets_new_voices_and_keeps_its_own() {
        let d = dir("old");
        std::fs::create_dir_all(&d).unwrap();
        // Каталог, разложенный прежней версией: два голоса и свой голос пользователя;
        // dostoevsky пользователь удалил.
        std::fs::write(d.join("turgenev.wav"), b"old").unwrap();
        std::fs::write(d.join("mine.wav"), b"mine").unwrap();
        std::fs::write(d.join("voices.json"),
                       br#"{"_default": "mine", "turgenev": {"text": "t"}, "mine": {"text": "m"}}"#).unwrap();
        seed(&d).unwrap();
        let idx = index(&d);
        assert_eq!(idx["_default"], "mine", "выбор пользователя не трогаем");
        assert_eq!(std::fs::read(d.join("turgenev.wav")).unwrap(), b"old");
        assert!(!d.join("dostoevsky.wav").exists(), "удалённый поставляемый голос не возвращается");
        assert!(d.join("akhmatova.wav").exists() && idx.contains_key("akhmatova"));
        assert_eq!(idx["_shipped"].as_array().unwrap().len(), SHIPPED_VOICES.len());

        // Пользователь удалил и новый голос — на следующем запуске он не вернётся.
        std::fs::remove_file(d.join("akhmatova.wav")).unwrap();
        seed(&d).unwrap();
        assert!(!d.join("akhmatova.wav").exists());
        std::fs::remove_dir_all(d).ok();
    }
}
