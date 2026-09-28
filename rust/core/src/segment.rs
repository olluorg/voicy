//! Text cut into pieces a synthesiser can start on early — for speech that
//! should not wait for the whole of a text: an LLM's answer as it is written,
//! or a long text read piece by piece.
//!
//! The cuts go where a listener expects a pause: the first piece at the first
//! clause boundary (the first sound is what the listener waits for), the rest
//! at sentence ends; short sentences are joined, long ones cut at their last
//! clause boundary.

use std::sync::OnceLock;

use fancy_regex::Regex;

const MAX_CHUNK: usize = 200; // символов — длиннее режем по запятой
const MIN_CHUNK: usize = 25; // символов — короче склеиваем с соседом, если он уже есть
const FIRST_CLAUSE_WORDS: usize = 3; // первый кусок можно отрезать по запятой после стольких слов

fn sentence() -> &'static Regex {
    // Конец предложения — знак, пробел и слово не со строчной буквы: так «т. е.»,
    // «см. выше» и «3.5» не режутся.
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#"[.!?…]+["»)\]]*(?=\s+[^\sa-zа-яё])|\n"#).unwrap())
}

fn clause() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[,;:—–](?=\s)").unwrap())
}

fn byte_at(s: &str, chars: usize) -> usize {
    s.char_indices().nth(chars).map_or(s.len(), |(i, _)| i)
}

// --------------------------------------------------------------------- нарезка

/// Text in any pieces — tokens, lines, whole paragraphs; speakable chunks out.
#[derive(Default)]
pub struct Segmenter {
    buf: String,
    emitted: usize, // кусков отдано наружу
    cuts: usize,    // кусков отрезано, включая ещё не отданные
}

impl Segmenter {
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        self.take(false)
    }

    pub fn flush(&mut self) -> Vec<String> {
        self.take(true)
    }

    /// The whole text at once: short sentences join their neighbours on both sides.
    pub fn split(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        self.take(true)
    }

    pub fn reset(&mut self) {
        *self = Segmenter::default();
    }

    fn cut(&mut self, fin: bool) -> Option<String> {
        let m = sentence().find(&self.buf).ok().flatten().map(|m| (m.start(), m.end()));
        if self.cuts == 0 {
            // первый кусок — до первой запятой, если она раньше конца предложения
            let end = m.map_or(self.buf.len(), |(s, _)| s);
            let region = &self.buf[..end];
            let at = clause()
                .find_iter(region)
                .flatten()
                .find(|c| region[..c.start()].split_whitespace().count() >= FIRST_CLAUSE_WORDS)
                .map(|c| c.end());
            if let Some(at) = at {
                return Some(self.split_at(at));
            }
        }
        if let Some((_, end)) = m {
            return Some(self.split_at(end));
        }
        if self.buf.chars().count() > MAX_CHUNK {
            let head_end = byte_at(&self.buf, MAX_CHUNK);
            let head = &self.buf[..head_end];
            let at = clause()
                .find_iter(head)
                .flatten()
                .last()
                .map(|c| c.end())
                .or_else(|| head.rfind(' ').map(|i| i + 1))
                .filter(|&i| i > 0)
                .unwrap_or(head_end);
            return Some(self.split_at(at));
        }
        if fin && !self.buf.trim().is_empty() {
            return Some(self.split_at(self.buf.len()));
        }
        None
    }

    fn split_at(&mut self, at: usize) -> String {
        let rest = self.buf.split_off(at);
        let piece = std::mem::replace(&mut self.buf, rest);
        if !piece.trim().is_empty() {
            self.cuts += 1;
        }
        piece
    }

    fn take(&mut self, fin: bool) -> Vec<String> {
        let mut pieces = vec![];
        while let Some(p) = self.cut(fin) {
            if !p.trim().is_empty() {
                pieces.push(p.trim().to_string());
            }
        }
        let mut out: Vec<String> = vec![];
        for p in pieces {
            // короткое склеиваем со следующим, раз оно уже есть; первый кусок
            // потока не трогаем — он ради того, чтобы звук начался раньше
            let stream_first = self.emitted == 0 && out.len() == 1;
            match out.last_mut() {
                Some(last) if last.chars().count() < MIN_CHUNK && !stream_first => {
                    last.push(' ');
                    last.push_str(&p);
                }
                _ => out.push(p),
            }
        }
        self.emitted += out.len();
        out
    }
}
