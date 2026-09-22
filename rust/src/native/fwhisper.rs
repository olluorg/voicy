//! Whisper as faster-whisper runs it, without Python.
//!
//! The model runs in CTranslate2 — the same prebuilt library faster-whisper
//! ships, reached through the C wrapper in rust/ct2shim. Everything around the
//! model is faster-whisper 1.2.1's Python, function for function, under the
//! same names: the log-mel, the 30-second windows, the prompt, the no-speech
//! check, word timings, the voice-detector pass for live speech. Called with
//! the settings server/engines/stt_whisper.py uses, it gives the same text.

use std::ffi::{CStr, CString, c_char, c_long, c_void};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, bail};
use libloading::Library;
use realfft::RealFftPlanner;
use serde_json::{Value, json};

use super::listen::Vad;
use super::whisper::Request;

const SR: usize = 16000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_FRAMES: usize = 3000; // окно модели: 30 с
const TIME_PER_FRAME: f64 = HOP as f64 / SR as f64;
const FRAMES_PER_SECOND: usize = SR / HOP;
const TOKENS_PER_SECOND: usize = SR / (HOP * 2);
const TIME_PRECISION: f64 = 0.02;
const MAX_LENGTH: usize = 448;

// TranscriptionOptions faster-whisper по умолчанию
const PATIENCE: f32 = 1.0;
const LENGTH_PENALTY: f32 = 1.0;
const BEST_OF: usize = 5;
const LOG_PROB_THRESHOLD: f64 = -1.0;
const NO_SPEECH_THRESHOLD: f64 = 0.6;
const PROMPT_RESET_ON_TEMPERATURE: f32 = 0.5;
const MAX_INITIAL_TIMESTAMP: f64 = 1.0;
const PREPEND_PUNCTUATIONS: &str = "\"'“¿([{-";
const APPEND_PUNCTUATIONS: &str = "\"'.。,，!！?？:：”)]}、";
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"; // string.punctuation

type P = *mut c_void;

#[repr(C)]
struct GenOpts {
    beam_size: usize,
    patience: f32,
    length_penalty: f32,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    max_length: usize,
    sampling_topk: usize,
    sampling_temperature: f32,
    num_hypotheses: usize,
    suppress_blank: i32,
    suppress_tokens: *const i32,
    n_suppress_tokens: usize,
    max_initial_timestamp_index: usize,
}

type ErrBuf = [c_char; 1024];

struct Api {
    load: unsafe extern "C" fn(*const c_char, i32, i32, *const c_char, usize, usize, *mut c_char, usize) -> P,
    free: unsafe extern "C" fn(P),
    is_multilingual: unsafe extern "C" fn(P) -> i32,
    n_mels: unsafe extern "C" fn(P) -> usize,
    encode: unsafe extern "C" fn(P, *const f32, usize, usize, *mut c_char, usize) -> P,
    free_sv: unsafe extern "C" fn(P),
    generate: unsafe extern "C" fn(P, P, *const usize, usize, *const GenOpts, *mut *mut usize, *mut usize,
                                   *mut f32, *mut f32, *mut c_char, usize) -> i32,
    free_buf: unsafe extern "C" fn(*mut c_void),
    detect_language: unsafe extern "C" fn(P, P, *mut *mut c_char, *mut *mut f32, *mut usize, *mut c_char, usize) -> i32,
    align: unsafe extern "C" fn(P, P, *const usize, usize, *const usize, usize, usize, c_long, *mut *mut i64,
                                *mut usize, *mut *mut f32, *mut usize, *mut c_char, usize) -> i32,
    _lib: Library,
}

fn err_text(e: &ErrBuf) -> String {
    unsafe { CStr::from_ptr(e.as_ptr()) }.to_string_lossy().into_owned()
}

impl Api {
    fn load(dir: &Path) -> anyhow::Result<Api> {
        // CTranslate2 открывает cuBLAS 12 по имени, когда он понадобится: загруженный
        // заранее глобально находится по этому имени, где бы ни лежал.
        #[cfg(unix)]
        for name in ["libcublasLt.so.12", "libcublas.so.12"] {
            use libloading::os::unix::{Library as L, RTLD_GLOBAL, RTLD_NOW};
            let p = dir.join(name);
            if p.exists() {
                if let Ok(lib) = unsafe { L::open(Some(&p), RTLD_NOW | RTLD_GLOBAL) } {
                    std::mem::forget(lib);
                }
            }
        }
        let lib = super::llama::open_lib(dir, "ct2shim")?;
        macro_rules! f {
            ($n:literal) => {
                unsafe { *lib.get(concat!($n, "\0").as_bytes())? }
            };
        }
        Ok(Api {
            load: f!("ct2w_load"),
            free: f!("ct2w_free"),
            is_multilingual: f!("ct2w_is_multilingual"),
            n_mels: f!("ct2w_n_mels"),
            encode: f!("ct2w_encode"),
            free_sv: f!("ct2w_free_sv"),
            generate: f!("ct2w_generate"),
            free_buf: f!("ct2w_free_buf"),
            detect_language: f!("ct2w_detect_language"),
            align: f!("ct2w_align"),
            _lib: lib,
        })
    }
}

/// An encoded window, freed with it.
struct Encoded<'a> {
    ptr: P,
    api: &'a Api,
}

impl Drop for Encoded<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.free_sv)(self.ptr) }
    }
}

struct Generated {
    tokens: Vec<u32>,
    score: f32,
    no_speech_prob: f32,
}

// ------------------------------------------------------------------ токенизатор

/// faster_whisper.tokenizer.Tokenizer.
struct Tok<'a> {
    hf: &'a tokenizers::Tokenizer,
    language: Option<u32>,
    task: Option<u32>,
    language_code: String,
    eot: u32,
    no_timestamps: u32,
    sot: u32,
}

impl Tok<'_> {
    fn id(hf: &tokenizers::Tokenizer, t: &str) -> anyhow::Result<u32> {
        hf.token_to_id(t).with_context(|| format!("no token {t}"))
    }

    fn timestamp_begin(&self) -> u32 {
        self.no_timestamps + 1
    }

    fn sot_sequence(&self) -> Vec<u32> {
        let mut s = vec![self.sot];
        s.extend(self.language);
        s.extend(self.task);
        s
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.hf.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let text: Vec<u32> = tokens.iter().copied().filter(|&t| t < self.eot).collect();
        self.hf.decode(&text, true).unwrap_or_default()
    }

    fn decode_with_timestamps(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        let mut run: Vec<u32> = vec![];
        for &t in tokens {
            if t >= self.timestamp_begin() {
                out.push_str(&self.hf.decode(&run, true).unwrap_or_default());
                run.clear();
                out.push_str(&format!("<|{:.2}|>", (t - self.timestamp_begin()) as f64 * 0.02));
            } else {
                run.push(t);
            }
        }
        out.push_str(&self.hf.decode(&run, true).unwrap_or_default());
        out
    }

    fn non_speech_tokens(&self) -> Vec<u32> {
        let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』".chars().map(String::from).collect();
        symbols.extend("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split(' ').map(String::from));
        let misc = "♩♪♫♬♭♮♯";
        let mut result = std::collections::BTreeSet::new();
        result.insert(self.encode(" -")[0]);
        result.insert(self.encode(" '")[0]);
        for symbol in symbols.iter().cloned().chain(misc.chars().map(String::from)) {
            let is_misc = symbol.chars().count() == 1 && misc.contains(&symbol);
            for tokens in [self.encode(&symbol), self.encode(&format!(" {symbol}"))] {
                if tokens.len() == 1 || (is_misc && !tokens.is_empty()) {
                    result.insert(tokens[0]);
                }
            }
        }
        result.into_iter().collect()
    }

    fn split_to_word_tokens(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        if ["zh", "ja", "th", "lo", "my", "yue"].contains(&self.language_code.as_str()) {
            return self.split_tokens_on_unicode(tokens);
        }
        self.split_tokens_on_spaces(tokens)
    }

    fn split_tokens_on_unicode(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let full: Vec<char> = self.decode_with_timestamps(tokens).chars().collect();
        let (mut words, mut word_tokens, mut current) = (vec![], vec![], vec![]);
        let mut offset = 0;
        for &t in tokens {
            current.push(t);
            let decoded = self.decode_with_timestamps(&current);
            let idx = decoded.chars().position(|c| c == '\u{fffd}').map(|i| i + offset);
            if idx.is_none_or(|i| i < full.len() && full[i] == '\u{fffd}') {
                offset += decoded.chars().count();
                words.push(decoded);
                word_tokens.push(std::mem::take(&mut current));
            }
        }
        (words, word_tokens)
    }

    fn split_tokens_on_spaces(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let (subwords, subword_tokens) = self.split_tokens_on_unicode(tokens);
        let (mut words, mut word_tokens): (Vec<String>, Vec<Vec<u32>>) = (vec![], vec![]);
        for (subword, toks) in subwords.into_iter().zip(subword_tokens) {
            let special = toks[0] >= self.eot;
            let with_space = subword.starts_with(' ');
            let punctuation = PUNCTUATION.contains(subword.trim());
            if special || with_space || punctuation || words.is_empty() {
                words.push(subword);
                word_tokens.push(toks);
            } else {
                words.last_mut().unwrap().push_str(&subword);
                word_tokens.last_mut().unwrap().extend(toks);
            }
        }
        (words, word_tokens)
    }
}

// --------------------------------------------------------------------- признаки

/// FeatureExtractor.get_mel_filters: [n_mels][n_fft/2 + 1].
fn mel_filters(n_mels: usize) -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let val = 1.0 / (N_FFT as f64 * (1.0 / SR as f64));
    let fftfreqs: Vec<f64> = (0..n_bins).map(|i| i as f64 * val).collect();
    let (min_mel, max_mel) = (0.0f64, 45.245640471924965f64);
    let div = (n_mels + 1) as f64;
    let step = (max_mel - min_mel) / div;
    let mut mels: Vec<f64> = (0..n_mels + 2).map(|i| i as f64 * step + min_mel).collect();
    mels[n_mels + 1] = max_mel; // np.linspace ставит конец точно
    let f_sp = 200.0 / 3.0;
    let mut freqs: Vec<f64> = mels.iter().map(|m| 0.0 + f_sp * m).collect();
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    for (f, &m) in freqs.iter_mut().zip(&mels) {
        if m >= min_log_mel {
            *f = min_log_hz * (logstep * (m - min_log_mel)).exp();
        }
    }
    let fdiff: Vec<f64> = freqs.windows(2).map(|w| w[1] - w[0]).collect();
    let mut w = vec![0f32; n_mels * n_bins];
    for i in 0..n_mels {
        let enorm = 2.0 / (freqs[i + 2] - freqs[i]);
        for j in 0..n_bins {
            let lower = -(freqs[i] - fftfreqs[j]) / fdiff[i];
            let upper = (freqs[i + 2] - fftfreqs[j]) / fdiff[i + 1];
            w[i * n_bins + j] = (0f64.max(lower.min(upper)) * enorm) as f32;
        }
    }
    w
}

/// FeatureExtractor.__call__: [n_mels][frames] and the number of frames.
fn log_mel(audio: &[f32], filters: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
    let mut x = audio.to_vec();
    x.extend_from_slice(&[0.0; HOP]); // padding=160
    let n = x.len();
    let pad = N_FFT / 2;
    // np.pad(mode="reflect"), в том числе когда отступ длиннее сигнала
    let at = |i: isize| -> f32 {
        let mut i = i;
        while i < 0 || i >= n as isize {
            if i < 0 {
                i = -i;
            }
            if i >= n as isize {
                i = 2 * (n as isize - 1) - i;
            }
        }
        x[i as usize]
    };
    let padded: Vec<f32> = (-(pad as isize)..(n + pad) as isize).map(at).collect();
    let frames = (padded.len() - N_FFT) / HOP; // 1 + …, и последний кадр отбрасывается
    // np.hanning(401)[:-1]
    let m = (N_FFT + 1) as f64;
    let window: Vec<f32> = (0..N_FFT)
        .map(|k| {
            let nn = (1.0 - m) + 2.0 * k as f64;
            (0.5 + 0.5 * (std::f64::consts::PI * nn / (m - 1.0)).cos()) as f32
        })
        .collect();
    let n_bins = N_FFT / 2 + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let mut mel = vec![0f32; n_mels * frames];
    const BLOCK: usize = 3000;
    let mut mag = vec![0f32; n_bins * BLOCK];
    let mut out = vec![0f32; n_mels * BLOCK];
    for b0 in (0..frames).step_by(BLOCK) {
        let nb = BLOCK.min(frames - b0);
        for t in 0..nb {
            let s = (b0 + t) * HOP;
            for k in 0..N_FFT {
                input[k] = (padded[s + k] * window[k]) as f64; // произведение — во float32, БПФ — во float64
            }
            fft.process(&mut input, &mut spec).expect("fft");
            for (j, c) in spec.iter().enumerate() {
                let a = (c.re as f32).hypot(c.im as f32); // complex64, np.abs
                mag[j * nb + t] = a * a;
            }
        }
        unsafe {
            matrixmultiply::sgemm(n_mels, n_bins, nb, 1.0, filters.as_ptr(), n_bins as isize, 1, mag.as_ptr(),
                                  nb as isize, 1, 0.0, out.as_mut_ptr(), nb as isize, 1);
        }
        for i in 0..n_mels {
            for t in 0..nb {
                mel[i * frames + b0 + t] = out[i * nb + t].max(1e-10).log10();
            }
        }
    }
    let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    (mel, frames)
}

/// features[:, from:from+len], then pad_or_trim to 3000 frames.
fn window_of(mel: &[f32], frames: usize, n_mels: usize, from: usize, len: usize) -> Vec<f32> {
    let mut w = vec![0f32; n_mels * N_FRAMES];
    let len = len.min(N_FRAMES).min(frames.saturating_sub(from));
    for i in 0..n_mels {
        w[i * N_FRAMES..i * N_FRAMES + len].copy_from_slice(&mel[i * frames + from..i * frames + from + len]);
    }
    w
}

// ------------------------------------------------------------ детектор голоса

/// vad.get_speech_timestamps with VadOptions() — [start, end) in samples.
fn speech_timestamps(probs: &[f32], n: usize) -> Vec<(i64, i64)> {
    let threshold = 0.5f32;
    let neg_threshold = (threshold - 0.15).max(0.01);
    let window = 512i64;
    let min_speech_samples = 0.0f64;
    let speech_pad_samples = SR as f64 * 400.0 / 1000.0;
    let max_speech_samples = f64::INFINITY;
    let min_silence_samples = SR as f64 * 2000.0 / 1000.0;
    let min_silence_at_max = SR as f64 * 98.0 / 1000.0;
    let len = n as i64;

    let mut triggered = false;
    let mut speeches: Vec<(i64, i64)> = vec![];
    let mut cur: Option<(i64, i64)> = None; // (start, end); end is set when closed
    let (mut temp_end, mut prev_end, mut next_start) = (0i64, 0i64, 0i64);
    for (i, &p) in probs.iter().enumerate() {
        let pos = window * i as i64;
        if p >= threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = pos;
            }
        }
        if p >= threshold && !triggered {
            triggered = true;
            cur = Some((pos, 0));
            continue;
        }
        if triggered && (pos - cur.map_or(0, |c| c.0)) as f64 > max_speech_samples {
            if prev_end != 0 {
                speeches.push((cur.unwrap().0, prev_end));
                cur = None;
                if next_start < prev_end {
                    triggered = false;
                } else {
                    cur = Some((next_start, 0));
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push((cur.unwrap().0, pos));
                cur = None;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }
        if p < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = pos;
            }
            if (pos - temp_end) as f64 > min_silence_at_max {
                prev_end = temp_end;
            }
            if ((pos - temp_end) as f64) < min_silence_samples {
                continue;
            }
            let start = cur.map_or(0, |c| c.0);
            if (temp_end - start) as f64 > min_speech_samples {
                speeches.push((start, temp_end));
            }
            cur = None;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
            continue;
        }
    }
    if let Some((start, _)) = cur {
        if (len - start) as f64 > min_speech_samples {
            speeches.push((start, len));
        }
    }
    let k = speeches.len();
    for i in 0..k {
        if i == 0 {
            speeches[0].0 = (speeches[0].0 as f64 - speech_pad_samples).max(0.0) as i64;
        }
        if i != k - 1 {
            let silence = speeches[i + 1].0 - speeches[i].1;
            if (silence as f64) < 2.0 * speech_pad_samples {
                speeches[i].1 += silence.div_euclid(2);
                speeches[i + 1].0 = (speeches[i + 1].0 - silence.div_euclid(2)).max(0);
            } else {
                speeches[i].1 = (len as f64).min(speeches[i].1 as f64 + speech_pad_samples) as i64;
                speeches[i + 1].0 = (speeches[i + 1].0 as f64 - speech_pad_samples).max(0.0) as i64;
            }
        } else {
            speeches[i].1 = (len as f64).min(speeches[i].1 as f64 + speech_pad_samples) as i64;
        }
    }
    speeches
}

/// vad.SpeechTimestampsMap: times in the speech-only audio → times in the original.
struct TimestampsMap {
    chunk_end_sample: Vec<i64>,
    total_silence_before: Vec<f64>,
}

impl TimestampsMap {
    fn new(chunks: &[(i64, i64)]) -> Self {
        let (mut prev_end, mut silent) = (0i64, 0i64);
        let (mut ends, mut before) = (vec![], vec![]);
        for &(s, e) in chunks {
            silent += s - prev_end;
            prev_end = e;
            ends.push(e - silent);
            before.push(silent as f64 / SR as f64);
        }
        TimestampsMap { chunk_end_sample: ends, total_silence_before: before }
    }

    fn chunk_index(&self, time: f64, is_end: bool) -> usize {
        let sample = (time * SR as f64) as i64;
        if is_end {
            if let Some(i) = self.chunk_end_sample.iter().position(|&e| e == sample) {
                return i;
            }
        }
        self.chunk_end_sample.partition_point(|&e| e <= sample).min(self.chunk_end_sample.len() - 1)
    }

    fn original_time(&self, time: f64, chunk: Option<usize>, is_end: bool) -> f64 {
        let i = chunk.unwrap_or_else(|| self.chunk_index(time, is_end));
        round_to(self.total_silence_before[i] + time, 2)
    }
}

// ------------------------------------------------------------------- разбор

#[derive(Clone, Debug)]
struct WordT {
    word: String,
    tokens: Vec<u32>,
    start: f64,
    end: f64,
    probability: f64,
}

struct Seg {
    seek: usize,
    start: f64,
    end: f64,
    tokens: Vec<u32>,
    words: Vec<WordT>,
}

/// Python's round(x, n): correctly rounded, ties to even.
fn round_to(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap_or(x)
}

fn get_end(segs: &[Seg]) -> Option<f64> {
    segs.iter().rev().flat_map(|s| s.words.iter().rev()).map(|w| w.end).next().or(segs.last().map(|s| s.end))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn merge_punctuations(alignment: &mut [WordT]) {
    if alignment.is_empty() {
        return;
    }
    let mut i = alignment.len() as isize - 2;
    let mut j = alignment.len() - 1;
    while i >= 0 {
        let iu = i as usize;
        let prev = alignment[iu].clone();
        if prev.word.starts_with(' ') && PREPEND_PUNCTUATIONS.contains(prev.word.trim()) {
            let f = &mut alignment[j];
            f.word = prev.word.clone() + &f.word;
            let mut t = prev.tokens.clone();
            t.extend(&f.tokens);
            f.tokens = t;
            alignment[iu].word.clear();
            alignment[iu].tokens.clear();
        } else {
            j = iu;
        }
        i -= 1;
    }
    let (mut i, mut j) = (0usize, 1usize);
    while j < alignment.len() {
        let following = alignment[j].clone();
        if !alignment[i].word.ends_with(' ') && APPEND_PUNCTUATIONS.contains(following.word.as_str()) {
            let p = &mut alignment[i];
            p.word.push_str(&following.word);
            p.tokens.extend(&following.tokens);
            alignment[j].word.clear();
            alignment[j].tokens.clear();
        } else {
            i = j;
        }
        j += 1;
    }
}

/// faster_whisper.audio.decode_audio hands a file's sound over as s16: the
/// samples faster-whisper sees when given a path, not an array.
pub fn through_s16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = (*v * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) / 32768.0;
    }
}

pub struct FasterWhisper {
    api: Api,
    model: Mutex<usize>, // ctranslate2::models::Whisper*: вызовы по очереди, как под замком в движке Python
    hf: tokenizers::Tokenizer,
    multilingual: bool,
    n_mels: usize,
    filters: Vec<f32>,
    vad: Option<Vad>,
    pub device: String,
    pub compute_type: String,
}

unsafe impl Send for FasterWhisper {}
unsafe impl Sync for FasterWhisper {}

impl Drop for FasterWhisper {
    fn drop(&mut self) {
        unsafe { (self.api.free)(*self.model.lock().unwrap() as P) }
    }
}

impl FasterWhisper {
    /// `dir` — a CTranslate2 Whisper model (model.bin, tokenizer.json), as
    /// faster-whisper downloads it; `vad` — Silero for live speech.
    pub fn load(dir: &Path, vad: Option<&Path>, gpu: bool) -> anyhow::Result<FasterWhisper> {
        let api = Api::load(&super::lib_dir()?)?;
        let device = if gpu { "cuda" } else { "cpu" };
        let compute_type = std::env::var("STT_COMPUTE_TYPE")
            .unwrap_or_else(|_| if gpu { "float16".into() } else { "int8".into() });
        let path = CString::new(dir.to_string_lossy().as_bytes())?;
        let ct = CString::new(compute_type.clone())?;
        let mut e: ErrBuf = [0; 1024];
        let model = unsafe { (api.load)(path.as_ptr(), gpu as i32, 0, ct.as_ptr(), 0, 1, e.as_mut_ptr(), e.len()) };
        if model.is_null() {
            bail!("CTranslate2 cannot load {}: {}", dir.display(), err_text(&e));
        }
        super::trim_heap();
        let hf = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        let n_mels = unsafe { (api.n_mels)(model) };
        let multilingual = unsafe { (api.is_multilingual)(model) } != 0;
        let vad = match vad.filter(|p| p.is_file()) {
            Some(p) => {
                super::init_onnx(&super::lib_dir()?)?;
                Some(Vad::load(p)?)
            }
            None => None,
        };
        Ok(FasterWhisper {
            api,
            model: Mutex::new(model as usize),
            hf,
            multilingual,
            n_mels,
            filters: mel_filters(n_mels),
            vad,
            device: device.into(),
            compute_type,
        })
    }

    fn encode<'a>(&'a self, model: P, window: &[f32]) -> anyhow::Result<Encoded<'a>> {
        let mut e: ErrBuf = [0; 1024];
        let ptr = unsafe { (self.api.encode)(model, window.as_ptr(), self.n_mels, N_FRAMES, e.as_mut_ptr(), e.len()) };
        if ptr.is_null() {
            bail!("encode: {}", err_text(&e));
        }
        Ok(Encoded { ptr, api: &self.api })
    }

    fn generate(&self, model: P, enc: &Encoded, prompt: &[u32], o: &GenOpts) -> anyhow::Result<Generated> {
        let prompt: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        let (mut ids, mut n, mut score, mut nsp) = (std::ptr::null_mut(), 0usize, 0f32, 0f32);
        let mut e: ErrBuf = [0; 1024];
        let rc = unsafe {
            (self.api.generate)(model, enc.ptr, prompt.as_ptr(), prompt.len(), o, &mut ids, &mut n, &mut score, &mut nsp,
                                e.as_mut_ptr(), e.len())
        };
        if rc != 0 {
            bail!("generate: {}", err_text(&e));
        }
        let tokens = unsafe { std::slice::from_raw_parts(ids, n) }.iter().map(|&t| t as u32).collect();
        unsafe { (self.api.free_buf)(ids as *mut c_void) };
        Ok(Generated { tokens, score, no_speech_prob: nsp })
    }

    fn detect_language(&self, model: P, enc: &Encoded) -> anyhow::Result<(String, f32)> {
        let (mut toks, mut probs, mut n) = (std::ptr::null_mut(), std::ptr::null_mut(), 0usize);
        let mut e: ErrBuf = [0; 1024];
        let rc = unsafe { (self.api.detect_language)(model, enc.ptr, &mut toks, &mut probs, &mut n, e.as_mut_ptr(), e.len()) };
        if rc != 0 {
            bail!("detect_language: {}", err_text(&e));
        }
        let names = unsafe { CStr::from_ptr(toks) }.to_string_lossy().into_owned();
        let p = if n > 0 { unsafe { *probs } } else { 0.0 };
        unsafe {
            (self.api.free_buf)(toks as *mut c_void);
            (self.api.free_buf)(probs as *mut c_void);
        }
        let first = names.lines().next().unwrap_or("<|en|>");
        Ok((first.trim_start_matches("<|").trim_end_matches("|>").to_string(), p))
    }

    fn align(&self, model: P, enc: &Encoded, start: &[u32], text: &[u32], num_frames: usize)
             -> anyhow::Result<(Vec<(i64, i64)>, Vec<f32>)> {
        let start: Vec<usize> = start.iter().map(|&t| t as usize).collect();
        let text: Vec<usize> = text.iter().map(|&t| t as usize).collect();
        let (mut pairs, mut np, mut probs, mut nq) = (std::ptr::null_mut(), 0usize, std::ptr::null_mut(), 0usize);
        let mut e: ErrBuf = [0; 1024];
        let rc = unsafe {
            (self.api.align)(model, enc.ptr, start.as_ptr(), start.len(), text.as_ptr(), text.len(), num_frames, 7,
                             &mut pairs, &mut np, &mut probs, &mut nq, e.as_mut_ptr(), e.len())
        };
        if rc != 0 {
            bail!("align: {}", err_text(&e));
        }
        let flat = unsafe { std::slice::from_raw_parts(pairs, np * 2) };
        let out = (flat.chunks(2).map(|c| (c[0], c[1])).collect(), unsafe { std::slice::from_raw_parts(probs, nq) }.to_vec());
        unsafe {
            (self.api.free_buf)(pairs as *mut c_void);
            (self.api.free_buf)(probs as *mut c_void);
        }
        Ok(out)
    }

    /// WhisperModel.find_alignment for one window.
    fn find_alignment(&self, model: P, tok: &Tok, text_tokens: &[u32], enc: &Encoded, num_frames: usize)
                      -> anyhow::Result<Vec<WordT>> {
        let mut with_eot = text_tokens.to_vec();
        with_eot.push(tok.eot);
        let (words, word_tokens) = tok.split_to_word_tokens(&with_eot);
        if word_tokens.len() <= 1 {
            return Ok(vec![]);
        }
        let (pairs, probs) = self.align(model, enc, &tok.sot_sequence(), text_tokens, num_frames)?;
        let mut boundaries = vec![0usize];
        for t in &word_tokens[..word_tokens.len() - 1] {
            boundaries.push(boundaries.last().unwrap() + t.len());
        }
        let mut jump_times = vec![];
        for (k, &(ti, time)) in pairs.iter().enumerate() {
            if k == 0 || ti != pairs[k - 1].0 {
                jump_times.push(time as f64 / TOKENS_PER_SECOND as f64);
            }
        }
        let jt = |i: usize| jump_times.get(i).or(jump_times.last()).copied().unwrap_or(0.0);
        let mut out = vec![];
        for w in 0..boundaries.len() - 1 {
            let (i, j) = (boundaries[w], boundaries[w + 1]);
            let slice = &probs[i.min(probs.len())..j.min(probs.len())];
            let probability = if slice.is_empty() {
                f64::NAN
            } else {
                slice.iter().map(|&p| p as f64).sum::<f64>() / slice.len() as f64
            };
            out.push(WordT { word: words[w].clone(), tokens: word_tokens[w].clone(), start: jt(i), end: jt(j), probability });
        }
        Ok(out)
    }

    /// WhisperModel.add_word_timestamps for one window's segments.
    fn add_word_timestamps(&self, model: P, tok: &Tok, segs: &mut [Seg], enc: &Encoded, num_frames: usize,
                           mut last_speech_timestamp: f64) -> anyhow::Result<()> {
        if segs.is_empty() {
            return Ok(());
        }
        let per_sub: Vec<Vec<u32>> = segs.iter().map(|s| s.tokens.iter().copied().filter(|&t| t < tok.eot).collect()).collect();
        let text_tokens: Vec<u32> = per_sub.concat();
        let mut alignment = self.find_alignment(model, tok, &text_tokens, enc, num_frames)?;
        let mut durations: Vec<f64> = alignment.iter().map(|w| w.end - w.start).filter(|&d| d != 0.0).collect();
        let median_duration = if durations.is_empty() { 0.0 } else { median(&mut durations) }.min(0.7);
        let max_duration = median_duration * 2.0;
        if !durations.is_empty() {
            let marks = ".。!！?？";
            for i in 1..alignment.len() {
                if alignment[i].end - alignment[i].start > max_duration {
                    if marks.contains(alignment[i].word.as_str()) {
                        alignment[i].end = alignment[i].start + max_duration;
                    } else if marks.contains(alignment[i - 1].word.as_str()) {
                        alignment[i].start = alignment[i].end - max_duration;
                    }
                }
            }
        }
        merge_punctuations(&mut alignment);

        let time_offset = segs[0].seek as f64 / FRAMES_PER_SECOND as f64;
        let mut word_index = 0;
        for (si, seg) in segs.iter_mut().enumerate() {
            let mut saved = 0;
            let mut words: Vec<WordT> = vec![];
            while word_index < alignment.len() && saved < per_sub[si].len() {
                let t = &alignment[word_index];
                if !t.word.is_empty() {
                    words.push(WordT {
                        word: t.word.clone(),
                        tokens: vec![],
                        start: round_to(time_offset + t.start, 2),
                        end: round_to(time_offset + t.end, 2),
                        probability: t.probability,
                    });
                }
                saved += t.tokens.len();
                word_index += 1;
            }
            if !words.is_empty() {
                if words[0].end - last_speech_timestamp > median_duration * 4.0
                    && (words[0].end - words[0].start > max_duration
                        || (words.len() > 1 && words[1].end - words[0].start > max_duration * 2.0))
                {
                    if words.len() > 1 && words[1].end - words[1].start > max_duration {
                        let boundary = (words[1].end / 2.0).max(words[1].end - max_duration);
                        words[0].end = boundary;
                        words[1].start = boundary;
                    }
                    words[0].start = 0f64.max(words[0].end - max_duration);
                }
                if seg.start < words[0].end && seg.start - 0.5 > words[0].start {
                    words[0].start = 0f64.max((words[0].end - median_duration).min(seg.start));
                } else {
                    seg.start = words[0].start;
                }
                let last = words.len() - 1;
                if seg.end > words[last].start && seg.end + 0.5 < words[last].end {
                    words[last].end = (words[last].start + median_duration).max(seg.end);
                } else {
                    seg.end = words[last].end;
                }
                last_speech_timestamp = seg.end;
            }
            seg.words = words;
        }
        Ok(())
    }

    /// WhisperModel._split_segments_by_timestamps.
    fn split_by_timestamps(tok: &Tok, tokens: &[u32], time_offset: f64, segment_size: usize, segment_duration: f64,
                           mut seek: usize) -> (Vec<Seg>, usize, bool) {
        let tb = tok.timestamp_begin();
        let mut out = vec![];
        let single_ending = tokens.len() >= 2 && tokens[tokens.len() - 2] < tb && tb <= tokens[tokens.len() - 1];
        let consecutive: Vec<usize> =
            (1..tokens.len()).filter(|&i| tokens[i] >= tb && tokens[i - 1] >= tb).collect();
        if !consecutive.is_empty() {
            let mut slices = consecutive;
            if single_ending {
                slices.push(tokens.len());
            }
            let mut last = 0;
            for &cur in &slices {
                let sliced = &tokens[last..cur];
                let start_pos = sliced[0] as i64 - tb as i64;
                let end_pos = sliced[sliced.len() - 1] as i64 - tb as i64;
                out.push(Seg {
                    seek,
                    start: time_offset + start_pos as f64 * TIME_PRECISION,
                    end: time_offset + end_pos as f64 * TIME_PRECISION,
                    tokens: sliced.to_vec(),
                    words: vec![],
                });
                last = cur;
            }
            if single_ending {
                seek += segment_size;
            } else {
                let last_pos = tokens[last - 1] as i64 - tb as i64;
                seek = (seek as i64 + last_pos * 2) as usize;
            }
        } else {
            let mut duration = segment_duration;
            let ts: Vec<u32> = tokens.iter().copied().filter(|&t| t >= tb).collect();
            if let Some(&l) = ts.last() {
                if l != tb {
                    duration = (l - tb) as f64 * TIME_PRECISION;
                }
            }
            out.push(Seg { seek, start: time_offset, end: time_offset + duration, tokens: tokens.to_vec(), words: vec![] });
            seek += segment_size;
        }
        (out, seek, single_ending)
    }

    /// 16 kHz mono → the transcript in the shape base.Transcript has, as
    /// stt_whisper.WhisperSTT.transcribe returns it. None if `on_segment`
    /// asked to stop.
    pub fn transcribe(&self, audio: &[f32], r: &Request, on_segment: &mut dyn FnMut(f64, f64, &str) -> bool)
                      -> anyhow::Result<Option<Value>> {
        let original = audio.to_vec();
        let duration = original.len() as f64 / SR as f64;

        let (audio, speech) = if r.live {
            let vad = self.vad.as_ref().context("no voice detector for live speech")?;
            let chunks = speech_timestamps(&vad.probs_whole(&original)?, original.len());
            let mut kept = vec![];
            for &(s, e) in &chunks {
                kept.extend_from_slice(&original[s as usize..e as usize]);
            }
            (kept, (!chunks.is_empty()).then(|| TimestampsMap::new(&chunks)))
        } else {
            (original, None)
        };

        let (mel, n_frames) = log_mel(&audio, &self.filters, self.n_mels);
        let guard = self.model.lock().unwrap();
        let model = *guard as P;

        let (language, language_probability) = match r.language.filter(|l| !l.is_empty()) {
            Some(l) if self.multilingual => (l.to_string(), 1.0f32),
            Some(_) | None if !self.multilingual => ("en".to_string(), 1.0),
            _ => {
                let enc = self.encode(model, &window_of(&mel, n_frames, self.n_mels, 0, N_FRAMES))?;
                self.detect_language(model, &enc)?
            }
        };
        let tok = Tok {
            hf: &self.hf,
            language: if self.multilingual {
                Some(Tok::id(&self.hf, &format!("<|{language}|>")).map_err(|_| anyhow::anyhow!("unknown language {language}"))?)
            } else {
                None
            },
            task: if self.multilingual {
                Some(Tok::id(&self.hf, if r.translate { "<|translate|>" } else { "<|transcribe|>" })?)
            } else {
                None
            },
            language_code: language.clone(),
            eot: Tok::id(&self.hf, "<|endoftext|>")?,
            no_timestamps: Tok::id(&self.hf, "<|notimestamps|>")?,
            sot: Tok::id(&self.hf, "<|startoftranscript|>")?,
        };
        let sot_prev = Tok::id(&self.hf, "<|startofprev|>")?;
        let no_speech = self.hf.token_to_id("<|nospeech|>").or_else(|| self.hf.token_to_id("<|nocaptions|>"));
        // get_suppressed_tokens(tokenizer, [-1])
        let mut suppress: Vec<i32> = tok.non_speech_tokens().into_iter().map(|t| t as i32).collect();
        for t in ["<|transcribe|>", "<|translate|>", "<|startoftranscript|>", "<|startofprev|>", "<|startoflm|>"] {
            suppress.extend(self.hf.token_to_id(t).map(|t| t as i32));
        }
        suppress.extend(no_speech.map(|t| t as i32));
        suppress.sort();
        suppress.dedup();

        // Подсказка и термины — как их склеивает движок Python (stt_whisper._prompt_and_hotwords)
        let terms: Option<String> = r.hotwords.map(|h| h.iter().filter(|x| !x.is_empty()).cloned().collect::<Vec<_>>().join(", "))
            .filter(|t| !t.is_empty());
        let prompt = r.prompt.filter(|p| !p.is_empty());
        let (initial_prompt, hotwords) = match (prompt, terms) {
            (Some(p), Some(t)) => (Some(format!("{p} {t}")), None),
            (p, t) => (p.map(String::from), t),
        };
        let beam_size = if r.draft { 2 } else { 5 };
        let condition_on_previous_text = !r.live;
        let temperature = r.temperature;

        let content_frames = n_frames - 1;
        let mut seek = 0usize;
        let mut all_tokens: Vec<u32> = vec![];
        let mut prompt_reset_since = 0;
        if let Some(p) = &initial_prompt {
            all_tokens.extend(tok.encode(&format!(" {}", p.trim())));
        }
        let mut last_speech_timestamp = 0.0;
        let mut segments = vec![];
        let mut texts = vec![];

        while seek < content_frames {
            let time_offset = seek as f64 * TIME_PER_FRAME;
            let segment_size = N_FRAMES.min(content_frames - seek);
            let segment_duration = segment_size as f64 * TIME_PER_FRAME;
            let previous_tokens = &all_tokens[prompt_reset_since..];
            let enc = self.encode(model, &window_of(&mel, n_frames, self.n_mels, seek, segment_size))?;

            // get_prompt
            let mut p: Vec<u32> = vec![];
            if !previous_tokens.is_empty() || hotwords.is_some() {
                p.push(sot_prev);
                if let Some(h) = &hotwords {
                    let mut ht = tok.encode(&format!(" {}", h.trim()));
                    if ht.len() >= MAX_LENGTH / 2 {
                        ht.truncate(MAX_LENGTH / 2 - 1);
                    }
                    p.extend(ht);
                }
                if !previous_tokens.is_empty() {
                    let k = previous_tokens.len().saturating_sub(MAX_LENGTH / 2 - 1);
                    p.extend_from_slice(&previous_tokens[k..]);
                }
            }
            p.extend(tok.sot_sequence());

            // generate_with_fallback с одной температурой: запасных нет, результат — этот
            let sampling = temperature > 0.0;
            let opts = GenOpts {
                beam_size: if sampling { 1 } else { beam_size },
                patience: PATIENCE,
                length_penalty: LENGTH_PENALTY,
                repetition_penalty: 1.0,
                no_repeat_ngram_size: 0,
                max_length: MAX_LENGTH,
                sampling_topk: if sampling { 0 } else { 1 },
                sampling_temperature: if sampling { temperature } else { 1.0 },
                num_hypotheses: if sampling { BEST_OF } else { 1 },
                suppress_blank: 1,
                suppress_tokens: suppress.as_ptr(),
                n_suppress_tokens: suppress.len(),
                max_initial_timestamp_index: (MAX_INITIAL_TIMESTAMP / TIME_PRECISION).round_ties_even() as usize,
            };
            let result = self.generate(model, &enc, &p, &opts)?;
            let seq_len = result.tokens.len() as f64;
            let cum_logprob = result.score as f64 * seq_len.powf(LENGTH_PENALTY as f64);
            let avg_logprob = cum_logprob / (seq_len + 1.0);

            let mut should_skip = result.no_speech_prob as f64 > NO_SPEECH_THRESHOLD;
            if avg_logprob > LOG_PROB_THRESHOLD {
                should_skip = false;
            }
            if should_skip {
                seek += segment_size;
                continue;
            }

            let (mut current, new_seek, single_ending) =
                Self::split_by_timestamps(&tok, &result.tokens, time_offset, segment_size, segment_duration, seek);
            seek = new_seek;
            if r.word_timestamps {
                self.add_word_timestamps(model, &tok, &mut current, &enc, segment_size, last_speech_timestamp)?;
                if !single_ending {
                    if let Some(end) = get_end(&current) {
                        if end > time_offset {
                            seek = (end * FRAMES_PER_SECOND as f64).round_ties_even() as usize;
                        }
                    }
                }
                if let Some(end) = get_end(&current) {
                    last_speech_timestamp = end;
                }
            }
            drop(enc);

            for mut s in current {
                let text = tok.decode(&s.tokens);
                if s.start == s.end || text.trim().is_empty() {
                    continue;
                }
                all_tokens.extend_from_slice(&s.tokens);
                // restore_speech_timestamps
                if let Some(map) = &speech {
                    if !s.words.is_empty() {
                        for w in s.words.iter_mut() {
                            let chunk = map.chunk_index((w.start + w.end) / 2.0, false);
                            w.start = map.original_time(w.start, Some(chunk), false);
                            w.end = map.original_time(w.end, Some(chunk), false);
                        }
                        s.start = s.words[0].start;
                        s.end = s.words[s.words.len() - 1].end;
                    } else {
                        s.start = map.original_time(s.start, None, false);
                        s.end = map.original_time(s.end, None, true);
                    }
                }
                let text = text.trim().to_string();
                let words: Vec<Value> = s.words.iter()
                    .map(|w| json!({"start": round_to(w.start, 3), "end": round_to(w.end, 3), "word": w.word}))
                    .collect();
                segments.push(json!({"start": round_to(s.start, 3), "end": round_to(s.end, 3), "text": text, "words": words}));
                if !on_segment(s.end, duration, &text) {
                    return Ok(None);
                }
                texts.push(text);
            }

            if !condition_on_previous_text || temperature > PROMPT_RESET_ON_TEMPERATURE {
                prompt_reset_since = all_tokens.len();
            }
        }
        drop(guard);
        Ok(Some(json!({
            "text": texts.join(" ").trim(),
            "language": language,
            "language_probability": round_to(language_probability as f64, 3),
            "duration": round_to(duration, 3),
            "segments": segments,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_pads_and_merges_like_faster_whisper() {
        // речь на кадрах 10–20 и 30–40 при тишине между меньше двух секунд — один отрезок
        let mut p = vec![0.0f32; 200];
        for i in (10..20).chain(30..40) {
            p[i] = 0.9;
        }
        let n = 200 * 512;
        assert_eq!(speech_timestamps(&p, n), vec![(0, 40 * 512 + 6400)]);
    }

    #[test]
    fn python_rounding() {
        assert_eq!(round_to(2.675, 2), 2.67); // 2.67499999… в двоичном виде
        assert_eq!(round_to(0.125, 2), 0.12);
        assert_eq!(round_to(1.005, 3), 1.005);
    }
}
