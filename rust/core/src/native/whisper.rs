//! Speech to text without Python: whisper.cpp through its C API, on the same
//! ggml and GPU backends as llama.cpp.
//!
//! Not the default: it is as fast as faster-whisper but less accurate where
//! voicy is measured — 12.8% CER against 10.4% on live Russian speech (Golos),
//! 71–74% of engineering terms against 83–86% (experiments/21). It is here for
//! a server that must run without Python: STT_ENGINE=whisper.cpp.
//!
//! Set up like faster-whisper in voicy: large-v3-turbo, beam of five, non-speech
//! tokens suppressed, the prompt and the terms as faster-whisper passes them,
//! Silero VAD before live windows; plus temperature fallback, without which its
//! beam search sometimes loops on the first word.
//!
//! Pinned to whisper.cpp b5130: the structs mirror its `whisper.h`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::bail;
use libloading::Library;
use serde_json::{Value, json};

use super::llama::{ggml, open_lib};

#[repr(C)]
#[derive(Clone, Copy)]
struct Aheads {
    n_heads: usize,
    heads: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ContextParams {
    use_gpu: bool,
    flash_attn: bool,
    gpu_device: i32,
    dtw_token_timestamps: bool,
    dtw_aheads_preset: i32,
    dtw_n_top: i32,
    dtw_aheads: Aheads,
    dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VadParams {
    threshold: f32,
    min_speech_duration_ms: i32,
    min_silence_duration_ms: i32,
    max_speech_duration_s: f32,
    speech_pad_ms: i32,
    samples_overlap: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FullParams {
    strategy: i32,
    n_threads: i32,
    n_max_text_ctx: i32,
    offset_ms: i32,
    duration_ms: i32,
    translate: bool,
    no_context: bool,
    no_timestamps: bool,
    single_segment: bool,
    print_special: bool,
    print_progress: bool,
    print_realtime: bool,
    print_timestamps: bool,
    token_timestamps: bool,
    thold_pt: f32,
    thold_ptsum: f32,
    max_len: i32,
    split_on_word: bool,
    max_tokens: i32,
    debug_mode: bool,
    audio_ctx: i32,
    tdrz_enable: bool,
    suppress_regex: *const c_char,
    initial_prompt: *const c_char,
    carry_initial_prompt: bool,
    prompt_tokens: *const i32,
    prompt_n_tokens: i32,
    language: *const c_char,
    detect_language: bool,
    suppress_blank: bool,
    suppress_nst: bool,
    temperature: f32,
    max_initial_ts: f32,
    length_penalty: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
    greedy_best_of: i32,
    beam_size: i32,
    beam_patience: f32,
    new_segment_callback: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, i32, *mut c_void)>,
    new_segment_callback_user_data: *mut c_void,
    progress_callback: *mut c_void,
    progress_callback_user_data: *mut c_void,
    encoder_begin_callback: *mut c_void,
    encoder_begin_callback_user_data: *mut c_void,
    abort_callback: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    abort_callback_user_data: *mut c_void,
    logits_filter_callback: *mut c_void,
    logits_filter_callback_user_data: *mut c_void,
    grammar_rules: *const c_void,
    n_grammar_rules: usize,
    i_start_rule: usize,
    grammar_penalty: f32,
    vad: bool,
    vad_model_path: *const c_char,
    vad_params: VadParams,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TokenData {
    id: i32,
    tid: i32,
    p: f32,
    plog: f32,
    pt: f32,
    ptsum: f32,
    t0: i64,
    t1: i64,
    t_dtw: i64,
    vlen: f32,
}

type C = *mut c_void;

struct Api {
    _lib: Library,
    context_default_params: unsafe extern "C" fn() -> ContextParams,
    init: unsafe extern "C" fn(*const c_char, ContextParams) -> C,
    full_default_params: unsafe extern "C" fn(i32) -> FullParams,
    full: unsafe extern "C" fn(C, FullParams, *const f32, i32) -> i32,
    n_segments: unsafe extern "C" fn(C) -> i32,
    seg_t0: unsafe extern "C" fn(C, i32) -> i64,
    seg_t1: unsafe extern "C" fn(C, i32) -> i64,
    seg_text: unsafe extern "C" fn(C, i32) -> *const c_char,
    n_tokens: unsafe extern "C" fn(C, i32) -> i32,
    token_text: unsafe extern "C" fn(C, i32, i32) -> *const c_char,
    token_data: unsafe extern "C" fn(C, i32, i32) -> TokenData,
    token_eot: unsafe extern "C" fn(C) -> i32,
    full_lang_id: unsafe extern "C" fn(C) -> i32,
    lang_str: unsafe extern "C" fn(i32) -> *const c_char,
    lang_max_id: unsafe extern "C" fn() -> i32,
    pcm_to_mel: unsafe extern "C" fn(C, *const f32, i32, i32) -> i32,
    lang_auto_detect: unsafe extern "C" fn(C, i32, i32, *mut f32) -> i32,
    log_set: unsafe extern "C" fn(*mut c_void, *mut c_void),
}

unsafe extern "C" fn quiet(_level: i32, _text: *const c_char, _data: *mut c_void) {}

impl Api {
    fn load(dir: &Path) -> anyhow::Result<Api> {
        ggml(dir)?;
        let lib = open_lib(dir, "whisper")?;
        macro_rules! f {
            ($n:literal) => {
                unsafe { *lib.get(concat!($n, "\0").as_bytes())? }
            };
        }
        Ok(Api {
            context_default_params: f!("whisper_context_default_params"),
            init: f!("whisper_init_from_file_with_params"),
            full_default_params: f!("whisper_full_default_params"),
            full: f!("whisper_full"),
            n_segments: f!("whisper_full_n_segments"),
            seg_t0: f!("whisper_full_get_segment_t0"),
            seg_t1: f!("whisper_full_get_segment_t1"),
            seg_text: f!("whisper_full_get_segment_text"),
            n_tokens: f!("whisper_full_n_tokens"),
            token_text: f!("whisper_full_get_token_text"),
            token_data: f!("whisper_full_get_token_data"),
            token_eot: f!("whisper_token_eot"),
            full_lang_id: f!("whisper_full_lang_id"),
            lang_str: f!("whisper_lang_str"),
            lang_max_id: f!("whisper_lang_max_id"),
            pcm_to_mel: f!("whisper_pcm_to_mel"),
            lang_auto_detect: f!("whisper_lang_auto_detect"),
            log_set: f!("whisper_log_set"),
            _lib: lib,
        })
    }
}

pub struct Request<'a> {
    pub language: Option<&'a str>,
    pub prompt: Option<&'a str>,
    pub hotwords: Option<&'a [String]>,
    pub temperature: f32,
    pub word_timestamps: bool,
    pub translate: bool,
    pub live: bool,
    pub draft: bool,
}

struct Callbacks<'a> {
    api: &'a Api,
    duration: f64,
    on_segment: &'a mut dyn FnMut(f64, f64, &str) -> bool,
    stop: bool,
}

unsafe extern "C" fn on_new_segment(ctx: *mut c_void, _state: *mut c_void, n_new: i32, ud: *mut c_void) {
    let cb = unsafe { &mut *(ud as *mut Callbacks) };
    let n = unsafe { (cb.api.n_segments)(ctx) };
    for i in (n - n_new).max(0)..n {
        let end = unsafe { (cb.api.seg_t1)(ctx, i) } as f64 / 100.0;
        let text = unsafe { CStr::from_ptr((cb.api.seg_text)(ctx, i)) }.to_string_lossy().trim().to_string();
        if !(cb.on_segment)(end.min(cb.duration), cb.duration, &text) {
            cb.stop = true;
        }
    }
}

unsafe extern "C" fn should_abort(ud: *mut c_void) -> bool {
    unsafe { (*(ud as *mut Callbacks)).stop }
}

pub struct Whisper {
    api: Api,
    ctx: Mutex<usize>, // whisper_context*: одно состояние на модель, вызовы по очереди
    vad_model: Option<CString>,
    pub device: String,
}

unsafe impl Send for Whisper {}
unsafe impl Sync for Whisper {}

impl Whisper {
    pub fn load(model: &Path, vad_model: Option<PathBuf>, gpu: bool) -> anyhow::Result<Whisper> {
        let api = Api::load(&super::lib_dir()?)?;
        let path = CString::new(model.to_string_lossy().as_bytes())?;
        let ctx = unsafe {
            if std::env::var_os("VOICY_WHISPER_LOG").is_none() {
                (api.log_set)(quiet as *mut c_void, std::ptr::null_mut());
            }
            let mut p = (api.context_default_params)();
            p.use_gpu = gpu;
            p.flash_attn = true;
            (api.init)(path.as_ptr(), p)
        };
        if ctx.is_null() {
            bail!("whisper.cpp cannot load {}", model.display());
        }
        Ok(Whisper {
            api,
            ctx: Mutex::new(ctx as usize),
            vad_model: vad_model.filter(|p| p.is_file()).and_then(|p| CString::new(p.to_string_lossy().as_bytes()).ok()),
            device: if gpu { "cuda".into() } else { "cpu".into() },
        })
    }

    /// 16 kHz mono → the transcript in the shape base.Transcript has. None if
    /// `on_segment` asked to stop.
    pub fn transcribe(&self, audio: &[f32], r: &Request, on_segment: &mut dyn FnMut(f64, f64, &str) -> bool)
                      -> anyhow::Result<Option<Value>> {
        let api = &self.api;
        let guard = self.ctx.lock().unwrap();
        let ctx = *guard as C;
        let duration = audio.len() as f64 / 16000.0;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8) as i32;

        // Язык: заданный, или определённый по первым 30 секундам — с вероятностью, как у faster-whisper.
        let (lang, lang_prob) = match r.language.filter(|l| !l.is_empty()) {
            Some(l) => (l.to_string(), 1.0),
            None => unsafe {
                (api.pcm_to_mel)(ctx, audio.as_ptr(), audio.len() as i32, threads);
                let mut probs = vec![0f32; (api.lang_max_id)() as usize + 1];
                let id = (api.lang_auto_detect)(ctx, 0, threads, probs.as_mut_ptr());
                if id < 0 {
                    ("en".to_string(), 0.0)
                } else {
                    (CStr::from_ptr((api.lang_str)(id)).to_string_lossy().into_owned(), probs[id as usize])
                }
            },
        };
        let lang_c = CString::new(lang.clone())?;
        // Подсказка и термины — как их передаёт faster-whisper: термины идут в каждое окно,
        // подсказка — в первое; вместе — одной строкой (движок Python склеивает так же).
        let terms = r.hotwords.filter(|h| !h.is_empty()).map(|h| h.join(", "));
        let (prompt, carry) = match (r.prompt.filter(|p| !p.is_empty()), terms) {
            (Some(p), Some(t)) => (Some(format!("{p} {t}")), false),
            (Some(p), None) => (Some(p.to_string()), false),
            (None, Some(t)) => (Some(format!(" {t}")), true),
            (None, None) => (None, false),
        };
        let prompt_c = prompt.map(CString::new).transpose()?;

        let mut cb = Callbacks { api, duration, on_segment, stop: false };
        let mut p = unsafe { (api.full_default_params)(1) }; // WHISPER_SAMPLING_BEAM_SEARCH
        p.n_threads = threads;
        p.translate = r.translate;
        // Каждый запрос — с чистого листа: состояние whisper.cpp живёт между вызовами,
        // и без этого текст прошлого запроса становится подсказкой следующему, а своя
        // подсказка вытесняется. Внутри файла окна по-прежнему видят предыдущий текст.
        p.no_context = true;
        p.print_progress = false;
        p.print_realtime = false;
        p.print_timestamps = false;
        p.print_special = false;
        p.token_timestamps = r.word_timestamps;
        p.language = lang_c.as_ptr();
        p.detect_language = false;
        p.suppress_blank = true;
        p.suppress_nst = true;
        p.temperature = r.temperature;
        // С повтором при срыве: без него лучевой поиск whisper.cpp иногда зацикливается
        // на первом слове («распознавание распознавание…»), чего faster-whisper не делал.
        p.temperature_inc = 0.2;
        p.beam_size = if r.draft { 2 } else { 5 };
        p.greedy_best_of = 5;
        p.initial_prompt = prompt_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        p.carry_initial_prompt = carry;
        p.new_segment_callback = Some(on_new_segment);
        p.new_segment_callback_user_data = &mut cb as *mut Callbacks as *mut c_void;
        p.abort_callback = Some(should_abort);
        p.abort_callback_user_data = &mut cb as *mut Callbacks as *mut c_void;
        if r.live {
            if let Some(vad) = &self.vad_model {
                // VadOptions faster-whisper по умолчанию
                p.vad = true;
                p.vad_model_path = vad.as_ptr();
                p.vad_params = VadParams {
                    threshold: 0.5,
                    min_speech_duration_ms: 0,
                    min_silence_duration_ms: 2000,
                    max_speech_duration_s: f32::MAX,
                    speech_pad_ms: 400,
                    samples_overlap: p.vad_params.samples_overlap,
                };
            }
        }
        let rc = unsafe { (api.full)(ctx, p, audio.as_ptr(), audio.len() as i32) };
        if cb.stop {
            return Ok(None);
        }
        if rc != 0 {
            bail!("whisper.cpp failed with {rc}");
        }
        let lang = if r.language.is_none() {
            unsafe { CStr::from_ptr((api.lang_str)((api.full_lang_id)(ctx))) }.to_string_lossy().into_owned()
        } else {
            lang
        };
        let eot = unsafe { (api.token_eot)(ctx) };
        let mut segments = vec![];
        let mut texts = vec![];
        for i in 0..unsafe { (api.n_segments)(ctx) } {
            let text = unsafe { CStr::from_ptr((api.seg_text)(ctx, i)) }.to_string_lossy().trim().to_string();
            let (t0, t1) = unsafe { ((api.seg_t0)(ctx, i) as f64 / 100.0, (api.seg_t1)(ctx, i) as f64 / 100.0) };
            let mut words: Vec<Value> = vec![];
            if r.word_timestamps {
                // слово — токены от пробела до пробела; время — от первого токена до последнего
                let mut cur: Option<(String, f64, f64)> = None;
                for j in 0..unsafe { (api.n_tokens)(ctx, i) } {
                    let d = unsafe { (api.token_data)(ctx, i, j) };
                    if d.id >= eot {
                        continue;
                    }
                    let piece = unsafe { CStr::from_ptr((api.token_text)(ctx, i, j)) }.to_string_lossy().into_owned();
                    let (s, e) = (d.t0 as f64 / 100.0, d.t1 as f64 / 100.0);
                    match cur.as_mut() {
                        Some(w) if !piece.starts_with(' ') => {
                            w.0.push_str(&piece);
                            w.2 = e;
                        }
                        _ => {
                            if let Some(w) = cur.take() {
                                words.push(json!({"start": round3(w.1), "end": round3(w.2), "word": w.0}));
                            }
                            cur = Some((piece, s, e));
                        }
                    }
                }
                if let Some(w) = cur.take() {
                    words.push(json!({"start": round3(w.1), "end": round3(w.2), "word": w.0}));
                }
            }
            texts.push(text.clone());
            segments.push(json!({"start": round3(t0), "end": round3(t1.min(duration.max(t1))), "text": text, "words": words}));
        }
        drop(guard);
        Ok(Some(json!({
            "text": texts.join(" ").trim(),
            "language": lang,
            "language_probability": round3(lang_prob as f64),
            "duration": round3(duration),
            "segments": segments,
        })))
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Offsets measured by compiling whisper.h of b5130 with gcc (x86-64 Linux).
    #[test]
    fn layout_matches_whisper_h() {
        assert_eq!(size_of::<FullParams>(), 304);
        assert_eq!(size_of::<ContextParams>(), 48);
        assert_eq!(size_of::<TokenData>(), 56);
        assert_eq!(size_of::<VadParams>(), 24);
        for (name, got, want) in [
            ("translate", offset_of!(FullParams, translate), 20),
            ("token_timestamps", offset_of!(FullParams, token_timestamps), 28),
            ("thold_pt", offset_of!(FullParams, thold_pt), 32),
            ("max_len", offset_of!(FullParams, max_len), 40),
            ("max_tokens", offset_of!(FullParams, max_tokens), 48),
            ("audio_ctx", offset_of!(FullParams, audio_ctx), 56),
            ("suppress_regex", offset_of!(FullParams, suppress_regex), 64),
            ("initial_prompt", offset_of!(FullParams, initial_prompt), 72),
            ("carry_initial_prompt", offset_of!(FullParams, carry_initial_prompt), 80),
            ("prompt_tokens", offset_of!(FullParams, prompt_tokens), 88),
            ("language", offset_of!(FullParams, language), 104),
            ("suppress_nst", offset_of!(FullParams, suppress_nst), 114),
            ("temperature", offset_of!(FullParams, temperature), 116),
            ("no_speech_thold", offset_of!(FullParams, no_speech_thold), 140),
            ("greedy", offset_of!(FullParams, greedy_best_of), 144),
            ("beam_search", offset_of!(FullParams, beam_size), 148),
            ("new_segment_callback", offset_of!(FullParams, new_segment_callback), 160),
            ("abort_callback", offset_of!(FullParams, abort_callback), 208),
            ("grammar_rules", offset_of!(FullParams, grammar_rules), 240),
            ("grammar_penalty", offset_of!(FullParams, grammar_penalty), 264),
            ("vad", offset_of!(FullParams, vad), 268),
            ("vad_model_path", offset_of!(FullParams, vad_model_path), 272),
            ("vad_params", offset_of!(FullParams, vad_params), 280),
        ] {
            assert_eq!(got, want, "{name}");
        }
    }
}
