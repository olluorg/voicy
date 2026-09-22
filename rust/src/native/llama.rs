//! llama.cpp through its C API, loaded at run time from prebuilt libraries.
//!
//! Pinned to release b11090: the structs below mirror its `llama.h` field for
//! field, and a different release may lay them out differently. The prebuilt
//! archives cover what voicy targets — CUDA, SYCL (Intel Arc), Vulkan, Metal,
//! CPU on x86-64 and ARM64 — so nothing here is compiled for a GPU.

use std::ffi::{CString, c_char, c_void};
use std::path::Path;

use anyhow::{Context as _, bail};
use libloading::Library;

pub type Token = i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Batch {
    pub n_tokens: i32,
    pub token: *mut Token,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChainParams {
    no_perf: bool,
}

type P = *mut c_void;

macro_rules! api {
    ($($name:ident : fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        pub struct Api {
            _libs: Vec<Library>,
            $(pub $name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }
        impl Api {
            unsafe fn bind(llama: &Library, libs: Vec<Library>) -> anyhow::Result<Api> {
                Ok(Api {
                    $($name: unsafe { *llama.get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                        concat!(stringify!($name), "\0").as_bytes()).with_context(|| stringify!($name))? },)*
                    _libs: libs,
                })
            }
        }
    };
}

api! {
    llama_backend_init: fn();
    llama_log_set: fn(P, P);
    llama_model_default_params: fn() -> ModelParams;
    llama_context_default_params: fn() -> ContextParams;
    llama_model_load_from_file: fn(*const c_char, ModelParams) -> P;
    llama_model_free: fn(P);
    llama_model_get_vocab: fn(P) -> P;
    llama_model_n_embd: fn(P) -> i32;
    llama_vocab_n_tokens: fn(P) -> i32;
    llama_init_from_model: fn(P, ContextParams) -> P;
    llama_free: fn(P);
    llama_batch_init: fn(i32, i32, i32) -> Batch;
    llama_batch_free: fn(Batch);
    llama_decode: fn(P, Batch) -> i32;
    llama_get_logits_ith: fn(P, i32) -> *mut f32;
    llama_get_embeddings: fn(P) -> *mut f32;
    llama_get_memory: fn(P) -> P;
    llama_memory_clear: fn(P, bool);
    llama_sampler_chain_default_params: fn() -> ChainParams;
    llama_sampler_chain_init: fn(ChainParams) -> P;
    llama_sampler_chain_add: fn(P, P);
    llama_sampler_init_greedy: fn() -> P;
    llama_sampler_init_dist: fn(u32) -> P;
    llama_sampler_init_top_k: fn(i32) -> P;
    llama_sampler_init_top_p: fn(f32, usize) -> P;
    llama_sampler_init_temp: fn(f32) -> P;
    llama_sampler_init_penalties: fn(i32, i32, f32, f32, f32) -> P;
    llama_sampler_sample: fn(P, P, i32) -> Token;
    llama_sampler_accept: fn(P, Token);
    llama_sampler_free: fn(P);
}

unsafe extern "C" fn quiet(_level: i32, _text: *const c_char, _data: *mut c_void) {}

fn lib_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

#[cfg(unix)]
fn open(path: &Path) -> anyhow::Result<Library> {
    use libloading::os::unix::{Library as U, RTLD_GLOBAL, RTLD_NOW};
    // глобально: ggml-cuda и остальные бэкенды ищут символы ggml среди уже загруженных
    let lib = unsafe { U::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }
        .with_context(|| format!("cannot load {}", path.display()))?;
    Ok(lib.into())
}

#[cfg(windows)]
fn open(path: &Path) -> anyhow::Result<Library> {
    unsafe { Library::new(path) }.with_context(|| format!("cannot load {}", path.display()))
}

impl Api {
    /// Load ggml and llama from `dir` and let ggml find its backends there.
    pub fn load(dir: &Path) -> anyhow::Result<Api> {
        let base = open(&dir.join(lib_name("ggml-base")))?;
        let ggml = open(&dir.join(lib_name("ggml")))?;
        let llama = open(&dir.join(lib_name("llama")))?;
        let dir_c = CString::new(dir.to_string_lossy().as_bytes())?;
        unsafe {
            let load_all: libloading::Symbol<unsafe extern "C" fn(*const c_char)> =
                ggml.get(b"ggml_backend_load_all_from_path\0")?;
            load_all(dir_c.as_ptr());
        }
        let api = unsafe { Api::bind(&llama, vec![])? };
        unsafe {
            if std::env::var_os("VOICY_LLAMA_LOG").is_none() {
                (api.llama_log_set)(quiet as *mut c_void, std::ptr::null_mut());
            }
            (api.llama_backend_init)();
        }
        Ok(Api { _libs: vec![base, ggml, llama], ..api })
    }
}

pub struct Model {
    pub ptr: P,
    pub vocab: P,
    pub n_embd: usize,
    pub n_vocab: usize,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load(api: &Api, path: &Path, gpu: bool) -> anyhow::Result<Model> {
        let c = CString::new(path.to_string_lossy().as_bytes())?;
        unsafe {
            let mut p = (api.llama_model_default_params)();
            p.n_gpu_layers = if gpu { -1 } else { 0 };
            let ptr = (api.llama_model_load_from_file)(c.as_ptr(), p);
            if ptr.is_null() {
                bail!("llama.cpp cannot load {}", path.display());
            }
            let vocab = (api.llama_model_get_vocab)(ptr);
            Ok(Model {
                ptr,
                vocab,
                n_embd: (api.llama_model_n_embd)(ptr) as usize,
                n_vocab: (api.llama_vocab_n_tokens)(vocab) as usize,
            })
        }
    }
}

pub struct Context {
    pub ptr: P,
    pub batch: Batch,
    pub n_vocab: usize,
    pub n_embd: usize,
    batch_cap: usize,
}

unsafe impl Send for Context {}

impl Context {
    /// As the reference implementation sets it up: flash attention on, the KQV
    /// on the GPU, half the cores for generation and all of them for batches.
    pub fn new(api: &Api, model: &Model, n_ctx: u32, embeddings: bool, batch_cap: usize) -> anyhow::Result<Context> {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get()) as i32;
        unsafe {
            let mut p = (api.llama_context_default_params)();
            p.n_ctx = n_ctx;
            p.n_batch = 2048;
            p.n_ubatch = 512;
            p.n_seq_max = 1;
            p.embeddings = embeddings;
            p.pooling_type = 0;
            p.flash_attn_type = 1;
            p.offload_kqv = true;
            p.no_perf = true;
            p.n_threads = (cpus / 2).max(1);
            p.n_threads_batch = cpus;
            let ptr = (api.llama_init_from_model)(model.ptr, p);
            if ptr.is_null() {
                bail!("llama.cpp cannot create a context");
            }
            let batch = (api.llama_batch_init)(batch_cap as i32, model.n_embd as i32, 1);
            Ok(Context { ptr, batch, n_vocab: model.n_vocab, n_embd: model.n_embd, batch_cap })
        }
    }

    pub fn clear(&self, api: &Api) {
        unsafe { (api.llama_memory_clear)((api.llama_get_memory)(self.ptr), true) }
    }

    /// Put `rows` embeddings in the batch. `pos` is either one plane (n values)
    /// or, for the talker's M-RoPE, four planes laid out plane after plane.
    pub fn decode_embd(&mut self, api: &Api, rows: &[f32], pos: &[i32]) -> anyhow::Result<()> {
        let n = rows.len() / self.n_embd;
        anyhow::ensure!(n <= self.batch_cap && pos.len() <= self.batch_cap,
                        "batch too small: {n} rows, {} positions > {}", pos.len(), self.batch_cap);
        unsafe {
            std::ptr::copy_nonoverlapping(rows.as_ptr(), self.batch.embd, rows.len());
            std::ptr::copy_nonoverlapping(pos.as_ptr(), self.batch.pos, pos.len());
            self.batch.n_tokens = n as i32;
            for i in 0..n {
                *self.batch.n_seq_id.add(i) = 1;
                *(*self.batch.seq_id.add(i)) = 0;
                *self.batch.logits.add(i) = (i == n - 1) as i8;
            }
            let rc = (api.llama_decode)(self.ptr, self.batch);
            anyhow::ensure!(rc == 0, "llama_decode failed with {rc}");
        }
        Ok(())
    }

    /// The hidden state of the last position (an embeddings context outputs all).
    pub fn last_embedding(&self, api: &Api, n_rows: usize) -> Vec<f32> {
        unsafe {
            let p = (api.llama_get_embeddings)(self.ptr);
            std::slice::from_raw_parts(p.add((n_rows - 1) * self.n_embd), self.n_embd).to_vec()
        }
    }

    pub fn last_logits(&self, api: &Api) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut((api.llama_get_logits_ith)(self.ptr, -1), self.n_vocab) }
    }
}

pub struct Sampler {
    pub ptr: P,
}

unsafe impl Send for Sampler {}

impl Sampler {
    /// The chain of the reference implementation: penalties, top-k, top-p when
    /// below one, temperature, then a seeded draw; greedy at temperature zero.
    pub fn new(api: &Api, temperature: f32, top_k: i32, top_p: f32, repeat_penalty: f32,
               penalty_last_n: i32, n_vocab: usize, seed: u32) -> Sampler {
        unsafe {
            let ptr = (api.llama_sampler_chain_init)((api.llama_sampler_chain_default_params)());
            if repeat_penalty != 1.0 {
                (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_penalties)(
                    n_vocab as i32, penalty_last_n, repeat_penalty, 0.0, 0.0));
            }
            if temperature > 0.0 {
                if top_k > 0 {
                    (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_top_k)(top_k));
                }
                if top_p < 1.0 {
                    (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_top_p)(top_p, 1));
                }
                (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_temp)(temperature));
                (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_dist)(seed));
            } else {
                (api.llama_sampler_chain_add)(ptr, (api.llama_sampler_init_greedy)());
            }
            Sampler { ptr }
        }
    }

    /// Draw from the last logits of `ctx`, only among [start, end) and `allow`.
    /// llama_sampler_sample accepts the token into the chain's history itself.
    pub fn sample(&self, api: &Api, ctx: &Context, start: usize, end: usize, allow: &[usize]) -> Token {
        let logits = ctx.last_logits(api);
        for (i, l) in logits.iter_mut().enumerate() {
            if (i < start || i >= end) && !allow.contains(&i) {
                *l = -1e10;
            }
        }
        unsafe { (api.llama_sampler_sample)(self.ptr, ctx.ptr, -1) }
    }

    pub fn accept(&self, api: &Api, token: Token) {
        unsafe { (api.llama_sampler_accept)(self.ptr, token) }
    }

    pub fn free(self, api: &Api) {
        unsafe { (api.llama_sampler_free)(self.ptr) }
    }
}
