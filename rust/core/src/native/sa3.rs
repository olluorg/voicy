//! Stable Audio 3 through sa3.cpp, loaded at run time.
//!
//! Like acestep.cpp it brings its own ggml fork, so it is one library,
//! `voicy-sa3`, with the fork linked in and hidden, behind a C face of six
//! functions (sa3/shim.cpp) over sa3.cpp's own stable ABI. It is opened
//! locally, so its symbols never meet the ggml llama.cpp loads.
//!
//! The model makes instrumental music and sound from an English description:
//! a T5Gemma encoder, a DiT, an autoencoder. It does not sing.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use libloading::Library;

/// Where the model is in its chain, as the progress callback reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Loading,
    Encoding,
    Sampling,
    Decoding,
    Done,
}

pub struct Piece {
    /// Interleaved, L R L R … for stereo.
    pub audio: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}

type Progress = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int) -> c_int;

struct Api {
    open: unsafe extern "C" fn(*const c_char, *const c_char, *mut c_char, c_int) -> *mut c_void,
    #[allow(clippy::type_complexity)]
    generate: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, f64, i64, c_int, f32, c_int,
                                   Option<Progress>, *mut c_void, *mut *mut f32, *mut u64, *mut c_int, *mut c_int,
                                   *mut c_char, c_int) -> c_int,
    free: unsafe extern "C" fn(*mut c_void),
    unload: unsafe extern "C" fn(*mut c_void),
    close: unsafe extern "C" fn(*mut c_void),
    _lib: Library,
}

pub struct Sa3 {
    api: Api,
    ctx: *mut c_void,
    resident: bool,
}

// Вызовы идут по одному — через очередь сервера.
unsafe impl Send for Sa3 {}
unsafe impl Sync for Sa3 {}

#[cfg(unix)]
fn open_local(path: &Path) -> anyhow::Result<Library> {
    use libloading::os::unix::{Library as U, RTLD_LOCAL, RTLD_NOW};
    let lib = unsafe { U::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.with_context(|| format!("cannot load {}", path.display()))?;
    Ok(lib.into())
}

#[cfg(windows)]
fn open_local(path: &Path) -> anyhow::Result<Library> {
    super::llama::open(path)
}

/// The library's file in a library directory.
pub fn lib_path(dir: &Path) -> PathBuf {
    dir.join(super::lib_file("voicy-sa3"))
}

struct Callback<'a> {
    f: &'a mut dyn FnMut(Stage, usize, usize) -> bool,
}

unsafe extern "C" fn trampoline(user: *mut c_void, stage: c_int, step: c_int, total: c_int) -> c_int {
    let cb = unsafe { &mut *(user as *mut Callback) };
    let stage = match stage {
        0 => Stage::Loading,
        1 => Stage::Encoding,
        2 => Stage::Sampling,
        3 => Stage::Decoding,
        _ => Stage::Done,
    };
    let go = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (cb.f)(stage, step.max(0) as usize, total.max(0) as usize)
    }));
    if go.unwrap_or(false) { 0 } else { 1 }
}

impl Sa3 {
    /// `models` holds sa3.cpp's GGUF; `variant` is `medium` or `small-sfx`.
    /// `resident` keeps the weights in VRAM between pieces.
    pub fn load(dir: &Path, models: &Path, variant: &str, resident: bool) -> anyhow::Result<Sa3> {
        let lib = open_local(&lib_path(dir))?;
        let api = unsafe {
            Api {
                open: *lib.get(b"vsa_open\0")?,
                generate: *lib.get(b"vsa_generate\0")?,
                free: *lib.get(b"vsa_free\0")?,
                unload: *lib.get(b"vsa_unload\0")?,
                close: *lib.get(b"vsa_close\0")?,
                _lib: lib,
            }
        };
        let m = CString::new(models.to_string_lossy().as_bytes())?;
        let v = CString::new(variant)?;
        let mut err = vec![0 as c_char; 1024];
        let ctx = unsafe { (api.open)(m.as_ptr(), v.as_ptr(), err.as_mut_ptr(), err.len() as c_int) };
        if ctx.is_null() {
            bail!("sa3: {}", unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy());
        }
        Ok(Sa3 { api, ctx, resident })
    }

    /// One piece of `seconds`. `seed` < 0 — random; `steps`/`cfg` 0 — the
    /// model's defaults. The callback gets the stage and its step of total;
    /// false cancels, and then the result is Ok(None).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&self, prompt: &str, negative: &str, seconds: f64, seed: i64, steps: usize, cfg: f32,
                    mut on_progress: impl FnMut(Stage, usize, usize) -> bool) -> anyhow::Result<Option<Piece>> {
        let p = CString::new(prompt)?;
        let n = CString::new(negative)?;
        let mut cb = Callback { f: &mut on_progress };
        let (mut samples, mut count, mut ch, mut sr) = (std::ptr::null_mut(), 0u64, 0, 0);
        let mut err = vec![0 as c_char; 1024];
        let rc = unsafe {
            (self.api.generate)(self.ctx, p.as_ptr(), n.as_ptr(), seconds, seed, steps as c_int, cfg,
                                self.resident as c_int, Some(trampoline), &mut cb as *mut Callback as *mut c_void,
                                &mut samples, &mut count, &mut ch, &mut sr, err.as_mut_ptr(), err.len() as c_int)
        };
        match rc {
            0 => {}
            1 => return Ok(None),
            _ => bail!("sa3: {}", unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()),
        }
        let (n, chans) = (count as usize, ch.max(1) as usize);
        let planar = unsafe { std::slice::from_raw_parts(samples, n * chans) };
        let mut audio = Vec::with_capacity(n * chans);
        for i in 0..n {
            for c in 0..chans {
                audio.push(planar[c * n + i]);
            }
        }
        unsafe { (self.api.free)(samples as *mut c_void) };
        Ok(Some(Piece { audio, channels: chans as u16, sample_rate: sr as u32 }))
    }

    /// Drop the weights from VRAM; the next piece loads them again.
    pub fn unload(&self) {
        unsafe { (self.api.unload)(self.ctx) };
    }
}

impl Drop for Sa3 {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.ctx) };
    }
}
