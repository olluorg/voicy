//! ACE-Step 1.5 music through acestep.cpp, loaded at run time.
//!
//! acestep.cpp brings its own ggml fork — the VAE needs two ops upstream ggml
//! lacks — so it is not loaded beside llama.cpp's ggml as llama is. It is one
//! library, `voicy-acestep`, with the fork linked in and hidden, and a C face
//! of five functions (acestep/shim.cpp). It is opened locally: its symbols
//! never meet the ggml that llama.cpp and whisper.cpp share.
//!
//! One request runs the model's whole chain: a 5 Hz LM plans the song —
//! metadata, codes, and lyrics when there are none — a DiT renders latents, a
//! VAE makes 48 kHz stereo.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, bail};
use libloading::Library;

pub const SAMPLE_RATE: u32 = 48_000;

/// The four GGUF files of one ACE-Step model.
#[derive(Clone, Debug)]
pub struct Files {
    pub lm: PathBuf,
    pub text_encoder: PathBuf,
    pub dit: PathBuf,
    pub vae: PathBuf,
}

/// Where the model is in its chain, as the progress callback reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The LM plans the song; `count` is tokens.
    Plan,
    /// The DiT renders; `count` is steps.
    Render,
    /// The VAE decodes; `count` is tiles.
    Decode,
}

pub struct Song {
    /// Interleaved stereo, L R L R …
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    /// The request as the LM filled it in: bpm, key, lyrics it wrote.
    pub plan: serde_json::Value,
}

type Progress = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;

struct Api {
    open: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char, c_int, c_int, c_int,
                               *mut c_char, c_int) -> *mut c_void,
    generate: unsafe extern "C" fn(*mut c_void, *const c_char, Option<Progress>, *mut c_void, *mut *mut f32,
                                   *mut c_int, *mut c_int, *mut *mut c_char, *mut c_char, c_int) -> c_int,
    free: unsafe extern "C" fn(*mut c_void),
    vram: unsafe extern "C" fn(*const c_void) -> usize,
    close: unsafe extern "C" fn(*mut c_void),
    _lib: Library,
}

pub struct AceStep {
    api: Api,
    ctx: *mut c_void,
}

// Контекст acestep.cpp защищён своим мьютексом (ModelStore); вызовы сюда
// и так идут по одному — через очередь сервера.
unsafe impl Send for AceStep {}
unsafe impl Sync for AceStep {}

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
    dir.join(super::lib_file("voicy-acestep"))
}

/// Latent frames the VAE decodes at once (ACESTEP_VAE_CHUNK); 0, the default,
/// lets the library pick it by free VRAM before each decode: the VAE is the
/// peak of the chain, and 1024 frames ran out beside speech (shim.cpp).
fn vae_chunk() -> usize {
    std::env::var("ACESTEP_VAE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

struct Callback<'a> {
    f: &'a mut dyn FnMut(Stage, usize) -> bool,
}

unsafe extern "C" fn trampoline(user: *mut c_void, stage: c_int, count: c_int) -> c_int {
    let cb = unsafe { &mut *(user as *mut Callback) };
    let stage = match stage {
        0 => Stage::Plan,
        1 => Stage::Render,
        _ => Stage::Decode,
    };
    // паника не должна пересечь границу C
    let go = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (cb.f)(stage, count.max(0) as usize)));
    if go.unwrap_or(false) { 0 } else { 1 }
}

impl AceStep {
    /// Load the model. `keep_loaded` keeps every part in VRAM between songs;
    /// otherwise one part is resident at a time and each song reloads them —
    /// slower, but it fits beside speech on a small card.
    pub fn load(dir: &Path, files: &Files, keep_loaded: bool) -> anyhow::Result<AceStep> {
        let lib = open_local(&lib_path(dir))?;
        let api = unsafe {
            Api {
                open: *lib.get(b"vac_open\0")?,
                generate: *lib.get(b"vac_generate\0")?,
                free: *lib.get(b"vac_free\0")?,
                vram: *lib.get(b"vac_vram\0")?,
                close: *lib.get(b"vac_close\0")?,
                _lib: lib,
            }
        };
        let c = |p: &Path| CString::new(p.to_string_lossy().as_bytes()).context("path");
        let (lm, te, dit, vae) = (c(&files.lm)?, c(&files.text_encoder)?, c(&files.dit)?, c(&files.vae)?);
        let mut err = vec![0 as c_char; 512];
        let ctx = unsafe {
            (api.open)(lm.as_ptr(), te.as_ptr(), dit.as_ptr(), vae.as_ptr(), keep_loaded as c_int, 0,
                       vae_chunk() as c_int, err.as_mut_ptr(), err.len() as c_int)
        };
        if ctx.is_null() {
            bail!("acestep: {}", unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy());
        }
        Ok(AceStep { api, ctx })
    }

    /// One song. `request` is acestep.cpp's request JSON: caption, lyrics
    /// ("[Instrumental]" for none), duration, vocal_language, seed. The
    /// callback is called between tokens, steps and tiles; false cancels,
    /// and then the result is Ok(None).
    pub fn generate(&self, request: &serde_json::Value, mut on_progress: impl FnMut(Stage, usize) -> bool)
                    -> anyhow::Result<Option<Song>> {
        let req = CString::new(request.to_string())?;
        let mut cb = Callback { f: &mut on_progress };
        let (mut samples, mut n, mut sr, mut meta) = (std::ptr::null_mut(), 0, 0, std::ptr::null_mut());
        let mut err = vec![0 as c_char; 512];
        let rc = unsafe {
            (self.api.generate)(self.ctx, req.as_ptr(), Some(trampoline), &mut cb as *mut Callback as *mut c_void,
                                &mut samples, &mut n, &mut sr, &mut meta, err.as_mut_ptr(), err.len() as c_int)
        };
        match rc {
            0 => {}
            1 => return Ok(None),
            _ => bail!("acestep: {}", unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()),
        }
        // плоско по каналам [L0..Ln, R0..Rn] → вперемешку
        let n = n as usize;
        let planar = unsafe { std::slice::from_raw_parts(samples, 2 * n) };
        let mut audio = Vec::with_capacity(2 * n);
        for i in 0..n {
            audio.push(planar[i]);
            audio.push(planar[n + i]);
        }
        let plan = unsafe { CStr::from_ptr(meta) }.to_string_lossy();
        let plan = serde_json::from_str(&plan).unwrap_or(serde_json::Value::Null);
        unsafe {
            (self.api.free)(samples as *mut c_void);
            (self.api.free)(meta as *mut c_void);
        }
        Ok(Some(Song { audio, sample_rate: sr as u32, plan }))
    }

    /// VRAM the model holds right now, bytes.
    pub fn vram(&self) -> usize {
        unsafe { (self.api.vram)(self.ctx) }
    }
}

impl Drop for AceStep {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.ctx) };
    }
}
