//! Engines that run inside the server, without Python.
//!
//! Their runtimes — llama.cpp and ONNX Runtime — are prebuilt shared libraries
//! for the platform, loaded at start from one directory: VOICY_LIB_DIR, or
//! `~/.cache/voicy/lib/<platform>`. Nothing here is compiled for a GPU; which
//! GPU is used is a matter of which libraries lie in that directory.

pub mod accent;
pub mod decode;
pub mod fwhisper;
pub mod listen;
pub mod llama;
pub mod mel;
pub mod npy;
pub mod qwen;
pub mod whisper;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Context;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

fn platform() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "linux-x64-cuda13",
        ("linux", "aarch64") => "linux-arm64",
        ("windows", "x86_64") => "windows-x64-cuda13",
        ("windows", "aarch64") => "windows-arm64",
        ("macos", _) => "macos-arm64",
        _ => "unknown",
    }
}

pub fn cache_dir() -> PathBuf {
    std::env::var_os("VOICY_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os(if cfg!(windows) { "LOCALAPPDATA" } else { "HOME" })
            .map(|h| if cfg!(windows) { PathBuf::from(h).join("voicy") } else { PathBuf::from(h).join(".cache").join("voicy") }))
        .unwrap_or_else(|| PathBuf::from(".voicy-cache"))
}

/// Weights are read into process memory on their way to the GPU, and glibc
/// keeps what was freed until asked to give it back: 1.5 GB for Whisper alone.
pub fn trim_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        malloc_trim(0);
    }
}

/// Where the libraries for this platform go; `lib_dir` is this, once it exists.
pub fn lib_dir_path() -> PathBuf {
    std::env::var_os("VOICY_LIB_DIR").map(PathBuf::from).unwrap_or_else(|| cache_dir().join("lib").join(platform()))
}

pub fn lib_dir() -> anyhow::Result<PathBuf> {
    let dir = lib_dir_path();
    anyhow::ensure!(dir.is_dir(), "no engine libraries in {} — voicy setup", dir.display());
    search_here(&dir);
    Ok(dir)
}

/// Windows ищет зависимости загружаемой библиотеки где угодно, только не рядом
/// с ней: каталог движков добавляется в список поиска один раз на процесс.
fn search_here(dir: &Path) {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        static DONE: OnceLock<()> = OnceLock::new();
        if DONE.set(()).is_err() {
            return;
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetDllDirectoryW(path: *const u16) -> i32;
            fn AddDllDirectory(path: *const u16) -> *mut std::ffi::c_void;
        }
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        // SetDllDirectory видят обычные LoadLibrary, AddDllDirectory — вызовы с флагами
        // LOAD_LIBRARY_SEARCH_*: так грузят свои зависимости ONNX Runtime и CTranslate2
        unsafe {
            SetDllDirectoryW(wide.as_ptr());
            AddDllDirectory(wide.as_ptr());
        }
        preload_cuda(dir);
    }
    #[cfg(not(windows))]
    let _ = dir;
}

/// CUDA libraries that ONNX Runtime and CTranslate2 load by name, lazily, when
/// the first request needs them — loaded here first, by full path. Windows
/// hands a loaded module to anyone who asks for it by name, whatever the
/// search order they ask with; and a library that does not load says so at
/// start, by name and reason, not as "cudnn64_9.dll with error 2" in the middle
/// of a request. Their own dependencies are looked up beside them
/// (LOAD_WITH_ALTERED_SEARCH_PATH).
#[cfg(windows)]
fn preload_cuda(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(path: *const u16, file: *mut std::ffi::c_void, flags: u32) -> *mut std::ffi::c_void;
    }
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
    const CUDA: [&str; 8] = ["cudart64_", "cublasLt64_", "cublas64_", "cudnn", "cufft64_", "curand64_", "nvJitLink_", "nvrtc"];
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".dll") && CUDA.iter().any(|c| n.starts_with(c)))
        .collect();
    // порядок зависимостей: рантайм, потом cuBLAS Lt, потом всё остальное
    names.sort_by_key(|n| CUDA.iter().position(|c| n.starts_with(c)).unwrap_or(CUDA.len()));
    for name in names {
        let wide: Vec<u16> = dir.join(&name).as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let handle = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        if handle.is_null() {
            let e = std::io::Error::last_os_error();
            let why = match e.raw_os_error() {
                Some(126) => " — не хватает библиотеки, от которой он зависит",
                Some(193) => " — файл повреждён или не для этой системы",
                _ => "",
            };
            eprintln!("voicy: {name} не загружается ({e}){why}; движкам на GPU он понадобится — voicy setup libs поставит библиотеки заново");
        }
        // модуль не выгружается: он нужен до конца работы процесса
    }
}

/// libfoo.so, foo.dll or libfoo.dylib, whichever this platform names it.
pub fn lib_file(stem: &str) -> String {
    llama::lib_name(stem)
}

/// The CUDA provider of ONNX Runtime has no search path of its own: the CUDA
/// libraries it needs are loaded first, globally, so it finds them by name.
#[cfg(unix)]
fn preload(dir: &Path) {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    for name in ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10", "libcufft.so.12",
                 "libnvJitLink.so.13", "libnvrtc.so.13", "libcudnn.so.9"] {
        let p = dir.join(name);
        if p.exists() {
            if let Ok(lib) = unsafe { Library::open(Some(&p), RTLD_NOW | RTLD_GLOBAL) } {
                std::mem::forget(lib); // на всё время работы
            }
        }
    }
}

#[cfg(not(unix))]
fn preload(_dir: &Path) {}

pub fn init_onnx(dir: &Path) -> anyhow::Result<()> {
    static DONE: OnceLock<bool> = OnceLock::new();
    if DONE.get().is_some() {
        return Ok(());
    }
    preload(dir);
    let name = if cfg!(windows) { "onnxruntime.dll" } else if cfg!(target_os = "macos") { "libonnxruntime.dylib" } else { "libonnxruntime.so" };
    ort::init_from(dir.join(name)).context("cannot load ONNX Runtime")?.commit();
    let _ = DONE.set(true);
    Ok(())
}

pub fn session(path: &Path, gpu: bool) -> anyhow::Result<Session> {
    let mut b = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| anyhow::anyhow!("{e}"))?;
    if gpu {
        b = b.with_execution_providers([ort::ep::CUDA::default().build()]).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    b.commit_from_file(path).with_context(|| format!("cannot load {}", path.display()))
}
