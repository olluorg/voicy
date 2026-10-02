//! Engines that run inside the server, without Python.
//!
//! Their runtimes — llama.cpp and ONNX Runtime — are prebuilt shared libraries
//! for the platform, loaded at start from one directory: VOICY_LIB_DIR, or
//! `~/.cache/voicy/lib/<platform>-<backend>`. Nothing here is compiled for a
//! GPU; which GPU is used is a matter of which libraries lie in that directory
//! (`Backend`).
//! Both directories can also be set from code (`set_dirs`).

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

/// What the engines compute on: one set of libraries per kind of GPU, and the
/// CPU. Each is the fastest path its hardware has, not a common denominator
/// (docs/adr/0022): CUDA stays CUDA where there is NVIDIA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA: every engine on the GPU.
    Cuda,
    /// Any GPU with a Vulkan driver — AMD, Intel, NVIDIA: llama.cpp and
    /// whisper.cpp on the GPU, CTranslate2 and ONNX Runtime on the CPU.
    Vulkan,
    /// AMD through HIP; the system ROCm runtime is needed. llama.cpp only.
    Rocm,
    /// Intel through oneAPI; the system oneAPI runtime is needed. llama.cpp only.
    Sycl,
    Cpu,
}

impl Backend {
    pub const ALL: [Backend; 5] = [Backend::Cuda, Backend::Vulkan, Backend::Rocm, Backend::Sycl, Backend::Cpu];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
            Backend::Sycl => "sycl",
            Backend::Cpu => "cpu",
        }
    }

    pub fn from_name(s: &str) -> Option<Backend> {
        Backend::ALL.into_iter().find(|b| b.name() == s.trim().to_ascii_lowercase())
    }

    /// The part of the library directory's name: `cuda13` for CUDA, as it was
    /// before there were others, so an installed set stays where it is.
    fn dir_suffix(self) -> &'static str {
        if self == Backend::Cuda { "cuda13" } else { self.name() }
    }

    /// Which ggml backend library a directory of this set has.
    fn ggml_lib(self) -> Option<&'static str> {
        match self {
            Backend::Cuda => Some("ggml-cuda"),
            Backend::Vulkan => Some("ggml-vulkan"),
            Backend::Rocm => Some("ggml-hip"),
            Backend::Sycl => Some("ggml-sycl"),
            Backend::Cpu => None,
        }
    }
}

fn os_arch() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "linux-x64",
        ("linux", "aarch64") => "linux-arm64",
        ("windows", "x86_64") => "windows-x64",
        ("windows", "aarch64") => "windows-arm64",
        ("macos", _) => "macos-arm64",
        _ => "unknown",
    }
}

/// `~/.cache/voicy/lib/<os>-<arch>-<backend>`.
fn backend_dir(b: Backend) -> PathBuf {
    cache_dir().join("lib").join(format!("{}-{}", os_arch(), b.dir_suffix()))
}

/// The backend of this process, chosen once: `VOICY_DEVICE` when it names
/// one; else the set in VOICY_LIB_DIR, by what lies there; else the best set
/// installed, CUDA first; else, with nothing installed, what the hardware
/// asks for — that is what setup then fetches.
pub fn backend() -> Backend {
    static B: OnceLock<Backend> = OnceLock::new();
    *B.get_or_init(|| {
        if let Ok(v) = std::env::var("VOICY_DEVICE") {
            match Backend::from_name(&v) {
                Some(b) => return b,
                None if v.is_empty() || v == "auto" => {}
                None => log::warn!("VOICY_DEVICE={v} — такого не знаю; есть auto, {}",
                                   Backend::ALL.map(Backend::name).join(", ")),
            }
        }
        // каталог задан, но ещё пуст (образ docker до setup) — решает железо
        if let Some(dir) = LIB_DIR.get().cloned().or_else(|| std::env::var_os("VOICY_LIB_DIR").map(PathBuf::from)) {
            return backend_of(&dir).unwrap_or_else(detect);
        }
        if let Some(b) = [Backend::Cuda, Backend::Rocm, Backend::Sycl, Backend::Vulkan, Backend::Cpu]
            .into_iter()
            .find(|&b| backend_dir(b).is_dir())
        {
            return b;
        }
        detect()
    })
}

/// A library directory's backend, by its ggml backend library; None while
/// there is no ggml in it at all.
fn backend_of(dir: &Path) -> Option<Backend> {
    if !dir.join(lib_file("ggml")).exists() {
        return None;
    }
    Some(Backend::ALL
        .into_iter()
        .find(|b| b.ggml_lib().is_some_and(|l| dir.join(lib_file(l)).exists()))
        .unwrap_or(Backend::Cpu))
}

/// What the hardware here asks for, by its drivers: NVIDIA's is CUDA; any
/// other GPU with a Vulkan driver, Vulkan; none, the CPU. ROCm and SYCL need
/// their runtimes installed and are not guessed — VOICY_DEVICE names them.
pub fn detect() -> Backend {
    let loads = |names: &[&str]| names.iter().any(|&n| unsafe { libloading::Library::new(n) }.is_ok());
    if loads(&["libcuda.so.1", "nvcuda.dll"]) {
        Backend::Cuda
    } else if loads(&["libvulkan.so.1", "vulkan-1.dll"]) {
        Backend::Vulkan
    } else {
        Backend::Cpu
    }
}

static CACHE: OnceLock<PathBuf> = OnceLock::new();
static LIB_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Where the libraries and the models are, for the whole process: the runtimes
/// are loaded once and stay. Set before the first engine loads; a directory
/// already set is not changed, and false says so.
pub fn set_dirs(cache: Option<PathBuf>, lib: Option<PathBuf>) -> bool {
    let cache_ok = cache.map_or(true, |c| CACHE.get() == Some(&c) || CACHE.set(c).is_ok());
    let lib_ok = lib.map_or(true, |l| LIB_DIR.get() == Some(&l) || LIB_DIR.set(l).is_ok());
    cache_ok && lib_ok
}

pub fn cache_dir() -> PathBuf {
    if let Some(c) = CACHE.get() {
        return c.clone();
    }
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
    if let Some(l) = LIB_DIR.get() {
        return l.clone();
    }
    std::env::var_os("VOICY_LIB_DIR").map(PathBuf::from).unwrap_or_else(|| backend_dir(backend()))
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
            log::warn!("{name} не загружается ({e}){why}; движкам на GPU он понадобится — voicy setup libs поставит библиотеки заново");
        }
        // модуль не выгружается: он нужен до конца работы процесса
    }
}

/// How many CUDA devices this process sees, by the CUDA runtime among the
/// engine libraries: 0 in a container the GPU was not passed into, or with
/// CUDA_VISIBLE_DEVICES empty. None where there is no CUDA runtime at all —
/// then nothing is known, and a GPU of another kind may be there.
pub fn cuda_devices() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| {
        let dir = lib_dir().ok()?;
        let path = std::fs::read_dir(&dir).ok()?.flatten().map(|e| e.path()).find(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n.starts_with("libcudart.so") || (n.starts_with("cudart64_") && n.ends_with(".dll"))
            })
        })?;
        unsafe {
            let lib = libloading::Library::new(&path).ok()?;
            let count = lib.get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cudaGetDeviceCount\0").ok()?;
            let mut n = 0i32;
            // ошибка — это и есть «устройств нет»: драйвера нет или GPU не проброшен
            let n = if count(&mut n) == 0 { n.max(0) as usize } else { 0 };
            std::mem::forget(lib); // тот же рантайм понадобится движкам
            Some(n)
        }
    })
}

/// The GPU if it is asked for and there is one: without a device the engines
/// load on the CPU instead of failing — the same image then runs with the GPU
/// passed into the container or without it. This is for the engines on ggml
/// (llama.cpp, whisper.cpp), which run on any backend's GPU.
pub fn use_gpu(wanted: bool) -> bool {
    let b = backend();
    let none = match b {
        Backend::Cpu => return false,
        _ if !wanted => return false,
        Backend::Cuda => cuda_devices() == Some(0),
        // устройства видит сам ggml: драйвер Vulkan есть, а карты под ним может не быть
        _ => lib_dir().ok().and_then(|d| llama::ggml(&d).ok()) == Some(0),
    };
    if none {
        static SAID: OnceLock<()> = OnceLock::new();
        if SAID.set(()).is_ok() {
            log::warn!("видеокарты для {} не видно — движки работают на процессоре, это заметно медленнее",
                       b.name());
        }
    }
    !none
}

/// The same for CTranslate2 and ONNX Runtime: they have a GPU only through CUDA.
pub fn use_cuda(wanted: bool) -> bool {
    backend() == Backend::Cuda && use_gpu(wanted)
}

/// What an engine on ggml reports it runs on: the backend, or the CPU.
pub fn device_name(gpu: bool) -> String {
    if gpu { backend().name().into() } else { "cpu".into() }
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

/// A model on ONNX Runtime; `gpu` puts it on the GPU where ONNX Runtime has
/// one: CUDA, or DirectML on Windows with any other backend. On Linux without
/// CUDA it stays on the CPU.
pub fn session(path: &Path, gpu: bool) -> anyhow::Result<Session> {
    let mut b = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| anyhow::anyhow!("{e}"))?;
    if gpu && backend() == Backend::Cuda {
        b = b.with_execution_providers([ort::ep::CUDA::default().build()]).map_err(|e| anyhow::anyhow!("{e}"))?;
    } else if gpu && cfg!(windows) {
        // DirectML не умеет ни шаблонов памяти, ни параллельного исполнения — так велит его документация
        b = b
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([ort::ep::DirectML::default().build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    b.commit_from_file(path).with_context(|| format!("cannot load {}", path.display()))
}
