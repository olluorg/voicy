//! Engines that run inside the server, without Python.
//!
//! Their runtimes — llama.cpp and ONNX Runtime — are prebuilt shared libraries
//! for the platform, loaded at start from one directory: VOICY_LIB_DIR, or
//! `~/.cache/voicy/lib/<platform>`. Nothing here is compiled for a GPU; which
//! GPU is used is a matter of which libraries lie in that directory.

pub mod llama;
pub mod mel;
pub mod npy;
pub mod qwen;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Context;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

fn platform() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "linux-x64-cuda13",
        ("linux", "aarch64") => "linux-arm64",
        ("windows", "x86_64") => "windows-x64",
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

pub fn lib_dir() -> anyhow::Result<PathBuf> {
    let dir = std::env::var_os("VOICY_LIB_DIR").map(PathBuf::from).unwrap_or_else(|| cache_dir().join("lib").join(platform()));
    anyhow::ensure!(dir.is_dir(), "no engine libraries in {}", dir.display());
    Ok(dir)
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
