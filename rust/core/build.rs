//! On Windows the CTranslate2 wrapper is built here, into the crate.
//!
//! `ct2shim.cpp` is a C face for CTranslate2's C++ API (ct2shim/). On Unix it
//! is compiled by `voicy setup`, which needs a compiler on the machine that
//! runs the engines; on Windows nobody has one, so it is compiled here, once,
//! by whoever builds the program.
//!
//! It is built as a DLL of its own, not into the program: the program then
//! does not import CTranslate2 at all, starts without it, and needs nothing
//! from whoever links it. The DLL's bytes go into the crate (native/fwhisper.rs)
//! and are put beside the other engine libraries the first time Whisper
//! loads, under a name with their hash: a DLL that another process holds open
//! is never overwritten, and a program built from other sources gets its own.
//!
//! No `ctranslate2.dll` is needed to build it: an import library is made from
//! a vendored list of the twenty symbols the wrapper calls.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=ct2shim/ct2shim.cpp");
    println!("cargo:rerun-if-changed=ct2shim/ctranslate2-msvc-x64.def");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let target = std::env::var("TARGET").expect("TARGET");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let shim = manifest.join("ct2shim");

    // Импортная библиотека из списка символов: сам DLL для этого не нужен
    let mut lib = cc::windows_registry::find(&target, "lib.exe").expect("lib.exe рядом с cl.exe");
    let status = lib
        .arg("/nologo")
        .arg(format!("/def:{}", shim.join("ctranslate2-msvc-x64.def").display()))
        .arg("/machine:x64")
        .arg(format!("/out:{}", out.join("ctranslate2.lib").display()))
        .status()
        .expect("lib.exe не запускается");
    assert!(status.success(), "импортная библиотека не собралась");

    let dll = out.join("ct2shim.dll");
    let mut cl = cc::windows_registry::find(&target, "cl.exe").expect("cl.exe из Visual Studio");
    let status = cl
        .current_dir(&out)
        .args(["/nologo", "/LD", "/EHsc", "/std:c++17", "/O2", "/MD"])
        .arg(format!("/I{}", shim.join("include").display()))
        .arg(shim.join("ct2shim.cpp"))
        .arg(format!("/Fe:{}", dll.display()))
        .arg("/link")
        .arg(out.join("ctranslate2.lib"))
        .status()
        .expect("cl.exe не запускается");
    assert!(status.success(), "ct2shim.dll не собрался");

    // имя по содержимому: другая сборка — другой файл рядом, а не поверх
    let bytes = std::fs::read(&dll).expect("ct2shim.dll");
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    println!("cargo:rustc-env=CT2SHIM_DLL_NAME=ct2shim-{:08x}.dll", hash as u32);
    println!("cargo:rustc-env=CT2SHIM_DLL_PATH={}", dll.display());
}
