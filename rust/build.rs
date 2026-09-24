//! On Windows the CTranslate2 wrapper is built into the binary.
//!
//! `ct2shim.cpp` is a C face for CTranslate2's C++ API (rust/ct2shim). On Unix
//! it is compiled by `voicy setup`, which needs a compiler on the machine that
//! runs the server; on Windows nobody has one, so it is compiled here, once,
//! by whoever builds the binary.
//!
//! Two things make that possible without having `ctranslate2.dll` at hand:
//! an import library made from a vendored list of the twenty symbols the
//! wrapper calls, and `/DELAYLOAD`, which puts off loading the DLL until the
//! first of those calls. So `voicy.exe` starts, and `voicy setup` downloads
//! CTranslate2, before anything needs it to be there.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=ct2shim/ct2shim.cpp");
    println!("cargo:rerun-if-changed=ct2shim/ctranslate2-msvc-x64.def");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let target = std::env::var("TARGET").expect("TARGET");

    // Импортная библиотека из списка символов: сам DLL для этого не нужен
    let mut lib = cc::windows_registry::find(&target, "lib.exe").expect("lib.exe рядом с cl.exe");
    let status = lib
        .arg("/nologo")
        .arg("/def:ct2shim/ctranslate2-msvc-x64.def")
        .arg("/machine:x64")
        .arg(format!("/out:{}", out.join("ctranslate2.lib").display()))
        .status()
        .expect("lib.exe не запускается");
    assert!(status.success(), "импортная библиотека не собралась");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include("ct2shim/include")
        .define("CT2SHIM_STATIC", None) // внутрь бинарника: наружу выносить нечего
        .file("ct2shim/ct2shim.cpp")
        .compile("ct2shim");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=ctranslate2");
    // отложенная загрузка: DLL понадобится только на первом распознавании
    println!("cargo:rustc-link-lib=dylib=delayimp");
    println!("cargo:rustc-link-arg=/DELAYLOAD:ctranslate2.dll");
}
