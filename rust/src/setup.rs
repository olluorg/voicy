//! `voicy setup` — everything the in-process engines need, fetched by the binary
//! itself.
//!
//! Prebuilt llama.cpp and whisper.cpp from their GitHub releases, CTranslate2
//! and ONNX Runtime out of the wheels their authors publish on PyPI, the models
//! from Hugging Face — into `~/.cache/voicy` (`VOICY_CACHE`). Versions are
//! pinned: the bindings in native/ are written against these and no others.
//!
//! Nothing is built here except `ct2shim`, a page of C++ over CTranslate2's C++
//! API, and only when a compiler is at hand. The one thing this cannot do is
//! Qwen3-TTS comes already converted; making those files from the official
//! weights is the one step that needs PyTorch (`scripts/convert_qwen.py`).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use futures::StreamExt;

use crate::native;

const LLAMA: &str = "b11090";
const WHISPER_CPP: &str = "b5130";
const CT2: &str = "4.8.2";
const ORT: &str = "1.30.0";
const CUBLAS12: &str = "12.8.4.1";
const WHISPER_MODEL: &str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo";
/// Qwen3-TTS в GGUF и ONNX: те же официальные веса, переведённые
/// scripts/convert_qwen.py. Свой репозиторий — VOICY_TTS_GGUF_REPO.
const QWEN_MODEL: &str = "sknyazev/qwen3-tts-12hz-1.7b-base-gguf";
const TURN_MODEL: &str = "pipecat-ai/smart-turn-v3";
/// Silero — из колеса faster-whisper: тот же файл, что слышит движок Python.
const FASTER_WHISPER: &str = "1.2.1";

/// What a platform needs, and where it comes from. A platform is here once its
/// archives are known; being here is not the same as being measured — that is
/// what rust/README.md says of each.
struct Platform {
    /// Сборки llama.cpp и whisper.cpp из их релизов: ggml с бэкендом CUDA — оттуда же.
    llama_assets: &'static [&'static str],
    whisper_asset: &'static str,
    /// Колёса PyPI: чем помечены под эту платформу и что из них берётся.
    wheels: &'static [&'static str],
    /// Все эти куски должны быть в имени колеса: одного «manylinux» мало —
    /// под ним лежат и x86-64, и aarch64, и первым в списке PyPI бывает не тот.
    wheel_tag: &'static [&'static str],
    wanted: &'static [&'static str],
}

const WHEELS: &[&str] = &["onnxruntime-gpu=={ort}", "nvidia-cudnn-cu13", "nvidia-curand", "nvidia-cufft",
                          "nvidia-nvjitlink", "nvidia-cuda-nvrtc", "ctranslate2=={ct2}",
                          "nvidia-cublas-cu12=={cublas12}"];

const LINUX_X64_CUDA13: Platform = Platform {
    llama_assets: &["llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz"],
    whisper_asset: "whisper-bin-ubuntu-x64.tar.gz",
    wheels: WHEELS,
    wheel_tag: &["manylinux", "x86_64"],
    wanted: &["libonnxruntime.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so", "libcudnn",
              "libcurand.so", "libcufft.so", "libnvJitLink.so", "libnvrtc", "libctranslate2", "libgomp",
              "libcublas.so.12", "libcublasLt.so.12"],
};

const WINDOWS_X64_CUDA13: Platform = Platform {
    llama_assets: &["llama-{v}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"],
    whisper_asset: "whisper-bin-x64.zip",
    wheels: WHEELS,
    wheel_tag: &["win_amd64"],
    // cudnn64_9.dll есть и у CTranslate2 (под CUDA 12), и в колесе cuDNN под CUDA 13;
    // имя одно, и в процессе останется тот, кто загрузился первым — cuDNN берём у CUDA 13.
    wanted: &["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll", "cudnn",
              "curand64_", "cufft64_", "nvJitLink_", "nvrtc", "ctranslate2.dll", "libiomp5md.dll", "cublas64_12.dll",
              "cublasLt64_12.dll"],
};

fn platform() -> anyhow::Result<&'static Platform> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok(&LINUX_X64_CUDA13),
        ("windows", "x86_64") => Ok(&WINDOWS_X64_CUDA13),
        (os, arch) => bail!(
            "no engine libraries for {os}-{arch} yet: the archives for this platform are the next step\n\
             (docs/adr/0022). Where there is a repository with server/, the Python engines still run: voicy serve."
        ),
    }
}

fn say(msg: impl AsRef<str>) {
    eprintln!("setup: {}", msg.as_ref());
}

fn expand(t: &str) -> String {
    t.replace("{v}", LLAMA).replace("{ort}", ORT).replace("{ct2}", CT2).replace("{cublas12}", CUBLAS12)
}

// ------------------------------------------------------------------ скачивание

async fn fetch(url: &str, to: &Path) -> anyhow::Result<()> {
    let name = url.rsplit('/').next().unwrap_or(url);
    let r = reqwest::Client::builder()
        .user_agent("voicy-setup")
        .build()?
        .get(url)
        .send()
        .await
        .with_context(|| format!("cannot reach {url}"))?
        .error_for_status()
        .with_context(|| format!("cannot download {url}"))?;
    let total = r.content_length().unwrap_or(0);
    let tmp = to.with_extension("part");
    let mut file = tokio::fs::File::create(&tmp).await?;
    let mut done = 0u64;
    let mut shown = 0u64;
    let mut stream = r.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        done += chunk.len() as u64;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        // раз в 50 МБ: качаются гигабайты, и молчание выглядит как зависание
        if done - shown > 50 << 20 {
            shown = done;
            match total {
                0 => say(format!("{name}: {} МБ", done >> 20)),
                t => say(format!("{name}: {} из {} МБ", done >> 20, t >> 20)),
            }
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    drop(file);
    // оборванная закачка — обычное дело; пусть она падает здесь, а не при распаковке
    if total != 0 && done != total {
        fs::remove_file(&tmp)?;
        bail!("{name}: скачано {done} из {total} байт");
    }
    fs::rename(&tmp, to)?;
    Ok(())
}

async fn cached(url: &str, dir: &Path) -> anyhow::Result<PathBuf> {
    let name = url.rsplit('/').next().unwrap_or("file");
    let path = dir.join(name);
    if !path.exists() {
        say(format!("скачиваю {name}"));
        fetch(url, &path).await?;
    }
    Ok(path)
}

/// The manylinux x86-64 wheel of `name[==version]`, as PyPI's JSON index gives it.
async fn wheel_url(spec: &str, tag: &[&str]) -> anyhow::Result<String> {
    let (name, version) = spec.split_once("==").map_or((spec, ""), |(n, v)| (n, v));
    let url = if version.is_empty() {
        format!("https://pypi.org/pypi/{name}/json")
    } else {
        format!("https://pypi.org/pypi/{name}/{version}/json")
    };
    let v: serde_json::Value = reqwest::get(&url).await?.error_for_status()?.json().await?;
    let files = v["urls"].as_array().context("pypi: no files")?;
    files
        .iter()
        .filter_map(|f| {
            let n = f["filename"].as_str()?;
            let ok = n.ends_with(".whl") && tag.iter().all(|t| n.contains(t)) && (n.contains("cp312") || n.contains("py3-none"));
            ok.then(|| f["url"].as_str().unwrap_or_default().to_string())
        })
        .next()
        .with_context(|| format!("no {} wheel for {spec}", tag.join("+")))
}

// ------------------------------------------------------------------ распаковка

/// Flattens out of a .tar.gz the files whose name `keep` accepts.
fn untar(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    let file = fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    let mut n = 0;
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let Some(name) = path.file_name().and_then(|s| s.to_str()).map(String::from) else { continue };
        if !keep(&name) {
            continue;
        }
        let out = dest.join(&name);
        let _ = fs::remove_file(&out);
        // libggml.so и прочие — ссылки на версию рядом; загрузчик ищет их по короткому имени
        if entry.header().entry_type().is_symlink() {
            let link = entry.link_name()?.context("symlink without target")?;
            let target = link.file_name().context("odd symlink")?.to_owned();
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &out)?;
            #[cfg(not(unix))]
            fs::copy(dest.join(&target), &out)?;
            n += 1;
            continue;
        }
        entry.unpack(&out)?;
        n += 1;
    }
    Ok(n)
}

/// The same for a wheel, which is a zip.
fn unzip(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    let mut zip = zip::ZipArchive::new(fs::File::open(archive)?)?;
    let mut n = 0;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        let Some(name) = f.enclosed_name().and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        else {
            continue;
        };
        if !f.is_file() || !keep(&name) {
            continue;
        }
        let mut bytes = vec![];
        f.read_to_end(&mut bytes)?;
        fs::write(dest.join(&name), bytes)?;
        n += 1;
    }
    Ok(n)
}

/// Unpacks the whole tree of a source tarball; only the headers are used.
fn untar_all(archive: &Path, dest: &Path) -> anyhow::Result<PathBuf> {
    let file = fs::File::open(archive)?;
    tar::Archive::new(flate2::read::GzDecoder::new(file)).unpack(dest)?;
    let root = fs::read_dir(dest)?.next().context("empty archive")??.path();
    Ok(root)
}

// ------------------------------------------------------------------- обёртка

/// ct2shim: a C face for CTranslate2's C++ API, and the one thing built here —
/// on Unix. On Windows it is compiled into the binary instead (build.rs), since
/// a compiler is not something to ask of whoever just runs the server.
/// Without it recognition falls back to whisper.cpp or to the Python engine.
async fn build_shim(lib: &Path, tmp: &Path) -> anyhow::Result<()> {
    if cfg!(all(windows, target_env = "msvc")) {
        return Ok(()); // там она вкомпилирована в бинарник (build.rs)
    }
    let src_archive = cached(&format!("https://github.com/OpenNMT/CTranslate2/archive/refs/tags/v{CT2}.tar.gz"), tmp).await?;
    let root = untar_all(&src_archive, &tmp.join("ct2-src"))?;
    let cpp = tmp.join("ct2shim.cpp");
    fs::write(&cpp, include_str!("../ct2shim/ct2shim.cpp"))?;
    let ct2 = fs::read_dir(lib)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libctranslate2") || n == "ctranslate2.dll"))
        .context("no CTranslate2 library in the library directory")?;
    let out = lib.join(native::lib_file("ct2shim"));
    if cfg!(windows) { msvc_shim(&cpp, &root.join("include"), &ct2, &out, tmp) } else { unix_shim(&cpp, &root.join("include"), &ct2, &out) }
}

fn unix_shim(cpp: &Path, include: &Path, ct2: &Path, out: &Path) -> anyhow::Result<()> {
    let Some(cxx) = std::env::var("CXX").ok().or_else(|| ["g++", "clang++", "c++"].into_iter().find(|c| which(c)).map(String::from))
    else {
        return no_compiler("sudo apt install build-essential (или xcode-select --install)");
    };
    say(format!("собираю обёртку над CTranslate2 ({cxx})"));
    let r = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-shared"])
        .arg(cpp)
        .arg("-I")
        .arg(include)
        .arg(ct2)
        // RPATH, а не RUNPATH: он же покрывает зависимости самой libctranslate2
        .args(["-Wl,--disable-new-dtags,-rpath,$ORIGIN", "-o"])
        .arg(out)
        .output()
        .with_context(|| format!("cannot run {cxx}"))?;
    finish(r)
}

/// The wheels ship DLLs without import libraries, and MSVC links against those.
/// One is made from the DLL's own export table; the `LIBRARY` line matters,
/// since otherwise the name comes from the .def file and whoever links against
/// it looks for a DLL that does not exist.
fn import_lib(dll: &Path, stem: &str) -> String {
    let name = dll.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    format!(
        "dumpbin /exports \"{dll}\" > {stem}-exports.txt || exit /b 1\r\n\
         powershell -NoProfile -Command \"$n = Select-String -Path {stem}-exports.txt -Pattern          '^\\s+\\d+\\s+[0-9A-F]+\\s+[0-9A-F]{{8}}\\s+(\\S+)' -AllMatches | ForEach-Object          {{ $_.Matches[0].Groups[1].Value }}; @('LIBRARY {name}', 'EXPORTS') + $n |          Set-Content -Encoding ascii {stem}.def\" || exit /b 1\r\n\
         lib /nologo /def:{stem}.def /machine:x64 /out:{stem}.lib >nul || exit /b 1\r\n",
        dll = dll.display())
}

/// On Windows CTranslate2 exports C++ names mangled the Microsoft way, so the
/// wrapper is built by MSVC — and linked against an import library made from
/// the DLL's own export table, since the wheel ships no .lib. The `LIBRARY`
/// line matters: without it the import library takes its name from the .def
/// file, and the wrapper then looks for a DLL that does not exist.
fn msvc_shim(cpp: &Path, include: &Path, ct2: &Path, out: &Path, tmp: &Path) -> anyhow::Result<()> {
    let Some(vcvars) = vcvars() else {
        return no_compiler("поставить «Build Tools for Visual Studio» с компонентом C++                             (winget install Microsoft.VisualStudio.2022.BuildTools --override                             \"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\")");
    };
    say("собираю обёртку над CTranslate2 (MSVC)");
    let script = tmp.join("build_shim.bat");
    // Всё одной командой cmd: переменные MSVC живут только внутри своего процесса
    fs::write(&script, format!(
        "@echo off\r\n\
         call \"{vcvars}\" >nul || exit /b 1\r\n\
         cd /d \"{tmp}\" || exit /b 1\r\n\
         {imports}\
         cl /nologo /LD /EHsc /std:c++17 /O2 /I \"{include}\" \"{cpp}\" /Fe:\"{out}\" /link ctranslate2.lib || exit /b 1\r\n",
        vcvars = vcvars.display(), tmp = tmp.display(), imports = import_lib(ct2, "ctranslate2"),
        include = include.display(), cpp = cpp.display(), out = out.display()))?;
    let r = Command::new("cmd").args(["/c"]).arg(&script).output().context("cannot run cmd")?;
    finish(r)
}

/// Two OpenMP runtimes in one process is what Intel's refuses to live with:
/// `libiomp5md.dll` (CTranslate2 brings it) aborts the program when it finds
/// `libomp.dll` (ggml brings it) already up, and a copy of one under the other's
/// name is still a second instance. They are the same runtime by origin and each
/// side uses only its standard entry points, so `libomp.dll` is replaced by a
/// stub that forwards them to `libiomp5md.dll`: one runtime, two names.
fn openmp_bridge(lib: &Path, tmp: &Path) -> anyhow::Result<()> {
    let (intel, llvm) = (lib.join("libiomp5md.dll"), lib.join("libomp.dll"));
    if !(intel.is_file() && llvm.is_file()) {
        return Ok(());
    }
    let Some(vcvars) = vcvars() else {
        // без компилятора остаётся ключ, который понимают оба рантайма (main::allow_two_openmp):
        // по замерам скорость та же, но Intel называет это неподдерживаемым (experiments/24)
        say("нет компилятора C++ — OpenMP останется в двух экземплярах, их разводит ключ KMP_DUPLICATE_LIB_OK");
        return Ok(());
    };
    let script = tmp.join("build_omp.bat");
    // Переадресацию линковщик делает только для символов, которые видит: отсюда
    // импортная библиотека Intel-рантайма в строке сборки.
    fs::write(&script, format!(
        "@echo off\r\n\
         call \"{vcvars}\" >nul || exit /b 1\r\n\
         cd /d \"{tmp}\" || exit /b 1\r\n\
         {imports}\
         del /q omp-imports.txt 2>nul\r\n\
         for %%F in (\"{lib}\\*.dll\") do dumpbin /imports:libomp.dll \"%%F\" >> omp-imports.txt\r\n\
         powershell -NoProfile -Command \"$n = Select-String -Path omp-imports.txt -Pattern          '\\b(?:omp_|__kmpc_|kmp_)\\w+' -AllMatches | ForEach-Object {{ $_.Matches }} | ForEach-Object          {{ $_.Value }} | Sort-Object -Unique; @('LIBRARY libomp.dll', 'EXPORTS') + ($n | ForEach-Object          {{ \\\"  $_=libiomp5md.$_\\\" }}) | Set-Content -Encoding ascii omp.def\" || exit /b 1\r\n\
         echo // forwarder > omp_stub.cpp\r\n\
         cl /nologo /LD omp_stub.cpp /link libiomp5md.lib /DEF:omp.def /OUT:\"{llvm}\" || exit /b 1\r\n",
        vcvars = vcvars.display(), tmp = tmp.display(), imports = import_lib(&intel, "libiomp5md"),
        lib = lib.display(), llvm = llvm.display()))?;
    let r = Command::new("cmd").args(["/c"]).arg(&script).output().context("cannot run cmd")?;
    finish(r)?;
    let n = fs::read_to_string(tmp.join("omp.def")).map(|d| d.lines().count().saturating_sub(2)).unwrap_or(0);
    say(format!("OpenMP один на процесс: libomp.dll переадресует {n} символов в libiomp5md.dll"));
    Ok(())
}

/// vcvars64.bat of the newest Visual Studio, as vswhere reports it.
fn vcvars() -> Option<PathBuf> {
    let pf = std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| "C:\\Program Files (x86)".into());
    let vswhere = PathBuf::from(&pf).join("Microsoft Visual Studio").join("Installer").join("vswhere.exe");
    let out = Command::new(&vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "-property", "installationPath"])
        .output()
        .ok()?;
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let bat = PathBuf::from(root).join("VC").join("Auxiliary").join("Build").join("vcvars64.bat");
    bat.is_file().then_some(bat)
}

fn no_compiler(how: &str) -> anyhow::Result<()> {
    say("нет компилятора C++ — обёртка над CTranslate2 не собрана; распознавание пойдёт мимо неё.");
    say(format!("  {how}, потом voicy setup снова"));
    Ok(())
}

fn finish(r: std::process::Output) -> anyhow::Result<()> {
    if !r.status.success() {
        bail!("обёртка не собралась:\n{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
    }
    Ok(())
}

fn which(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|d| d.join(cmd).is_file()))
        .unwrap_or(false)
}

// -------------------------------------------------------------------- шаги

/// Unpacks by what the archive is: the same releases come as .tar.gz and .zip.
fn extract(archive: &Path, dest: &Path, keep: impl Fn(&str) -> bool) -> anyhow::Result<usize> {
    if archive.extension().is_some_and(|e| e == "zip") {
        unzip(archive, dest, keep)
    } else {
        untar(archive, dest, keep)
    }
}

/// The extension shared libraries have here, as the archives name them.
fn dyn_ext() -> &'static str {
    if cfg!(windows) { ".dll" } else if cfg!(target_os = "macos") { ".dylib" } else { ".so" }
}

async fn libs(tmp: &Path) -> anyhow::Result<()> {
    let p = platform()?;
    let lib = native::lib_dir_path();
    fs::create_dir_all(&lib)?;
    for asset in p.llama_assets {
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/{}", expand(asset));
        let archive = cached(&url, tmp).await?;
        extract(&archive, &lib, |n| n.contains(dyn_ext()))?;
        // квантование весов Qwen3-TTS делает эта же сборка (scripts/convert_qwen.py)
        extract(&archive, lib.parent().unwrap_or(&lib), |n| n == "llama-quantize" || n == "llama-quantize.exe")?;
    }
    // whisper.cpp — из сборки для процессора: CUDA ей даёт ggml llama.cpp (experiments/21)
    let url = format!("https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_CPP}/{}", p.whisper_asset);
    let archive = cached(&url, tmp).await?;
    let whisper = native::lib_file("whisper");
    extract(&archive, &lib, |n| n.starts_with(&whisper))?;

    for spec in p.wheels {
        let spec = expand(spec);
        let url = wheel_url(&spec, p.wheel_tag).await?;
        let wheel = cached(&url, tmp).await?;
        unzip(&wheel, &lib, |n| n.contains(dyn_ext()) && p.wanted.iter().any(|w| n.starts_with(w)))?;
    }
    if cfg!(windows) {
        openmp_bridge(&lib, tmp)?;
    }
    // В колесе лежит libonnxruntime.so.1.30.0, а ищется короткое имя
    #[cfg(unix)]
    if !lib.join(native::lib_file("onnxruntime")).exists() {
        let so = lib.join(native::lib_file("onnxruntime"));
        let mut versioned: Vec<PathBuf> = fs::read_dir(&lib)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libonnxruntime.so.")))
            .collect();
        versioned.sort();
        if let Some(v) = versioned.last() {
            std::os::unix::fs::symlink(v.file_name().unwrap_or_default(), &so)?;
        }
    }
    build_shim(&lib, tmp).await?;
    say(format!("библиотеки — {} ({} файлов)", lib.display(), fs::read_dir(&lib)?.count()));
    Ok(())
}

async fn hf(repo: &str, file: &str, to: &Path) -> anyhow::Result<()> {
    if to.join(file).exists() {
        return Ok(());
    }
    fs::create_dir_all(to.join(file).parent().unwrap_or(to))?;
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true");
    say(format!("скачиваю {repo}/{file}"));
    fetch(&url, &to.join(file)).await
}

async fn models(tmp: &Path) -> anyhow::Result<()> {
    let models = native::cache_dir().join("models");
    let whisper = models.join("whisper").join("faster-whisper-large-v3-turbo");
    for f in ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"] {
        hf(WHISPER_MODEL, f, &whisper).await?;
    }
    hf(TURN_MODEL, "smart-turn-v3.2-cpu.onnx", &models.join("turn")).await?;

    let vad = models.join("vad");
    if !vad.join("silero_vad_v6.onnx").exists() {
        fs::create_dir_all(&vad)?;
        let url = wheel_url(&format!("faster-whisper=={FASTER_WHISPER}"), &["py3-none"]).await.or_else(|_| {
            Ok::<String, anyhow::Error>(format!(
                "https://files.pythonhosted.org/packages/py3/f/faster-whisper/faster_whisper-{FASTER_WHISPER}-py3-none-any.whl"
            ))
        })?;
        let wheel = cached(&url, tmp).await?;
        unzip(&wheel, &vad, |n| n == "silero_vad_v6.onnx")?;
    }

    // Qwen3-TTS: говорящая часть в том варианте, который попросили (q5_k по умолчанию)
    let repo = std::env::var("VOICY_TTS_GGUF_REPO").unwrap_or_else(|_| QWEN_MODEL.into());
    let variant = std::env::var("TTS_GGUF_TALKER").unwrap_or_else(|_| "q5_k".into());
    let gguf = std::env::var_os("VOICY_TTS_GGUF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| models.join("qwen3-tts-12hz-1.7b-base-gguf"));
    let mut files = vec![
        "tokenizer.json".to_string(),
        format!("qwen3_tts_talker.{variant}.gguf"),
        "qwen3_tts_predictor.q8_0.gguf".into(),
        "qwen3_tts_decoder.fp16.onnx".into(),
        "qwen3_tts_codec_encoder.fp32.onnx".into(),
        "qwen3_tts_codec_encoder.fp32.onnx.data".into(),
        "qwen3_tts_speaker_encoder.fp32.onnx".into(),
        "qwen3_tts_speaker_encoder.fp32.onnx.data".into(),
        "embeddings/proj_bias.npy".into(),
        "embeddings/proj_weight.npy".into(),
        "embeddings/text_embedding_projected.npy".into(),
    ];
    files.extend((0..16).map(|i| format!("embeddings/codec_embedding_{i}.npy")));
    fs::create_dir_all(gguf.join("embeddings"))?;
    for f in files {
        hf(&repo, &f, &gguf).await?;
    }
    say(format!("модели — {}", models.display()));
    Ok(())
}

pub async fn run(what: &str) -> anyhow::Result<()> {
    let tmp = native::cache_dir().join("downloads");
    fs::create_dir_all(&tmp)?;
    if what != "models" {
        libs(&tmp).await?;
    }
    if what != "libs" {
        models(&tmp).await?;
    }
    say(format!("скачанные архивы можно удалить: {}", tmp.display()));
    Ok(())
}
