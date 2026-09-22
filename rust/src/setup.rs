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

/// What a platform needs, and where it comes from. Only the platform this was
/// measured on is here; the others get their own archives as they are measured.
struct Platform {
    /// Сборка llama.cpp из релиза: она же даёт ggml с бэкендом CUDA.
    llama_assets: &'static [&'static str],
    whisper_asset: &'static str,
    /// Колёса PyPI и то, что из них берётся.
    wheels: &'static [&'static str],
    wanted: &'static [&'static str],
}

const LINUX_X64_CUDA13: Platform = Platform {
    llama_assets: &["llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-{v}-bin-ubuntu-cuda-13.4-x64.tar.gz"],
    whisper_asset: "whisper-bin-ubuntu-x64.tar.gz",
    wheels: &["onnxruntime-gpu=={ort}", "nvidia-cudnn-cu13", "nvidia-curand", "nvidia-cufft", "nvidia-nvjitlink",
              "nvidia-cuda-nvrtc", "ctranslate2=={ct2}", "nvidia-cublas-cu12=={cublas12}"],
    wanted: &["libonnxruntime.so", "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so", "libcudnn",
              "libcurand.so", "libcufft.so", "libnvJitLink.so", "libnvrtc", "libctranslate2", "libgomp",
              "libcublas.so.12", "libcublasLt.so.12"],
};

fn platform() -> anyhow::Result<&'static Platform> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok(&LINUX_X64_CUDA13),
        (os, arch) => bail!(
            "no engine libraries for {os}-{arch} yet: only linux-x86_64 with NVIDIA is prepared and measured.\n\
             The server runs there with the Python engines (voicy serve), and the libraries for this platform\n\
             are the next step — docs/adr/0022."
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
async fn wheel_url(spec: &str) -> anyhow::Result<String> {
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
            let ok = n.ends_with(".whl") && n.contains("x86_64") && n.contains("manylinux")
                && (n.contains("cp312") || n.contains("py3-none"));
            ok.then(|| f["url"].as_str().unwrap_or_default().to_string())
        })
        .next()
        .with_context(|| format!("no linux x86-64 wheel for {spec}"))
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

/// ct2shim: a C face for CTranslate2's C++ API, and the one thing built here.
/// Without it recognition falls back to the Python engine.
async fn build_shim(lib: &Path, tmp: &Path) -> anyhow::Result<()> {
    let cxx = std::env::var("CXX").ok().or_else(|| ["g++", "clang++", "c++"].into_iter().find(|c| which(c)).map(String::from));
    let Some(cxx) = cxx else {
        say("нет компилятора C++ — обёртка над CTranslate2 не собрана; распознавание пойдёт через Python.");
        say("  поставить: sudo apt install build-essential (или xcode-select --install), потом voicy setup снова");
        return Ok(());
    };
    let src_archive = cached(&format!("https://github.com/OpenNMT/CTranslate2/archive/refs/tags/v{CT2}.tar.gz"), tmp).await?;
    let root = untar_all(&src_archive, &tmp.join("ct2-src"))?;
    let cpp = tmp.join("ct2shim.cpp");
    fs::write(&cpp, include_str!("../ct2shim/ct2shim.cpp"))?;
    let ct2_so = fs::read_dir(lib)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libctranslate2")))
        .context("no libctranslate2 in the library directory")?;
    say(format!("собираю обёртку над CTranslate2 ({cxx})"));
    let out = Command::new(&cxx)
        .args(["-std=c++17", "-O2", "-fPIC", "-shared"])
        .arg(&cpp)
        .arg("-I")
        .arg(root.join("include"))
        .arg(&ct2_so)
        // RPATH, а не RUNPATH: он же покрывает зависимости самой libctranslate2
        .args(["-Wl,--disable-new-dtags,-rpath,$ORIGIN", "-o"])
        .arg(lib.join(native::lib_file("ct2shim")))
        .output()
        .with_context(|| format!("cannot run {cxx}"))?;
    if !out.status.success() {
        bail!("обёртка не собралась:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(())
}

fn which(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|d| d.join(cmd).is_file()))
        .unwrap_or(false)
}

// -------------------------------------------------------------------- шаги

async fn libs(tmp: &Path) -> anyhow::Result<()> {
    let p = platform()?;
    let lib = native::lib_dir_path();
    fs::create_dir_all(&lib)?;
    for asset in p.llama_assets {
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/{}", expand(asset));
        let archive = cached(&url, tmp).await?;
        untar(&archive, &lib, |n| n.contains(".so"))?;
        // квантование весов Qwen3-TTS делает эта же сборка (scripts/convert_qwen.py)
        untar(&archive, lib.parent().unwrap_or(&lib), |n| n == "llama-quantize")?;
    }
    // libwhisper — из сборки для процессора: CUDA ей даёт ggml llama.cpp (experiments/21)
    let url = format!("https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_CPP}/{}", p.whisper_asset);
    let archive = cached(&url, tmp).await?;
    untar(&archive, &lib, |n| n.starts_with("libwhisper.so"))?;

    for spec in p.wheels {
        let spec = expand(spec);
        let url = wheel_url(&spec).await?;
        let wheel = cached(&url, tmp).await?;
        unzip(&wheel, &lib, |n| n.contains(".so") && p.wanted.iter().any(|w| n.starts_with(w)))?;
    }
    let so = lib.join("libonnxruntime.so");
    if !so.exists() {
        let mut versioned: Vec<PathBuf> = fs::read_dir(&lib)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("libonnxruntime.so.")))
            .collect();
        versioned.sort();
        if let Some(v) = versioned.last() {
            #[cfg(unix)]
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
        let url = wheel_url(&format!("faster-whisper=={FASTER_WHISPER}")).await.or_else(|_| {
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
