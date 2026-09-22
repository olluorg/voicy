//! Qwen3-TTS without Python: the talker and the code predictor run in llama.cpp
//! from GGUF, the codec's encoder and decoder and the speaker encoder in ONNX
//! Runtime. A port of the pipeline from HaujetZhao/Qwen3-TTS-GGUF (MIT), which
//! converts the official weights into those files.
//!
//! One audio frame (1920 samples at 24 kHz, 80 ms) is one step: the talker
//! predicts the first of sixteen codebooks, the predictor the other fifteen,
//! and the sum of their sixteen embeddings goes back into the talker together
//! with the next text token. The decoder turns frames into sound twelve at a
//! time, carrying its state across; it starts from the state it reaches after
//! the voice's own reference, so the clone continues the reference's sound.
//!
//! Measured against the PyTorch engine on the same phrases (experiments/20):
//! the same intelligibility and voice similarity, twice the speed, less than
//! half the video memory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, bail};
use half::f16;
use ort::session::Session;
use ort::value::Tensor;

use super::llama::{Api, Context as Ctx, Model, Sampler};
use super::mel;
use super::npy::Table;

const PAD: usize = 2148;
const BOS: usize = 2149;
const EOS: usize = 2150;
const TTS_BOS: usize = 151672;
const TTS_EOS: usize = 151673;
const TTS_PAD: usize = 151671;
const THINK: usize = 2154;
const THINK_BOS: usize = 2156;
const THINK_EOS: usize = 2157;
const NOTHINK: usize = 2155;
const Q: usize = 16;
pub const SAMPLE_RATE: u32 = 24000;
pub const SAMPLES_PER_FRAME: usize = 1920;
const DECODE_CHUNK: usize = 12;
const LAYERS: usize = 8;
const TALKER_CTX: u32 = 4096;
const MAX_PIECE: usize = 1500; // символов за один проход: ~120 с звука, с запасом до TALKER_CTX

/// Whole sentences, at most `max` characters a piece (a longer sentence alone).
fn split_long(text: &str, max: usize) -> Vec<String> {
    if text.chars().count() <= max {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = vec![];
    let mut cur = String::new();
    let mut sentence = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        sentence.push(c);
        let end = matches!(c, '.' | '!' | '?' | '…' | '\n') && chars.peek().is_none_or(|n| n.is_whitespace());
        if end || chars.peek().is_none() {
            if !cur.is_empty() && cur.chars().count() + sentence.chars().count() > max {
                out.push(std::mem::take(&mut cur).trim().to_string());
            }
            cur.push_str(&sentence);
            sentence.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub const LANGUAGES: [(&str, &str, usize); 10] = [
    ("ru", "russian", 2069), ("en", "english", 2050), ("de", "german", 2053),
    ("fr", "french", 2061), ("es", "spanish", 2054), ("it", "italian", 2070),
    ("pt", "portuguese", 2071), ("ja", "japanese", 2058), ("ko", "korean", 2064),
    ("zh", "chinese", 2055),
];

pub fn language_id(code: &str) -> Option<usize> {
    let c = code.trim().to_lowercase();
    LANGUAGES.iter().find(|(iso, name, _)| *iso == c || *name == c).map(|l| l.2)
}

/// What the files for one model are called inside its directory.
pub struct Files {
    pub dir: PathBuf,
    pub talker: String,
    pub predictor: String,
}

struct Assets {
    text: Table,           // [151936, 2048] float16
    codec: Vec<Table>,     // 16 × [n, 2048] float32
    codec_small: Vec<Vec<f32>>, // те же таблицы, уже спроецированные в 1024 — для малой модели
    proj_w: Vec<f32>,      // [1024, 2048]
    proj_b: Vec<f32>,      // [1024]
    hidden: usize,
    small: usize,
}

impl Assets {
    fn load(dir: &Path) -> anyhow::Result<Assets> {
        let e = dir.join("embeddings");
        let text = Table::open(&e.join("text_embedding_projected.npy"))?;
        let codec = (0..Q).map(|i| Table::open(&e.join(format!("codec_embedding_{i}.npy")))).collect::<Result<Vec<_>, _>>()?;
        let w = Table::open(&e.join("proj_weight.npy"))?;
        let b = Table::open(&e.join("proj_bias.npy"))?;
        let (proj_w, proj_b) = (w.all(), b.all());
        let (hidden, small) = (text.cols, w.rows);
        // заранее, как делает эталон: иначе шестнадцать проекций на каждом кадре
        let codec_small = codec
            .iter()
            .map(|t| {
                let x = t.all();
                let mut y = vec![0f32; t.rows * small];
                sgemm_t(&x, t.rows, hidden, &proj_w, small, &mut y);
                for row in y.chunks_exact_mut(small) {
                    add(row, &proj_b);
                }
                y
            })
            .collect();
        Ok(Assets { hidden, small, proj_w, proj_b, text, codec, codec_small })
    }

    fn codec_small(&self, q: usize, code: usize) -> &[f32] {
        &self.codec_small[q][code * self.small..(code + 1) * self.small]
    }

    fn text(&self, id: usize) -> Vec<f32> {
        self.text.row(id)
    }

    fn codec(&self, q: usize, code: usize) -> Vec<f32> {
        self.codec[q].row(code)
    }

    /// 2048 → 1024 for the predictor: x @ W.T + b.
    fn project(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.small];
        sgemm_t(x, 1, self.hidden, &self.proj_w, self.small, &mut y);
        add(&mut y, &self.proj_b);
        y
    }
}

/// y[m×n] = x[m×k] · w[n×k]ᵀ
fn sgemm_t(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, y: &mut [f32]) {
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, x.as_ptr(), k as isize, 1, w.as_ptr(), 1, k as isize,
                              0.0, y.as_mut_ptr(), n as isize, 1);
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

// ------------------------------------------------------------------ декодер

#[derive(Clone)]
struct DecoderState {
    pre_conv: (Vec<i64>, Vec<f16>),
    latent: (Vec<i64>, Vec<f16>),
    conv: (Vec<i64>, Vec<f16>),
    kv: Vec<(Vec<i64>, Vec<f16>)>, // key_0, value_0, key_1, …
    skip: usize,
}

impl DecoderState {
    fn empty() -> Self {
        DecoderState {
            pre_conv: (vec![1, 512, 0], vec![]),
            latent: (vec![1, 1024, 0], vec![]),
            conv: (vec![1, 1024, 0], vec![]),
            kv: (0..2 * LAYERS).map(|_| (vec![1, 16, 0, 64], vec![])).collect(),
            skip: 0,
        }
    }
}

fn t16(x: &(Vec<i64>, Vec<f16>)) -> anyhow::Result<Tensor<f16>> {
    Ok(Tensor::from_array((x.0.clone(), x.1.clone()))?)
}

fn take16(v: &ort::value::DynValue) -> anyhow::Result<(Vec<i64>, Vec<f16>)> {
    let (shape, data) = v.try_extract_tensor::<f16>()?;
    Ok((shape.iter().copied().collect(), data.to_vec()))
}

/// Frames → sound, twelve at a time, carrying the decoder's state (decoder.py).
fn decode(sess: &mut Session, codes: &[[i64; Q]], mut state: DecoderState, is_final: bool)
          -> anyhow::Result<(Vec<f32>, DecoderState)> {
    let mut audio = vec![];
    let chunks: Vec<&[[i64; Q]]> = codes.chunks(DECODE_CHUNK).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let last = is_final && i + 1 == chunks.len();
        let flat: Vec<i64> = chunk.iter().flatten().copied().collect();
        let mut inputs: Vec<(String, ort::session::SessionInputValue)> = vec![
            ("audio_codes".into(), Tensor::from_array((vec![1i64, chunk.len() as i64, Q as i64], flat))?.into()),
            ("is_last".into(), Tensor::from_array((vec![1i64], vec![f16::from_f32(if last { 1.0 } else { 0.0 })]))?.into()),
            ("pre_conv_history".into(), t16(&state.pre_conv)?.into()),
            ("latent_buffer".into(), t16(&state.latent)?.into()),
            ("conv_history".into(), t16(&state.conv)?.into()),
        ];
        for l in 0..LAYERS {
            inputs.push((format!("past_key_{l}"), t16(&state.kv[2 * l])?.into()));
            inputs.push((format!("past_value_{l}"), t16(&state.kv[2 * l + 1])?.into()));
        }
        let out = sess.run(inputs)?;
        let wav: Vec<f32> = match out[0].try_extract_tensor::<f16>() {
            Ok((_, d)) => d.iter().map(|v| v.to_f32()).collect(),
            Err(_) => out[0].try_extract_tensor::<f32>()?.1.to_vec(),
        };
        let valid = match out[1].try_extract_tensor::<i64>() {
            Ok((_, d)) => d[0] as usize,
            Err(_) => out[1].try_extract_tensor::<i32>()?.1[0] as usize,
        };
        let skip = state.skip;
        let mut next = DecoderState {
            pre_conv: take16(&out[2])?,
            latent: take16(&out[3])?,
            conv: take16(&out[4])?,
            kv: vec![],
            skip: 0,
        };
        for l in 0..LAYERS {
            next.kv.push(take16(&out[5 + l])?);
            next.kv.push(take16(&out[5 + LAYERS + l])?);
        }
        let mut piece = if last { wav } else { wav[..valid.min(wav.len())].to_vec() };
        let mut skip_left = skip;
        if skip_left > 0 && !piece.is_empty() {
            if piece.len() <= skip_left {
                skip_left -= piece.len();
                piece.clear();
            } else {
                piece.drain(..skip_left);
                skip_left = 0;
            }
        }
        next.skip = if last { 4 * SAMPLES_PER_FRAME } else { skip_left };
        audio.extend(piece);
        state = next;
    }
    Ok((audio, state))
}

// ------------------------------------------------------------------- голос

#[derive(Clone)]
pub struct Voice {
    text: String,
    codes: Vec<[i64; Q]>,
    spk: Vec<f32>,
    state: DecoderState,
}

struct Sessions {
    decoder: Session,
    codec_enc: Session,
    spk_enc: Session,
}

pub struct Qwen {
    api: Api,
    talker: Model,
    predictor: Model,
    assets: Assets,
    tokenizer: tokenizers::Tokenizer,
    run: Mutex<Run>,
    voices: Mutex<HashMap<(PathBuf, String), Voice>>,
    pub device: String,
}

struct Run {
    talker: Ctx,
    predictor: Ctx,
    s: Sessions,
}

pub struct Progress {
    pub frames: usize,
}

impl Qwen {
    pub fn load(files: &Files, gpu: bool) -> anyhow::Result<Qwen> {
        let lib = super::lib_dir()?;
        let api = Api::load(&lib)?;
        let talker = Model::load(&api, &files.dir.join(&files.talker), gpu)?;
        let predictor = Model::load(&api, &files.dir.join(&files.predictor), gpu)?;
        let assets = Assets::load(&files.dir)?;
        let tokenizer = tokenizers::Tokenizer::from_file(files.dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let t_ctx = Ctx::new(&api, &talker, TALKER_CTX, true, TALKER_CTX as usize)?;
        let p_ctx = Ctx::new(&api, &predictor, 64, false, 2)?;
        super::init_onnx(&lib)?;
        let s = Sessions {
            decoder: super::session(&files.dir.join("qwen3_tts_decoder.fp16.onnx"), gpu)?,
            codec_enc: super::session(&files.dir.join("qwen3_tts_codec_encoder.fp32.onnx"), false)?,
            spk_enc: super::session(&files.dir.join("qwen3_tts_speaker_encoder.fp32.onnx"), false)?,
        };
        super::trim_heap();
        Ok(Qwen {
            device: if gpu { "cuda".into() } else { "cpu".into() },
            api,
            talker,
            predictor,
            assets,
            tokenizer,
            run: Mutex::new(Run { talker: t_ctx, predictor: p_ctx, s }),
            voices: Mutex::default(),
        })
    }

    fn ids(&self, text: &str) -> anyhow::Result<Vec<usize>> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as usize).collect())
    }

    /// Codes, speaker vector and decoder state of a reference clip, cached per
    /// file and transcript: preparing them costs more than a short phrase.
    fn voice(&self, run: &mut Run, wav: &Path, text: &str) -> anyhow::Result<Voice> {
        let key = (wav.to_path_buf(), text.to_string());
        if let Some(v) = self.voices.lock().unwrap().get(&key) {
            return Ok(v.clone());
        }
        let data = std::fs::read(wav)?;
        let (samples, sr) = crate::audio::read_wav(&data).context("reference must be a wav")?;
        let samples = if sr != SAMPLE_RATE { crate::audio::Resampler::whole(sr, SAMPLE_RATE, &samples) } else { samples };
        let n = samples.len();
        let out = run.s.codec_enc.run(ort::inputs!["input_values" => Tensor::from_array((vec![1i64, n as i64], samples.clone()))?])?;
        let (shape, codes) = out["audio_codes"].try_extract_tensor::<i64>()?;
        let frames = shape[1] as usize;
        let codes: Vec<[i64; Q]> = (0..frames).map(|t| std::array::from_fn(|q| codes[t * Q + q])).collect();
        drop(out);
        let mels = mel::log_mel(&samples);
        let flat: Vec<f32> = mels.iter().flatten().copied().collect();
        let out = run.s.spk_enc.run(ort::inputs!["mels" => Tensor::from_array((vec![1i64, mels.len() as i64, 128], flat))?])?;
        let spk = out["spk_emb"].try_extract_tensor::<f32>()?.1.to_vec();
        drop(out);
        let (_, state) = decode(&mut run.s.decoder, &codes, DecoderState::empty(), true)?;
        let v = Voice { text: text.into(), codes, spk, state };
        self.voices.lock().unwrap().insert(key, v.clone());
        Ok(v)
    }

    /// The prompt for cloning (prompt_builder.py, the ICL mode): a role header,
    /// language and speaker, then the reference transcript + text fused step by
    /// step with the reference audio; the text that does not fit goes into the
    /// generation loop one token per frame.
    fn prompt(&self, text: &str, voice: &Voice, lang: Option<usize>) -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let a = &self.assets;
        let h = a.hidden;
        let pad = a.text(TTS_PAD);
        let mut rows: Vec<Vec<f32>> = vec![];
        for id in self.ids("<|im_start|>assistant\n")? {
            rows.push(a.text(id));
        }
        let pre = match lang {
            Some(l) => vec![THINK, THINK_BOS, l, THINK_EOS],
            None => vec![NOTHINK, THINK_BOS, THINK_EOS],
        };
        for id in pre {
            let mut r = pad.clone();
            add(&mut r, &a.codec(0, id));
            rows.push(r);
        }
        let mut r = pad.clone();
        add(&mut r, &voice.spk);
        rows.push(r);
        let mut r = a.text(TTS_BOS);
        add(&mut r, &a.codec(0, PAD));
        rows.push(r);

        let mut text_ids = self.ids(&format!("{}{}", voice.text, text))?;
        text_ids.push(TTS_EOS);
        let mut audio: Vec<Vec<f32>> = vec![a.codec(0, BOS)];
        for frame in &voice.codes {
            let mut s = vec![0f32; h];
            for (q, &c) in frame.iter().enumerate() {
                a.codec[q].add_row(c as usize, &mut s);
            }
            audio.push(s);
        }
        let mut trailing = vec![];
        for (i, v) in audio.iter().enumerate() {
            let mut r = if i < text_ids.len() { a.text(text_ids[i]) } else { pad.clone() };
            add(&mut r, v);
            rows.push(r);
        }
        if text_ids.len() > audio.len() {
            trailing = text_ids[audio.len()..].iter().map(|&id| a.text(id)).collect();
        }
        Ok((rows.concat(), trailing))
    }

    /// Clone the voice and read `text`. `on_frame` is told the frame count after
    /// each one and returns false to stop; then the result is None.
    ///
    /// A long text is read in pieces of whole sentences, each from the same
    /// reference: one pass holds at most TALKER_CTX positions — the prompt and
    /// every frame — and the talker's memory grows with them.
    pub fn speak(&self, text: &str, ref_wav: &Path, ref_text: &str, lang: Option<usize>, seed: Option<u32>,
                 mut on_frame: impl FnMut(Progress) -> bool) -> anyhow::Result<Option<Vec<f32>>> {
        let mut run = self.run.lock().unwrap();
        let mut audio = vec![];
        let mut done = 0;
        for piece in split_long(text, MAX_PIECE) {
            let mut frames = 0;
            let out = self.speak_piece(&mut run, &piece, ref_wav, ref_text, lang, seed, |p| {
                frames = p.frames;
                on_frame(Progress { frames: done + p.frames })
            })?;
            let Some(a) = out else { return Ok(None) };
            audio.extend(a);
            done += frames;
        }
        Ok(Some(audio))
    }

    #[allow(clippy::too_many_arguments)]
    fn speak_piece(&self, run: &mut Run, text: &str, ref_wav: &Path, ref_text: &str, lang: Option<usize>,
                   seed: Option<u32>, mut on_frame: impl FnMut(Progress) -> bool) -> anyhow::Result<Option<Vec<f32>>> {
        let voice = self.voice(run, ref_wav, ref_text)?;
        let (prompt, trailing) = self.prompt(text, &voice, lang)?;
        let api = &self.api;
        let a = &self.assets;
        let h = a.hidden;
        let n_p = prompt.len() / h;
        let max_frames = (TALKER_CTX as usize).saturating_sub(n_p + 2);
        if max_frames < 16 {
            bail!("text is too long for one pass");
        }
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as u32
        });
        let Run { talker, predictor, s } = run;
        talker.clear(api);
        let pos: Vec<i32> = (0..4).flat_map(|p| (0..n_p as i32).map(move |i| if p < 3 { i } else { 0 })).collect();
        talker.decode_embd(api, &prompt, &pos)?;
        let mut hidden = talker.last_embedding(api, n_p);
        let mut cur = n_p as i32;
        // выборка — как у генерации Qwen: temperature 0.9, top-k 50, штраф за повтор 1.05
        let ts = Sampler::new(api, 0.9, 50, 1.0, 1.05, 128, self.talker.n_vocab, seed);
        let ps = Sampler::new(api, 0.9, 50, 1.0, 1.0, 64, self.predictor.n_vocab, seed);
        let mut codes: Vec<[i64; Q]> = vec![];
        let result = (|| -> anyhow::Result<bool> {
            for step in 0..max_frames {
                let c0 = ts.sample(api, talker, 0, 2048, &[EOS, PAD, BOS]) as usize;
                ts.accept(api, c0 as i32);
                if c0 == EOS {
                    break;
                }
                // малая модель: пятнадцать остальных кодбуков по одному
                let mut frame = [0i64; Q];
                frame[0] = c0 as i64;
                let mut summed = a.codec(0, c0);
                predictor.clear(api);
                let mut first = a.project(&hidden);
                first.extend_from_slice(a.codec_small(0, c0));
                predictor.decode_embd(api, &first, &[0, 1])?;
                for q in 1..Q {
                    let start = (q - 1) * 2048;
                    let code = ps.sample(api, predictor, start, start + 2048, &[]) as usize - start;
                    frame[q] = code as i64;
                    let e = a.codec(q, code);
                    add(&mut summed, &e);
                    if q < Q - 1 {
                        predictor.decode_embd(api, a.codec_small(q, code), &[q as i32 + 1])?;
                    }
                }
                codes.push(frame);
                // обратно в основную модель: звук этого кадра плюс очередной токен текста
                let mut fused = summed;
                match trailing.get(step) {
                    Some(t) => add(&mut fused, t),
                    None => add(&mut fused, &a.text(TTS_PAD)),
                }
                talker.decode_embd(api, &fused, &[cur, cur, cur, 0])?;
                hidden = talker.last_embedding(api, 1);
                cur += 1;
                if !on_frame(Progress { frames: codes.len() }) {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        ts.free(api);
        ps.free(api);
        if !result? {
            return Ok(None);
        }
        let (audio, _) = decode(&mut s.decoder, &codes, voice.state.clone(), true)?;
        Ok(Some(audio))
    }
}
