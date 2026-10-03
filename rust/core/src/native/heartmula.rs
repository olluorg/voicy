//! HeartMuLa without Python: songs to given lyrics. The backbone (Llama-3.2
//! 3B) and the decoder of the other codebooks (300M) run in llama.cpp from
//! GGUF, the codec (HeartCodec) in ONNX Runtime. A port of heartlib's
//! pipeline (HeartMuLa/heartlib, Apache 2.0); the files are made by
//! scripts/convert_heartmula.py and scripts/convert_heartcodec.py.
//!
//! The shape is Qwen3-TTS's (qwen.rs): one frame (80 ms at 12.5 Hz) is one
//! step; the backbone predicts the first of eight codebooks, the decoder the
//! other seven, and the sum of their eight embeddings goes back into the
//! backbone. Unlike Qwen every draw is guided (CFG 1.5): a second, unconditional
//! pass whose text is one learned embedding runs beside, in its own contexts.
//! The song ends when the model draws an end code, or at the asked ceiling.
//!
//! The codec turns codes into 48 kHz stereo in windows of 29.76 s: conditions
//! from the codes, flow matching in ten Euler steps (CFG 1.25), a convolutional
//! decoder; windows overlap by 4.16 s and are crossfaded (heartcodec/decoder.py).

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context as _, bail};
use half::f16;
use ort::session::Session;
use ort::value::Tensor;

use super::llama::{Api, Context as Ctx, Model, Sampler};
use super::npy::Table;

pub const SAMPLE_RATE: u32 = 48_000;
pub const FRAME_RATE: f64 = 12.5;
const V: usize = 8197; // коды одного кодбука, вместе со служебными
const NQ: usize = 8;
const EMBD: usize = 3072;
const TEXT_BOS: u32 = 128_000;
const TEXT_EOS: u32 = 128_001;
const AUDIO_EOS: usize = 8193; // код ≥ этого — конец песни
const CFG: f32 = 1.5;
const TOP_K: i32 = 50;
const BACKBONE_CTX: u32 = 4096; // текст и до ~5 минут кадров

// кодек: окно 29.76 с
const WIN_CODES: usize = 372;
const WIN_LATENT: usize = 744;
const WIN_SAMPLES: usize = 1_428_480;
const HOP_CODES: usize = WIN_CODES / 93 * 80; // 320
const LATENT_DIM: usize = 256;
const COND_DIM: usize = 512;
const FM_STEPS: usize = 10;
const FM_CFG: f32 = 1.25;
// декодер кодека — кусками: на целом окне он берёт 4.7 ГБ, кусками по 186 кадров — 1.9;
// свёртки каузальные, кроме первой, которая заглядывает на 2 кадра вперёд
const DEC_CHUNK: usize = 186;
const DEC_LEFT: usize = 32;
const DEC_RIGHT: usize = 4;
const SAMPLES_PER_LATENT: usize = 1920;

#[derive(Clone, Debug)]
pub struct Files {
    pub dir: PathBuf,
    /// например heartmula_backbone.q8_0.gguf
    pub backbone: String,
    pub decoder: String,
}

/// Where the song is: frames planned so far, then codec windows decoded.
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    Frames(usize),
    Codec { window: usize, windows: usize },
}

struct Tables {
    audio: Table,
    audio_proj: Table,
    /// проекция в декодер, плотно в f32: она нужна дважды на каждый кадр
    projection: Vec<f32>,
    text: Table,
    uncond: Vec<f32>,
    muq: Vec<f32>,
}

/// Flow matching: conditions from codes and the DiT.
struct Flow {
    cond_net: Session,
    dit: Session,
    /// DiT в fp16 и с входами в fp16 (scripts/convert_heartcodec.py)
    dit_f16: bool,
}

/// The language side: backbone and decoder. The backbone's two contexts are
/// made for each song as long as it needs: two of 4096 positions are 0.9 GB.
struct Lm {
    backbone: Model,
    decoder: Model,
    dec_cond: Ctx,
    dec_uncond: Ctx,
}

impl Lm {
    fn load(api: &Api, files: &Files, gpu: bool) -> anyhow::Result<Lm> {
        let backbone = Model::load(api, &files.dir.join(&files.backbone), gpu)?;
        let decoder = Model::load(api, &files.dir.join(&files.decoder), gpu)?;
        Ok(Lm {
            dec_cond: Ctx::new(api, &decoder, 64, false, 2)?,
            dec_uncond: Ctx::new(api, &decoder, 64, false, 2)?,
            backbone,
            decoder,
        })
    }

    fn free(self, api: &Api) {
        self.dec_cond.free(api);
        self.dec_uncond.free(api);
        self.backbone.free(api);
        self.decoder.free(api);
    }
}

/// The backbone's contexts for one song: conditional and unconditional.
struct Pass {
    cond: Ctx,
    uncond: Ctx,
}

/// What is in VRAM. Without `keep` only the working part is: the language
/// side while frames are written, the codec while they are decoded. Both at
/// once took 9.9 GB of a 10 GB card beside speech, the rest went to shared
/// memory, and a 75 s song took 952 s.
#[derive(Default)]
struct State {
    lm: Option<Lm>,
    flow: Option<Flow>,
    /// свёрточный декодер кодека: на полутора миллионах отсчётов окна он сам берёт
    /// гигабайты, поэтому и он не живёт рядом с DiT
    decode: Option<Session>,
}

pub struct HeartMuLa {
    pub device: String,
    files: Files,
    gpu: bool,
    keep: bool,
    api: Api,
    t: Tables,
    tokenizer: tokenizers::Tokenizer,
    state: Mutex<State>,
}

// Модели и контексты живут, пока жива HeartMuLa; вызовы идут по одному (Mutex).
unsafe impl Send for HeartMuLa {}
unsafe impl Sync for HeartMuLa {}

/// Small xorshift for the codec's noise: seeded, reproducible, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Standard normal, Box–Muller.
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

/// y = W x for a square `n×n` row-major matrix.
fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    w.chunks_exact(n).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

/// Write guided logits over [start, end) of the conditional pass into its own
/// buffer, so the sampler draws from them.
fn guide(api: &Api, cond: &Ctx, uncond: &Ctx, start: usize, end: usize, scale: f32) {
    let u: Vec<f32> = uncond.last_logits(api)[start..end].to_vec();
    let c = &mut cond.last_logits(api)[start..end];
    for (ci, ui) in c.iter_mut().zip(u) {
        *ci = ui + (*ci - ui) * scale;
    }
}

impl HeartMuLa {
    /// `keep` holds every part in VRAM between songs; otherwise each part is
    /// loaded for its phase and given back after it.
    pub fn load(files: &Files, gpu: bool, keep: bool) -> anyhow::Result<HeartMuLa> {
        let lib = super::lib_dir()?;
        let api = Api::load(&lib)?;
        let d = &files.dir;
        let t = Tables {
            audio: Table::open(&d.join("audio_embeddings.npy"))?,
            audio_proj: Table::open(&d.join("audio_embeddings_projected.npy"))?,
            projection: Table::open(&d.join("projection.npy"))?.all(),
            text: Table::open(&d.join("text_embeddings.npy"))?,
            uncond: Table::open(&d.join("unconditional.npy"))?.all(),
            muq: Table::open(&d.join("muq_bias.npy"))?.all(),
        };
        let tokenizer = tokenizers::Tokenizer::from_file(d.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        super::init_onnx(&lib)?;
        let mut state = State::default();
        if keep {
            state.lm = Some(Lm::load(&api, files, gpu)?);
        }
        super::trim_heap();
        Ok(HeartMuLa { device: super::device_name(gpu), files: files.clone(), gpu, keep, api, t, tokenizer,
                       state: Mutex::new(state) })
    }

    fn lm<'a>(&self, st: &'a mut State) -> anyhow::Result<&'a mut Lm> {
        if st.lm.is_none() {
            st.lm = Some(Lm::load(&self.api, &self.files, self.gpu)?);
        }
        Ok(st.lm.as_mut().unwrap())
    }

    fn flow<'a>(&self, st: &'a mut State) -> anyhow::Result<&'a mut Flow> {
        if st.flow.is_none() {
            let d = &self.files.dir;
            // fp16, когда он есть: в fp32 DiT кодека (5.9 ГБ) рядом с бэкбоном не помещается
            let f16 = d.join("heartcodec_dit.fp16.onnx");
            let dit_f16 = f16.is_file();
            st.flow = Some(Flow {
                cond_net: super::lean_session(&d.join("heartcodec_cond.onnx"), self.gpu)?,
                dit: super::lean_session(&if dit_f16 { f16 } else { d.join("heartcodec_dit.onnx") }, self.gpu)?,
                dit_f16,
            });
        }
        Ok(st.flow.as_mut().unwrap())
    }

    fn decoder<'a>(&self, st: &'a mut State) -> anyhow::Result<&'a mut Session> {
        if st.decode.is_none() {
            st.decode = Some(super::lean_session(&self.files.dir.join("heartcodec_decode.onnx"), self.gpu)?);
        }
        Ok(st.decode.as_mut().unwrap())
    }

    /// Text ids as heartlib makes them: lowercased, with BOS and EOS.
    fn ids(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text.to_lowercase(), false).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if ids.first() != Some(&TEXT_BOS) {
            ids.insert(0, TEXT_BOS);
        }
        if ids.last() != Some(&TEXT_EOS) {
            ids.push(TEXT_EOS);
        }
        Ok(ids)
    }

    /// The prompt's rows for both passes: tags, the MuQ slot, lyrics.
    fn prompt(&self, tags: &str, lyrics: &str) -> anyhow::Result<(Vec<f32>, Vec<f32>, usize)> {
        let tags = tags.to_lowercase();
        let tags = if tags.starts_with("<tag>") { tags } else { format!("<tag>{tags}") };
        let tags = if tags.ends_with("</tag>") { tags } else { format!("{tags}</tag>") };
        let (t, l) = (self.ids(&tags)?, self.ids(lyrics)?);
        let n = t.len() + 1 + l.len();
        let (mut cond, mut uncond) = (Vec::with_capacity(n * EMBD), Vec::with_capacity(n * EMBD));
        for &id in &t {
            cond.extend(self.t.text.row(id as usize));
            uncond.extend_from_slice(&self.t.uncond);
        }
        // место MuQ: эталонного звука у heartlib нет, вход нулевой — от muq_linear остаётся сдвиг
        cond.extend_from_slice(&self.t.muq);
        uncond.extend_from_slice(&self.t.uncond);
        for &id in &l {
            cond.extend(self.t.text.row(id as usize));
            uncond.extend_from_slice(&self.t.uncond);
        }
        Ok((cond, uncond, n))
    }

    /// For checking the port against heartlib: the conditional pass only, no
    /// sampling — codebook-0 logits after the prompt, and the decoder's
    /// codebook-1 logits given codebook 0 = `c0`.
    #[doc(hidden)]
    pub fn probe(&self, tags: &str, lyrics: &str, c0: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let api = &self.api;
        let mut guard = self.state.lock().unwrap();
        let r = self.lm(&mut guard)?;
        let (pc, _, n) = self.prompt(tags, lyrics)?;
        let mut cond = Ctx::new(api, &r.backbone, (n + 16) as u32, true, n.max(16))?;
        cond.decode_embd(api, &pc, &(0..n as i32).collect::<Vec<_>>())?;
        let l0 = cond.last_logits(api)[..V].to_vec();
        let h = cond.last_embedding(api, n);
        cond.free(api);
        r.dec_cond.clear(api);
        let mut rows = matvec(&self.t.projection, &h, EMBD);
        rows.extend(self.t.audio_proj.row(c0));
        r.dec_cond.decode_embd(api, &rows, &[0, 1])?;
        let l1 = r.dec_cond.last_logits(api)[..V].to_vec();
        Ok((l0, l1))
    }

    /// One song: `tags` like "pop, female vocal, piano", `lyrics` with
    /// [verse]/[chorus] marks, `max_seconds` the ceiling. The callback gets the
    /// stage and says whether to go on; Ok(None) when it said no.
    pub fn generate(&self, tags: &str, lyrics: &str, max_seconds: f64, seed: u64,
                    mut on_progress: impl FnMut(Stage) -> bool) -> anyhow::Result<Option<Vec<f32>>> {
        let api = &self.api;
        let mut guard = self.state.lock().unwrap();
        let (pc, pu, n) = self.prompt(tags, lyrics)?;
        let max_frames = (max_seconds * FRAME_RATE).ceil() as usize;
        if n + max_frames + 1 > BACKBONE_CTX as usize {
            bail!("lyrics and length do not fit: {n} text tokens + {max_frames} frames > {BACKBONE_CTX}");
        }
        // кодек прошлой песни, если остался, уступает место языковой части
        if !self.keep {
            guard.flow = None;
            guard.decode = None;
        }
        let r = self.lm(&mut guard)?;
        // контексты бэкбона — ровно на эту песню: подсказка и потолок кадров
        let ctx_len = (n + max_frames + 8).next_multiple_of(256) as u32;
        let mut pass = Pass {
            cond: Ctx::new(api, &r.backbone, ctx_len, true, n.max(16))?,
            uncond: Ctx::new(api, &r.backbone, ctx_len, true, n.max(16))?,
        };
        let pos: Vec<i32> = (0..n as i32).collect();
        pass.cond.decode_embd(api, &pc, &pos)?;
        pass.uncond.decode_embd(api, &pu, &pos)?;
        let mut h_c = pass.cond.last_embedding(api, n);
        let mut h_u = pass.uncond.last_embedding(api, n);

        let seed32 = (seed ^ (seed >> 32)) as u32;
        let s0 = Sampler::new(api, 1.0, TOP_K, 1.0, 1.0, 0, V, seed32);
        let sd = Sampler::new(api, 1.0, TOP_K, 1.0, 1.0, 0, (NQ - 1) * V, seed32.wrapping_add(1));
        let mut frames: Vec<[i64; NQ]> = vec![];
        let result = (|| -> anyhow::Result<bool> {
            let mut cur = n as i32;
            for _ in 0..max_frames {
                let mut frame = [0i64; NQ];
                guide(api, &pass.cond, &pass.uncond, 0, V, CFG);
                let c0 = s0.sample(api, &pass.cond, 0, V, &[]) as usize;
                frame[0] = c0 as i64;
                // семь остальных кодбуков: декодер, по коду за шаг, в обеих ветвях
                for (dec, h) in [(&mut r.dec_cond, &h_c), (&mut r.dec_uncond, &h_u)] {
                    dec.clear(api);
                    let mut rows = matvec(&self.t.projection, h, EMBD);
                    rows.extend(self.t.audio_proj.row(c0));
                    dec.decode_embd(api, &rows, &[0, 1])?;
                }
                for q in 1..NQ {
                    let (start, end) = ((q - 1) * V, q * V);
                    guide(api, &r.dec_cond, &r.dec_uncond, start, end, CFG);
                    let code = sd.sample(api, &r.dec_cond, start, end, &[]) as usize - start;
                    frame[q] = code as i64;
                    if q < NQ - 1 {
                        let e = self.t.audio_proj.row(q * V + code);
                        r.dec_cond.decode_embd(api, &e, &[q as i32 + 1])?;
                        r.dec_uncond.decode_embd(api, &e, &[q as i32 + 1])?;
                    }
                }
                if frame.iter().any(|&c| c as usize >= AUDIO_EOS) {
                    break;
                }
                frames.push(frame);
                // обратно в бэкбон: сумма восьми эмбеддингов кадра, одна и та же в обеих ветвях
                let mut row = vec![0f32; EMBD];
                for (q, &c) in frame.iter().enumerate() {
                    self.t.audio.add_row(q * V + c as usize, &mut row);
                }
                pass.cond.decode_embd(api, &row, &[cur])?;
                pass.uncond.decode_embd(api, &row, &[cur])?;
                h_c = pass.cond.last_embedding(api, 1);
                h_u = pass.uncond.last_embedding(api, 1);
                cur += 1;
                if !on_progress(Stage::Frames(frames.len())) {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        s0.free(api);
        sd.free(api);
        pass.cond.free(api);
        pass.uncond.free(api);
        // языковая часть отдаёт видеопамять кодеку (и при отмене, и при ошибке)
        if !self.keep {
            if let Some(lm) = guard.lm.take() {
                lm.free(api);
            }
        }
        if !result? {
            return Ok(None);
        }
        if frames.is_empty() {
            bail!("the model ended the song before it began");
        }
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let (latents, target) = match flow_latents(self.flow(&mut guard)?, &frames, &mut rng, &mut on_progress)? {
            Some(l) => l,
            None => return Ok(None),
        };
        if !self.keep {
            guard.flow = None;
        }
        let out = decode_latents(self.decoder(&mut guard)?, &latents, target);
        if !self.keep {
            guard.decode = None;
        }
        super::trim_heap();
        out.map(Some)
    }
}

fn t32(shape: Vec<i64>, data: Vec<f32>) -> anyhow::Result<ort::session::SessionInputValue<'static>> {
    Ok(Tensor::from_array((shape, data))?.into())
}

fn out32(v: &ort::value::DynValue) -> anyhow::Result<Vec<f32>> {
    Ok(v.try_extract_tensor::<f32>().context("codec output")?.1.to_vec())
}

/// Codes → latents window by window, as heartcodec's detokenize does it;
/// also the length of the sound in samples.
fn flow_latents(r: &mut Flow, frames: &[[i64; NQ]], rng: &mut Rng, on_progress: &mut impl FnMut(Stage) -> bool)
                -> anyhow::Result<Option<(Vec<Vec<f32>>, usize)>> {
    let ovlp_codes = WIN_CODES - HOP_CODES;
    let ovlp_latent = ovlp_codes * 2;
    let target = (frames.len() as f64 / FRAME_RATE * SAMPLE_RATE as f64) as usize;
    // короткие коды повторяются до окна, длинные — до целого числа шагов
    let mut codes: Vec<[i64; NQ]> = frames.to_vec();
    while codes.len() < WIN_CODES {
        codes.extend_from_within(..);
    }
    let rem = (codes.len() - ovlp_codes) % HOP_CODES;
    if rem > 0 {
        let want = (codes.len() - ovlp_codes).div_ceil(HOP_CODES) * HOP_CODES + ovlp_codes;
        while codes.len() < want {
            codes.extend_from_within(..);
        }
        codes.truncate(want);
    }
    codes.truncate(codes.len().max(WIN_CODES));
    let starts: Vec<usize> = (0..=codes.len() - WIN_CODES).step_by(HOP_CODES).collect();

    let mut latents: Vec<Vec<f32>> = vec![];
    for (w, &s) in starts.iter().enumerate() {
        if !on_progress(Stage::Codec { window: w, windows: starts.len() }) {
            return Ok(None);
        }
        // условия: коды окна [1, 8, 372], кодбук за кодбуком
        let mut flat = Vec::with_capacity(NQ * WIN_CODES);
        for q in 0..NQ {
            flat.extend(codes[s..s + WIN_CODES].iter().map(|f| f[q]));
        }
        let mu = {
            let out = r.cond_net.run(vec![("codes".to_string(),
                ort::session::SessionInputValue::from(Tensor::from_array((vec![1i64, NQ as i64, WIN_CODES as i64], flat))?))])?;
            out32(&out[0])?
        };
        // начало окна — из конца прошлого латента
        let inc = if w == 0 { 0 } else { ovlp_latent };
        let mut incontext = vec![0f32; WIN_LATENT * LATENT_DIM];
        if inc > 0 {
            let prev = latents.last().unwrap();
            incontext[..inc * LATENT_DIM].copy_from_slice(&prev[(WIN_LATENT - inc) * LATENT_DIM..]);
        }
        let noise: Vec<f32> = (0..WIN_LATENT * LATENT_DIM).map(|_| rng.normal()).collect();
        let mut x = noise.clone();
        let dt = 1.0 / FM_STEPS as f32;
        let mut t = 0f32;
        for _ in 0..FM_STEPS {
            for i in 0..inc * LATENT_DIM {
                x[i] = (1.0 - (1.0 - 1e-6) * t) * noise[i] + t * incontext[i];
            }
            // батч: безусловная ветвь (нулевые условия), потом условная
            let width = 2 * LATENT_DIM + COND_DIM;
            let mut inp = vec![0f32; 2 * WIN_LATENT * width];
            for b in 0..2 {
                for f in 0..WIN_LATENT {
                    let o = (b * WIN_LATENT + f) * width;
                    inp[o..o + LATENT_DIM].copy_from_slice(&x[f * LATENT_DIM..(f + 1) * LATENT_DIM]);
                    inp[o + LATENT_DIM..o + 2 * LATENT_DIM].copy_from_slice(&incontext[f * LATENT_DIM..(f + 1) * LATENT_DIM]);
                    if b == 1 {
                        inp[o + 2 * LATENT_DIM..o + width].copy_from_slice(&mu[f * COND_DIM..(f + 1) * COND_DIM]);
                    }
                }
            }
            let shape = vec![2, WIN_LATENT as i64, width as i64];
            let v = if r.dit_f16 {
                let h = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
                let out = r.dit.run(vec![
                    ("x".to_string(), ort::session::SessionInputValue::from(Tensor::from_array((shape, h(inp)))?)),
                    ("t".to_string(), ort::session::SessionInputValue::from(Tensor::from_array((vec![2i64], h(vec![t, t])))?)),
                ])?;
                out[0].try_extract_tensor::<f16>()?.1.iter().map(|v| v.to_f32()).collect::<Vec<f32>>()
            } else {
                let out = r.dit.run(vec![("x".to_string(), t32(shape, inp)?), ("t".to_string(), t32(vec![2], vec![t, t])?)])?;
                out32(&out[0])?
            };
            let half = WIN_LATENT * LATENT_DIM;
            for i in 0..half {
                let (vu, vc) = (v[i], v[half + i]);
                x[i] += dt * (vu + FM_CFG * (vc - vu));
            }
            t += dt;
        }
        x[..inc * LATENT_DIM].copy_from_slice(&incontext[..inc * LATENT_DIM]);
        latents.push(x);
    }
    Ok(Some((latents, target)))
}

/// Latents → interleaved stereo at 48 kHz, windows crossfaded.
fn decode_latents(decode: &mut Session, latents: &[Vec<f32>], target: usize) -> anyhow::Result<Vec<f32>> {

    // латент [744, 256] → два канала по 128: [2, 128, 744]
    let ovlp = WIN_SAMPLES - WIN_SAMPLES / 93 * 80;
    let mut out: Vec<Vec<f32>> = vec![vec![], vec![]];
    for lat in latents {
        let mut wav = vec![Vec::with_capacity(WIN_SAMPLES), Vec::with_capacity(WIN_SAMPLES)];
        for a in (0..WIN_LATENT).step_by(DEC_CHUNK) {
            let b = (a + DEC_CHUNK).min(WIN_LATENT);
            let (s0, e) = (a.saturating_sub(DEC_LEFT), (b + DEC_RIGHT).min(WIN_LATENT));
            let len = e - s0;
            // латент [кадры, 256] → два канала по 128: [2, 128, len]
            let mut inp = vec![0f32; 2 * 128 * len];
            for f in 0..len {
                for c in 0..2 {
                    for k in 0..128 {
                        inp[(c * 128 + k) * len + f] = lat[(s0 + f) * LATENT_DIM + c * 128 + k];
                    }
                }
            }
            let res = decode.run(vec![("latent".to_string(), t32(vec![2, 128, len as i64], inp)?)])?;
            let y = out32(&res[0])?;
            let n = y.len() / 2;
            let (from, to) = ((a - s0) * SAMPLES_PER_LATENT, (b - s0) * SAMPLES_PER_LATENT);
            for c in 0..2 {
                wav[c].extend_from_slice(&y[c * n + from..c * n + to.min(n)]);
            }
        }
        for c in 0..2 {
            let cur = &wav[c][..wav[c].len().min(WIN_SAMPLES)];
            let o = &mut out[c];
            if o.is_empty() {
                o.extend_from_slice(cur);
            } else {
                let start = o.len() - ovlp;
                for i in 0..ovlp {
                    let w = i as f32 / (ovlp - 1) as f32;
                    o[start + i] = o[start + i] * (1.0 - w) + cur[i] * w;
                }
                o.extend_from_slice(&cur[ovlp..]);
            }
        }
    }
    let len = out[0].len().min(target);
    Ok((0..len).flat_map(|i| [out[0][i], out[1][i]]).collect())
}
