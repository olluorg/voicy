//! GigaAM-v3 CTC recognition on ONNX Runtime CPU (`ort`), with Silero VAD
//! chunking so peak RAM stays flat (~350-420 MB) even on long audio files.

use std::path::Path;
use std::sync::Mutex;

use anyhow::Context;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use realfft::RealFftPlanner;
use serde_json::{Value, json};

use super::listen::{VAD_FRAME, Vad, VadStream};
use super::whisper::Request;

const SR: usize = 16_000;
const N_FFT: usize = 320;
const HOP: usize = 160;
const N_MELS: usize = 64;
const CLAMP_MIN: f32 = 1e-9;
const CLAMP_MAX: f32 = 1e9;
const MAX_CHUNK_SAMPLES: usize = 18 * SR; // keep Conformer activations small on 800 MB VPS

fn cpu_threads() -> usize {
    if let Ok(v) = std::env::var("OMP_NUM_THREADS")
        && let Ok(n) = v.parse::<usize>()
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get().min(4))
}

fn hz_to_mel_htk(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz_htk(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// HTK triangular mel filterbank [n_bins][n_mels] for 16 kHz, n_fft=320, 64 mels (0..8000 Hz).
fn htk_mel_basis() -> Vec<[f32; N_MELS]> {
    let n_bins = N_FFT / 2 + 1;
    let freqs: Vec<f64> = (0..n_bins).map(|i| i as f64 * (SR as f64 / 2.0) / (n_bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel_htk(0.0), hz_to_mel_htk(SR as f64 / 2.0));
    let m_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_htk(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![[0.0f32; N_MELS]; n_bins];
    for (k, &f) in freqs.iter().enumerate() {
        for m in 0..N_MELS {
            let (l, c, u) = (m_pts[m], m_pts[m + 1], m_pts[m + 2]);
            let up = (f - l) / (c - l);
            let down = (u - f) / (u - c);
            fb[k][m] = up.min(down).max(0.0) as f32;
        }
    }
    fb
}

/// Periodic Hann window of length N_FFT.
fn periodic_hann() -> Vec<f32> {
    (0..N_FFT)
        .map(|i| (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())) as f32)
        .collect()
}

/// Compute [1, 64, n_frames] log-mel features matching `GigaamPreprocessorNumpy("gigaam_v3")`.
fn log_mel_v3(wav: &[f32], window: &[f32], fb: &[[f32; N_MELS]]) -> (Vec<f32>, usize) {
    if wav.len() < N_FFT {
        return (vec![], 0);
    }
    let n_frames = (wav.len() - N_FFT) / HOP + 1;
    let n_bins = N_FFT / 2 + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let rfft = planner.plan_fft_forward(N_FFT);
    let mut buf = rfft.make_input_vec();
    let mut spec = rfft.make_output_vec();
    let mut scratch = rfft.make_scratch_vec();

    // Layout: [N_MELS][n_frames] (channel-first)
    let mut out = vec![0.0f32; N_MELS * n_frames];
    let mut power = vec![0.0f32; n_bins];

    for t in 0..n_frames {
        let start = t * HOP;
        for i in 0..N_FFT {
            buf[i] = wav[start + i] * window[i];
        }
        rfft.process_with_scratch(&mut buf, &mut spec, &mut scratch).expect("rfft");
        for k in 0..n_bins {
            let c = spec[k];
            power[k] = c.re * c.re + c.im * c.im;
        }
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for k in 0..n_bins {
                acc += power[k] * fb[k][m];
            }
            out[m * n_frames + t] = acc.clamp(CLAMP_MIN, CLAMP_MAX).ln();
        }
    }
    (out, n_frames)
}

fn load_vocab(path: &Path) -> anyhow::Result<(Vec<String>, usize)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("cannot read vocab {}", path.display()))?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for line in raw.lines() {
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() {
            continue;
        }
        let Some((tok, id_str)) = line.rsplit_once(' ') else {
            continue;
        };
        let id: usize = id_str.parse().with_context(|| format!("bad vocab id in {line:?}"))?;
        pairs.push((id, tok.to_string()));
    }
    let max_id = pairs.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut tokens = vec![String::new(); max_id + 1];
    for (id, tok) in pairs {
        tokens[id] = tok;
    }
    let blank_id = tokens.iter().position(|t| t == "<blk>").unwrap_or(tokens.len().saturating_sub(1));
    Ok((tokens, blank_id))
}

fn decode_ctc(logits: &[f32], out_len: usize, vocab_size: usize, tokens: &[String], blank_id: usize) -> String {
    let mut prev = blank_id;
    let mut text = String::new();
    for t in 0..out_len {
        let row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i;
            }
        }
        if best_id != blank_id && best_id != prev {
            if let Some(tok) = tokens.get(best_id) {
                if tok != "<unk>" && tok != "<blk>" {
                    for ch in tok.chars() {
                        text.push(if ch == '\u{2581}' { ' ' } else { ch });
                    }
                }
            }
        }
        prev = best_id;
    }
    text.trim().to_string()
}

/// Split audio into speech chunks <= MAX_CHUNK_SAMPLES using Silero VAD.
fn vad_chunks(samples: &[f32], vad: Option<&Vad>) -> Vec<(usize, usize)> {
    if samples.len() <= MAX_CHUNK_SAMPLES || vad.is_none() {
        return force_split(0, samples.len());
    }
    let vad = vad.unwrap();
    let mut stream = VadStream::default();
    let probs = match stream.feed(vad, samples) {
        Ok(p) => p,
        Err(_) => return force_split(0, samples.len()),
    };
    if probs.is_empty() {
        return force_split(0, samples.len());
    }

    // Group frames with prob >= 0.45, bridging short pauses <= 10 frames (~320 ms)
    let min_silence_frames = 10usize;
    let pad_frames = 6usize; // ~192 ms padding around speech
    let mut raw_segs: Vec<(usize, usize)> = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0usize;
    let mut silence_run = 0usize;

    for (i, &p) in probs.iter().enumerate() {
        if p >= 0.45 {
            if !in_speech {
                in_speech = true;
                seg_start = i.saturating_sub(pad_frames);
            }
            silence_run = 0;
        } else if in_speech {
            silence_run += 1;
            if silence_run >= min_silence_frames {
                let end = (i + 1 - silence_run + pad_frames).min(probs.len());
                raw_segs.push((seg_start * VAD_FRAME, (end * VAD_FRAME).min(samples.len())));
                in_speech = false;
                silence_run = 0;
            }
        }
    }
    if in_speech {
        raw_segs.push((seg_start * VAD_FRAME, samples.len()));
    }
    if raw_segs.is_empty() {
        return vec![];
    }

    // Merge nearby segments up to MAX_CHUNK_SAMPLES, and split any oversized segment
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in raw_segs {
        if let Some(last) = merged.last_mut()
            && e - last.0 <= MAX_CHUNK_SAMPLES
            && s.saturating_sub(last.1) <= SR
        {
            last.1 = e;
            continue;
        }
        merged.extend(force_split(s, e));
    }
    merged
}

fn force_split(start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cur = start;
    while cur < end {
        let next = (cur + MAX_CHUNK_SAMPLES).min(end);
        out.push((cur, next));
        cur = next;
    }
    out
}

pub struct GigaAm {
    session: Mutex<Session>,
    vad: Option<Vad>,
    tokens: Vec<String>,
    blank_id: usize,
    window: Vec<f32>,
    fb: Vec<[f32; N_MELS]>,
    pub model_name: String,
    pub device: String,
}

impl GigaAm {
    pub fn load(model_path: &Path, vocab_path: &Path, vad_path: Option<&Path>, model_name: &str) -> anyhow::Result<GigaAm> {
        let lib = super::lib_dir()?;
        super::init_onnx(&lib)?;
        let threads = cpu_threads();
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", model_path.display()))?;

        let (tokens, blank_id) = load_vocab(vocab_path)?;
        let vad = vad_path.filter(|p| p.is_file()).and_then(|p| Vad::load(p).ok());

        Ok(GigaAm {
            session: Mutex::new(session),
            vad,
            tokens,
            blank_id,
            window: periodic_hann(),
            fb: htk_mel_basis(),
            model_name: model_name.to_string(),
            device: "cpu".to_string(),
        })
    }

    fn recognize_chunk(&self, samples: &[f32]) -> anyhow::Result<String> {
        let (feats, n_frames) = log_mel_v3(samples, &self.window, &self.fb);
        if n_frames == 0 {
            return Ok(String::new());
        }
        let f_tensor = Tensor::from_array(([1usize, N_MELS, n_frames], feats))?;
        let len_tensor = Tensor::from_array(([1usize], vec![n_frames as i64]))?;

        let mut session = self.session.lock().map_err(|_| anyhow::anyhow!("gigaam session poisoned"))?;
        let out = session.run(ort::inputs!["features" => f_tensor, "feature_lengths" => len_tensor])?;
        let (shape, data) = out["log_probs"].try_extract_tensor::<f32>()?;
        let out_len = *shape.get(1).ok_or_else(|| anyhow::anyhow!("unexpected log_probs shape"))? as usize;
        let vocab_size = *shape.get(2).ok_or_else(|| anyhow::anyhow!("unexpected log_probs shape"))? as usize;
        let actual_len = out_len.min((n_frames - 1) / 4 + 1);
        Ok(decode_ctc(data, actual_len, vocab_size, &self.tokens, self.blank_id))
    }

    pub fn transcribe(&self, samples: &[f32], _req: &Request, on_segment: &mut dyn FnMut(f64, f64, &str) -> bool)
                      -> anyhow::Result<Option<Value>> {
        let duration = samples.len() as f64 / SR as f64;
        let chunks = vad_chunks(samples, self.vad.as_ref());
        let mut segments = Vec::new();
        let mut full_text = Vec::new();

        for (s, e) in chunks {
            if e <= s + N_FFT {
                continue;
            }
            let text = self.recognize_chunk(&samples[s..e])?;
            let start_s = (s as f64 / SR as f64 * 100.0).round() / 100.0;
            let end_s = (e as f64 / SR as f64 * 100.0).round() / 100.0;
            if !text.is_empty() {
                if !on_segment(end_s, duration, &text) {
                    return Ok(None);
                }
                full_text.push(text.clone());
                segments.push(json!({
                    "start": start_s,
                    "end": end_s,
                    "text": text,
                    "words": []
                }));
            }
        }

        Ok(Some(json!({
            "text": full_text.join(" "),
            "language": "ru",
            "language_probability": 1.0,
            "duration": (duration * 100.0).round() / 100.0,
            "segments": segments
        })))
    }
}
