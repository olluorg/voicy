//! The listening side without Python: the Silero voice detector and Smart Turn,
//! both ONNX models on the CPU — what server/engines/vad_silero.py and
//! turn_smart.py run, in the same runtime.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use realfft::RealFftPlanner;

use super::mel::slaney_basis;

fn cpu_session(path: &Path, threads: usize) -> anyhow::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", path.display()))
}

// ------------------------------------------------------------ детектор голоса

pub const VAD_FRAME: usize = 512; // отсчётов на кадр, 32 мс
const VAD_CONTEXT: usize = 64; // столько предыдущих отсчётов видит кадр

pub struct Vad {
    session: Mutex<Session>,
}

impl Vad {
    pub fn load(path: &Path) -> anyhow::Result<Vad> {
        Ok(Vad { session: Mutex::new(cpu_session(path, 1)?) })
    }
}

/// One session's detector. Each frame sees only the 64 samples before it, so
/// frames are scored as they arrive with the same result as all at once.
pub struct VadStream {
    tail: [f32; VAD_CONTEXT],
    rest: Vec<f32>,
}

impl Default for VadStream {
    fn default() -> Self {
        VadStream { tail: [0.0; VAD_CONTEXT], rest: vec![] }
    }
}

impl VadStream {
    pub fn pending(&self) -> usize {
        self.rest.len()
    }

    pub fn feed(&mut self, vad: &Vad, x: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.rest.extend_from_slice(x);
        let n = self.rest.len() / VAD_FRAME;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut batch = Vec::with_capacity(n * (VAD_CONTEXT + VAD_FRAME));
        for f in 0..n {
            let frame = &self.rest[f * VAD_FRAME..(f + 1) * VAD_FRAME];
            batch.extend_from_slice(&self.tail);
            batch.extend_from_slice(frame);
            self.tail.copy_from_slice(&frame[VAD_FRAME - VAD_CONTEXT..]);
        }
        self.rest.drain(..n * VAD_FRAME);
        let zeros = || Tensor::from_array((vec![1i64, 1, 128], vec![0f32; 128]));
        let mut s = vad.session.lock().unwrap();
        let out = s.run(ort::inputs![
            "input" => Tensor::from_array((vec![n as i64, (VAD_CONTEXT + VAD_FRAME) as i64], batch))?,
            "h" => zeros()?,
            "c" => zeros()?,
        ])?;
        Ok(out["speech_probs"].try_extract_tensor::<f32>()?.1.to_vec())
    }
}

// --------------------------------------------------------------- конец реплики

const TURN_SR: usize = 16000;
const TURN_SECONDS: usize = 8; // столько модель видит, считая от конца
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;

pub struct Turn {
    session: Mutex<Session>,
    basis: Vec<Vec<f32>>,
}

impl Turn {
    pub fn load(path: &Path) -> anyhow::Result<Turn> {
        // два потока: 20 мс хватает и так, ядра нужнее серверу
        Ok(Turn { session: Mutex::new(cpu_session(path, 2)?), basis: slaney_basis(TURN_SR as f64, N_FFT, N_MELS, 8000.0) })
    }

    /// Whisper's log-mel, as transformers' WhisperFeatureExtractor computes it:
    /// zero-mean unit-variance audio, centred STFT with a periodic Hann window,
    /// power spectrum, log10, floor at max − 8, then (x + 4) / 4; the last frame
    /// is dropped. [80][800] for eight seconds.
    fn features(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len();
        let mean = audio.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = audio.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let x: Vec<f32> = audio.iter().map(|&v| ((v as f64 - mean) / (var + 1e-7).sqrt()) as f32).collect();
        let pad = N_FFT / 2;
        let at = |i: isize| -> f32 {
            let mut i = i;
            while i < 0 || i >= n as isize {
                if i < 0 { i = -i; }
                if i >= n as isize { i = 2 * (n as isize - 1) - i; }
            }
            x[i as usize]
        };
        let padded: Vec<f32> = (-(pad as isize)..(n + pad) as isize).map(at).collect();
        let frames = 1 + (padded.len() - N_FFT) / HOP - 1; // последний кадр отбрасывается
        let window: Vec<f32> =
            (0..N_FFT).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32).collect();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut input = fft.make_input_vec();
        let mut spec = fft.make_output_vec();
        let mut mel = vec![0f32; N_MELS * frames];
        for t in 0..frames {
            for (k, v) in input.iter_mut().enumerate() {
                *v = padded[t * HOP + k] * window[k];
            }
            fft.process(&mut input, &mut spec).expect("fft");
            let power: Vec<f32> = spec.iter().map(|c| c.norm_sqr()).collect();
            for (m, b) in self.basis.iter().enumerate() {
                let s: f32 = b.iter().zip(&power).map(|(w, p)| w * p).sum();
                mel[m * frames + t] = s.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f32::MIN, f32::max);
        for v in mel.iter_mut() {
            *v = (v.max(max - 8.0) + 4.0) / 4.0;
        }
        mel
    }

    /// 16 kHz mono of the turn so far, the pause after it included; only the
    /// last eight seconds are looked at. Short audio is padded with zeros at
    /// the *start*, as the model was trained: padding at the end would show it
    /// a phrase and then seconds of silence, and every phrase would look done.
    pub fn probability(&self, audio: &[f32]) -> anyhow::Result<f32> {
        let n = TURN_SECONDS * TURN_SR;
        let tail = &audio[audio.len().saturating_sub(n)..];
        let mut x = vec![0f32; n - tail.len()];
        x.extend_from_slice(tail);
        let feats = self.features(&x);
        let frames = feats.len() / N_MELS;
        let mut s = self.session.lock().unwrap();
        let out = s.run(ort::inputs!["input_features" => Tensor::from_array((vec![1i64, N_MELS as i64, frames as i64], feats))?])?;
        Ok(out[0].try_extract_tensor::<f32>()?.1[0]) // уже после сигмоиды
    }
}
