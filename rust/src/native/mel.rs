//! Log-mel spectrogram for the speaker encoder, as the reference implementation
//! computes it (librosa-compatible): 24 kHz, n_fft 1024, hop 256, 128 Slaney
//! mel bands up to 12 kHz, reflect padding of (n_fft - hop) / 2.

use realfft::RealFftPlanner;

const SR: f64 = 24000.0;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: usize = 128;
const FMAX: f64 = 12000.0;

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
}

/// Slaney-normalised filter bank, [N_MELS][N_FFT/2+1], built in f64 and kept in f32.
fn basis() -> Vec<Vec<f32>> {
    let n_bins = N_FFT / 2 + 1;
    let fft: Vec<f64> = (0..n_bins).map(|i| i as f64 * (SR / 2.0) / (n_bins - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(FMAX));
    let mels: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
    (0..N_MELS)
        .map(|i| {
            let (l, c, u) = (mels[i], mels[i + 1], mels[i + 2]);
            let enorm = 2.0 / (u - l);
            fft.iter()
                .map(|&f| (((f - l) / (c - l)).min((u - f) / (u - c)).max(0.0) * enorm) as f32)
                .collect()
        })
        .collect()
}

/// [frames][128]
pub fn log_mel(wav: &[f32]) -> Vec<[f32; N_MELS]> {
    let pad = (N_FFT - HOP) / 2;
    let n = wav.len();
    // отражение без повторения крайнего отсчёта, как np.pad(mode="reflect")
    let at = |i: isize| -> f32 {
        let mut i = i;
        while i < 0 || i >= n as isize {
            if i < 0 { i = -i; }
            if i >= n as isize { i = 2 * (n as isize - 1) - i; }
        }
        wav[i as usize]
    };
    let padded: Vec<f32> = (-(pad as isize)..(n + pad) as isize).map(at).collect();
    let frames = if padded.len() >= N_FFT { 1 + (padded.len() - N_FFT) / HOP } else { 0 };
    let window: Vec<f32> =
        (0..N_FFT).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32).collect();
    let basis = basis();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let mut out = Vec::with_capacity(frames);
    for t in 0..frames {
        for (k, x) in input.iter_mut().enumerate() {
            *x = padded[t * HOP + k] * window[k];
        }
        fft.process(&mut input, &mut spec).expect("fft");
        let mag: Vec<f32> = spec.iter().map(|c| (c.norm_sqr() + 1e-9).sqrt()).collect();
        let mut row = [0f32; N_MELS];
        for (m, b) in basis.iter().enumerate() {
            let s: f32 = b.iter().zip(&mag).map(|(w, v)| w * v).sum();
            row[m] = s.max(1e-5).ln();
        }
        out.push(row);
    }
    out
}
