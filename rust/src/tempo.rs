//! Темп без изменения высоты голоса — WSOLA, тем же приёмом, что `atempo`
//! у ffmpeg: звук нарезается окнами, и каждое следующее окно берётся не строго
//! по расписанию, а там, где оно лучше всего совпадает с уже собранным хвостом.
//! Иначе на стыках слышны щелчки.
//!
//! Растяжение применяется к готовому звуку, а не задаётся модели: у модели
//! замедление роняет разборчивость впятеро (ADR 0005). Замеры своего
//! растяжения против ffmpeg — experiments/24.

/// `factor` — во столько раз речь быстрее: 0.8 даёт звук в 1.25 раза длиннее.
pub fn stretch(x: &[f32], sr: u32, factor: f64) -> Vec<f32> {
    if (factor - 1.0).abs() < 1e-3 || x.is_empty() {
        return x.to_vec();
    }
    let factor = factor.clamp(0.5, 2.0);
    // 30 мс окно и 10 мс поиска — то, на чём обычно останавливаются реализации
    // WSOLA для речи: окно короче ломает низкие частоты, длиннее размазывает атаки.
    let w = (0.030 * sr as f64) as usize & !1;
    let hop_out = w / 2;
    let search = (0.010 * sr as f64) as usize;
    if x.len() < w + search * 2 {
        return x.to_vec();
    }
    let window: Vec<f32> = (0..w)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / w as f64).cos()) as f32)
        .collect();

    let out_len = (x.len() as f64 / factor) as usize + w;
    let mut out = vec![0f32; out_len];
    // «хвост» — вторая половина прошлого окна: с ней ищется совпадение
    let mut tail: Vec<f32> = vec![0.0; hop_out];
    let mut pos_out = 0usize;
    let mut pos_in = 0f64;
    let mut first = true;

    while pos_out + w <= out.len() {
        let want = pos_in as usize;
        let start = if first {
            0
        } else {
            let from = want.saturating_sub(search);
            let to = (want + search).min(x.len().saturating_sub(w));
            best_match(x, &tail, from, to)
        };
        if start + w > x.len() {
            break;
        }
        for i in 0..w {
            out[pos_out + i] += x[start + i] * window[i];
        }
        tail.copy_from_slice(&x[start + hop_out..start + w]);
        pos_out += hop_out;
        pos_in += hop_out as f64 * factor;
        first = false;
        if pos_in as usize + w >= x.len() {
            break;
        }
    }
    out.truncate((x.len() as f64 / factor).round() as usize);
    out
}

/// Где в `x[from..to]` начало отрезка, больше всего похожего на `tail`.
/// Похожесть — нормированная корреляция: громкость на выбор влиять не должна.
fn best_match(x: &[f32], tail: &[f32], from: usize, to: usize) -> usize {
    let n = tail.len();
    let mut best = (from, f32::NEG_INFINITY);
    for start in from..=to.max(from) {
        if start + n > x.len() {
            break;
        }
        let seg = &x[start..start + n];
        let mut dot = 0f32;
        let mut energy = 1e-9f32;
        for i in 0..n {
            dot += seg[i] * tail[i];
            energy += seg[i] * seg[i];
        }
        let score = dot / energy.sqrt();
        if score > best.1 {
            best = (start, score);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f64, sr: u32, hz: f64) -> Vec<f32> {
        (0..(seconds * sr as f64) as usize)
            .map(|i| (i as f64 * hz * 2.0 * std::f64::consts::PI / sr as f64).sin() as f32 * 0.5)
            .collect()
    }

    #[test]
    fn length_follows_the_factor() {
        let x = tone(2.0, 24000, 220.0);
        for factor in [0.8, 0.9, 1.1, 1.2] {
            let y = stretch(&x, 24000, factor);
            let want = x.len() as f64 / factor;
            let off = (y.len() as f64 - want).abs() / want;
            assert!(off < 0.02, "factor {factor}: {} против {want}", y.len());
        }
    }

    #[test]
    fn tone_survives_stretching() {
        // растянутый синус остаётся синусом: без совпадения окон он рассыпается
        let x = tone(1.0, 24000, 220.0);
        let y = stretch(&x, 24000, 0.8);
        let rms = |v: &[f32]| (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt();
        let middle = &y[y.len() / 4..y.len() / 2];
        assert!((rms(middle) / rms(&x) - 1.0).abs() < 0.2, "громкость уехала: {} против {}", rms(middle), rms(&x));
        let zeros = middle.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        let want = 2.0 * 220.0 * middle.len() as f64 / 24000.0;
        assert!((zeros as f64 - want).abs() / want < 0.1, "частота уехала: {zeros} против {want}");
    }

    #[test]
    fn speed_one_is_untouched() {
        let x = tone(0.2, 24000, 440.0);
        assert_eq!(stretch(&x, 24000, 1.0), x);
    }
}
