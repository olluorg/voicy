//! Окружающий звук после модели: петля без шва, фон любой длины, фон под речью.
//!
//! Модели звуков отдают от десятка секунд до пары минут за вызов, а ветру под
//! получасовой статьёй нужна вся её длина. Удлиняет не модель, а обработка:
//! хвост куска вплетается в его начало, и кусок можно повторять без щелчка
//! и без провала громкости на стыке.
//!
//! Под речью фон приглушается там, где звучит голос, и возвращается в паузах —
//! так делают в радиоэфире, и так речь остаётся разборчивой (ADR 0001 мерит
//! её тем же CER).

use std::f64::consts::FRAC_PI_2;

/// с — сколько хвоста вплетается в начало петли. Полсекунды хватает, чтобы
/// шов растворился в шуме воды или ветра, и почти не съедает длину куска.
pub const LOOP_FADE: f64 = 0.5;

/// с — затухание по краям готового фона, как у реплик (ADR 0009), только длиннее:
/// фон должен входить и уходить, а не включаться.
pub const EDGE_FADE: f64 = 0.3;

/// Кусок, который повторяется без шва: последние `fade` секунд вплетены
/// в начало с равной мощностью (синус и косинус), и кусок становится на них
/// короче. Конец результата переходит в его же начало так же гладко, как
/// исходный звук шёл дальше. Для шумовых текстур — вода, ветер, дождь —
/// равная мощность и нужна: некоррелированные сигналы при линейной склейке
/// дали бы провал громкости на 3 дБ посередине шва.
pub fn seamless(x: &[f32], sr: u32, fade: f64) -> Vec<f32> {
    let n = x.len();
    let f = ((fade * sr as f64) as usize).min(n / 2);
    if f < 2 {
        return x.to_vec();
    }
    let mut y = x[..n - f].to_vec();
    for i in 0..f {
        let t = (i as f64 + 0.5) / f as f64 * FRAC_PI_2;
        y[i] = x[i] * t.sin() as f32 + x[n - f + i] * t.cos() as f32;
    }
    y
}

/// Петля, повторённая до `len` отсчётов. Петля должна быть из [`seamless`]:
/// иначе на каждом повторе будет щелчок.
pub fn tile(lp: &[f32], len: usize) -> Vec<f32> {
    if lp.is_empty() {
        return vec![0.0; len];
    }
    lp.iter().copied().cycle().take(len).collect()
}

/// Затухание по приподнятому косинусу на обоих краях, на месте.
pub fn fade_edges(x: &mut [f32], sr: u32, seconds: f64) {
    let f = ((seconds * sr as f64) as usize).min(x.len() / 2);
    let n = x.len();
    for i in 0..f {
        let g = (0.5 - 0.5 * (std::f64::consts::PI * (i as f64 + 0.5) / f as f64).cos()) as f32;
        x[i] *= g;
        x[n - 1 - i] *= g;
    }
}

/// Как фон уступает голосу.
#[derive(Clone, Debug)]
pub struct Duck {
    /// дБ — насколько глубже уходит фон, пока звучит голос
    pub depth_db: f32,
    /// с — за сколько фон уходит; уходит заранее, на столько же раньше голоса
    pub attack: f64,
    /// с — за сколько возвращается после голоса
    pub release: f64,
    /// RMS блока 10 мс, выше которого блок считается голосом
    pub threshold: f32,
}

impl Default for Duck {
    fn default() -> Self {
        Duck { depth_db: -12.0, attack: 0.08, release: 0.6, threshold: 0.01 }
    }
}

const BLOCK: f64 = 0.010;

/// Усиление фона по отсчётам: 1 в паузах, `depth_db` под голосом. Голос
/// ищется блоками по 10 мс; уход начинается на `attack` раньше голоса, чтобы
/// первый слог не тонул в фоне, а возврат плавный — паузы между словами
/// короче `release`, и фон не дышит на каждой из них.
pub fn duck_gain(voice: &[f32], sr: u32, d: &Duck) -> Vec<f32> {
    let n = voice.len();
    let block = ((BLOCK * sr as f64) as usize).max(1);
    let active: Vec<bool> = voice
        .chunks(block)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt() > d.threshold)
        .collect();
    let ahead = (d.attack / BLOCK).ceil() as usize;
    let lowered: Vec<bool> = (0..active.len()).map(|i| active[i..(i + ahead + 1).min(active.len())].iter().any(|&a| a)).collect();

    let low = 10f32.powf(d.depth_db / 20.0);
    let coef = |s: f64| if s <= 0.0 { 0.0 } else { (-1.0 / (s * sr as f64)).exp() as f32 };
    let (down, up) = (coef(d.attack / 3.0), coef(d.release / 3.0));
    let mut g = 1f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let target = if lowered[i / block] { low } else { 1.0 };
        let c = if target < g { down } else { up };
        g = target + (g - target) * c;
        out.push(g);
    }
    out
}

/// Голос поверх фона: фон на `bed_db` от собственной громкости и уступает
/// голосу по `duck`. Фон короче голоса повторяется (он должен быть петлёй),
/// длиннее — обрезается. Если сумма вышла за предел, тише становится всё
/// целиком, а не обрезаются пики.
pub fn under(voice: &[f32], bed: &[f32], sr: u32, bed_db: f32, duck: &Duck) -> Vec<f32> {
    let bed = if bed.len() >= voice.len() { bed[..voice.len()].to_vec() } else { tile(bed, voice.len()) };
    let level = 10f32.powf(bed_db / 20.0);
    let gain = duck_gain(voice, sr, duck);
    let mut out: Vec<f32> = voice.iter().zip(&bed).zip(&gain).map(|((v, b), g)| v + b * level * g).collect();
    let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.99 {
        let k = 0.99 / peak;
        out.iter_mut().for_each(|v| *v *= k);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16000;

    /// Синус с нецелым числом периодов в куске: повтор как есть даёт скачок на стыке.
    fn sine(seconds: f64, hz: f64) -> Vec<f32> {
        (0..(seconds * SR as f64) as usize)
            .map(|i| (i as f64 * hz * 2.0 * std::f64::consts::PI / SR as f64).sin() as f32 * 0.5)
            .collect()
    }

    /// Шум с повторяемым началом — без внешних зависимостей.
    fn noise(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32 - 0.5) * 0.4
            })
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn loop_has_no_seam() {
        let x = sine(1.0, 101.3);
        let raw_jump = (x[0] - x[x.len() - 1]).abs();
        let y = seamless(&x, SR, 0.1);
        assert_eq!(y.len(), x.len() - SR as usize / 10);
        let jump = (y[0] - y[y.len() - 1]).abs();
        let step = (x[1] - x[0]).abs().max((x[2] - x[1]).abs());
        assert!(raw_jump > 5.0 * step, "у заготовки должен быть шов: {raw_jump} против {step}");
        assert!(jump < 2.0 * step, "шов остался: {jump} против шага {step}");
    }

    #[test]
    fn loop_keeps_loudness_across_the_join() {
        let x = noise(SR as usize * 2, 7);
        let y = seamless(&x, SR, 0.5);
        let f = SR as usize / 2;
        // шов — первые полсекунды петли; громкость там та же, что посередине
        let (join, body) = (rms(&y[..f]), rms(&y[f..2 * f]));
        assert!((join / body - 1.0).abs() < 0.15, "провал на шве: {join} против {body}");
    }

    #[test]
    fn tile_repeats_to_length() {
        let lp = vec![1.0, 2.0, 3.0];
        assert_eq!(tile(&lp, 7), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]);
        assert_eq!(tile(&[], 2), vec![0.0, 0.0]);
    }

    #[test]
    fn edges_fade_to_zero() {
        let mut x = vec![1.0f32; SR as usize];
        fade_edges(&mut x, SR, 0.1);
        assert!(x[0] < 0.01 && x[x.len() - 1] < 0.01);
        assert_eq!(x[x.len() / 2], 1.0);
    }

    /// Тишина — голос — тишина, по секунде.
    fn speech_like() -> Vec<f32> {
        let mut v = vec![0f32; SR as usize];
        v.extend(sine(1.0, 200.0));
        v.extend(vec![0f32; SR as usize]);
        v
    }

    #[test]
    fn bed_steps_back_under_the_voice() {
        let d = Duck::default();
        let g = duck_gain(&speech_like(), SR, &d);
        let at = |s: f64| g[(s * SR as f64) as usize];
        let low = 10f32.powf(d.depth_db / 20.0);
        assert!(at(0.5) > 0.99, "в паузе фон не тронут: {}", at(0.5));
        assert!(at(1.0) < 0.5, "к началу голоса фон уже ушёл: {}", at(1.0));
        assert!((at(1.5) - low).abs() < 0.01, "под голосом — {low}: {}", at(1.5));
        assert!(at(2.05) < 0.6, "сразу после голоса фон возвращается плавно: {}", at(2.05));
        assert!(at(2.95) > 0.95, "к концу паузы вернулся: {}", at(2.95));
    }

    #[test]
    fn mix_never_clips() {
        let voice = sine(1.0, 200.0).iter().map(|v| v * 1.9).collect::<Vec<_>>();
        let bed = noise(SR as usize / 3, 3);
        let out = under(&voice, &bed, SR, 0.0, &Duck { depth_db: 0.0, ..Duck::default() });
        assert_eq!(out.len(), voice.len());
        assert!(out.iter().all(|v| v.abs() <= 0.99 + 1e-6));
    }
}
