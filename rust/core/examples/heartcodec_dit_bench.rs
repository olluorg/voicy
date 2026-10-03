//! Один шаг DiT кодека HeartMuLa на случайных данных: сколько видеопамяти и времени.
//!     cargo run --release -p voicy-core --features internal --example heartcodec_dit_bench -- DIR [steps]
use std::time::Instant;

use half::f16;
use ort::value::Tensor;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&a[0]);
    let steps: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    voicy_core::native::init_onnx(&voicy_core::native::lib_dir()?)?;
    let t = Instant::now();
    let mut s = voicy_core::native::lean_session(&dir.join("heartcodec_dit.fp16.onnx"), true)?;
    eprintln!("загружен за {:.1} с", t.elapsed().as_secs_f64());
    for i in 0..steps {
        let x: Vec<f16> = (0..2 * 744 * 1024).map(|k| f16::from_f32(((k % 97) as f32 - 48.0) / 50.0)).collect();
        let tt = Instant::now();
        let out = s.run(vec![
            ("x".to_string(), ort::session::SessionInputValue::from(Tensor::from_array((vec![2i64, 744, 1024], x))?)),
            ("t".to_string(), ort::session::SessionInputValue::from(Tensor::from_array((vec![2i64], vec![f16::from_f32(0.3); 2]))?)),
        ])?;
        let v = out[0].try_extract_tensor::<f16>()?.1.len();
        eprintln!("шаг {i}: {:.2} с, выход {v}", tt.elapsed().as_secs_f64());
    }
    Ok(())
}
