//! Звук в opus, mp3 и flac — в самом процессе, без ffmpeg.
//!
//! Настройки те же, что стояли в строке ffmpeg: opus 24 кбит/с, mp3 64 кбит/с,
//! flac по умолчанию, — и те же библиотеки, libopus и LAME, только собранные
//! в бинарник. Остаётся за ffmpeg одно: `aac` и изменение темпа.
//!
//! Зачем: ffmpeg — единственное, что voicy просил поставить отдельно, и на
//! Windows его обычно нет. Без него отказывал даже формат по умолчанию.

use std::os::raw::c_int;

use anyhow::{Context, bail};

const OPUS_BITRATE: i32 = 24_000; // mp3 задаётся перечислением LAME: Bitrate::Kbps64

/// Что кодируется здесь; остальное — забота ffmpeg.
pub fn own(fmt: &str) -> bool {
    matches!(fmt, "opus" | "mp3" | "flac")
}

pub fn encode(x: &[f32], sr: u32, fmt: &str) -> anyhow::Result<Vec<u8>> {
    match fmt {
        "opus" => opus(x, sr),
        "mp3" => mp3(x, sr),
        "flac" => flac(x, sr),
        _ => bail!("формат {fmt} кодируется не здесь"),
    }
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    x.iter().map(|&v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16).collect()
}

// ---------------------------------------------------------------------- opus

/// Ogg Opus: заголовок OpusHead, теги, дальше пакеты по 20 мс.
///
/// Частоты, которые понимает кодировщик, — 8, 12, 16, 24 и 48 кГц; синтез
/// отдаёт 24 кГц. Позиции в контейнере считаются в отсчётах 48 кГц независимо
/// от входной частоты — так устроен формат.
fn opus(x: &[f32], sr: u32) -> anyhow::Result<Vec<u8>> {
    const FRAME_MS: usize = 20;
    anyhow::ensure!([8000, 12000, 16000, 24000, 48000].contains(&sr), "opus не берёт {sr} Гц");
    let frame = sr as usize * FRAME_MS / 1000;

    let mut err: c_int = 0;
    let enc = unsafe { audiopus_sys::opus_encoder_create(sr as i32, 1, audiopus_sys::OPUS_APPLICATION_AUDIO, &mut err) };
    if enc.is_null() || err != 0 {
        bail!("opus: кодировщик не создан ({err})");
    }
    let _guard = scopeguard(|| unsafe { audiopus_sys::opus_encoder_destroy(enc) });
    unsafe {
        audiopus_sys::opus_encoder_ctl(enc, audiopus_sys::OPUS_SET_BITRATE_REQUEST, OPUS_BITRATE);
    }
    // задержка кодировщика: столько отсчётов в начале проигрыватель пропускает
    let mut lookahead: c_int = 0;
    unsafe {
        audiopus_sys::opus_encoder_ctl(enc, audiopus_sys::OPUS_GET_LOOKAHEAD_REQUEST, &mut lookahead);
    }
    let pre_skip = (lookahead as u64 * 48000) / sr as u64;

    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1); // версия
    head.push(1); // каналов
    head.extend_from_slice(&(pre_skip as u16).to_le_bytes());
    head.extend_from_slice(&sr.to_le_bytes());
    head.extend_from_slice(&0u16.to_le_bytes()); // усиление
    head.push(0); // раскладка каналов
    let vendor = b"voicy";
    let mut tags = Vec::with_capacity(32);
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes()); // комментариев нет

    let mut out = std::io::Cursor::new(Vec::new());
    let mut w = ogg::writing::PacketWriter::new(&mut out);
    let serial = 0x766f_6963; // «voic»: поток один, лишь бы не менялся
    w.write_packet(head, serial, ogg::writing::PacketWriteEndInfo::EndPage, 0)?;
    w.write_packet(tags, serial, ogg::writing::PacketWriteEndInfo::EndPage, 0)?;

    let mut buf = vec![0u8; 4000];
    let mut granule = pre_skip;
    let total = x.len().div_ceil(frame);
    for (i, chunk) in x.chunks(frame).enumerate() {
        let mut pcm = chunk.to_vec();
        pcm.resize(frame, 0.0); // последний кусок дописывается тишиной
        let n = unsafe {
            audiopus_sys::opus_encode_float(enc, pcm.as_ptr(), frame as i32, buf.as_mut_ptr(), buf.len() as i32)
        };
        if n < 0 {
            bail!("opus: кодирование не удалось ({n})");
        }
        granule += (frame as u64 * 48000) / sr as u64;
        let last = i + 1 == total;
        let end = if last { ogg::writing::PacketWriteEndInfo::EndStream } else { ogg::writing::PacketWriteEndInfo::NormalPacket };
        w.write_packet(buf[..n as usize].to_vec(), serial, end, granule)?;
    }
    Ok(out.into_inner())
}

/// Освободить кодировщик на любом выходе из функции, включая ошибку.
fn scopeguard<F: FnMut()>(f: F) -> impl Drop {
    struct Guard<F: FnMut()>(F);
    impl<F: FnMut()> Drop for Guard<F> {
        fn drop(&mut self) {
            (self.0)()
        }
    }
    Guard(f)
}

// ----------------------------------------------------------------------- mp3

fn mp3(x: &[f32], sr: u32) -> anyhow::Result<Vec<u8>> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};

    let mut b = Builder::new().context("mp3: кодировщик не создан")?;
    b.set_num_channels(1).map_err(|e| anyhow::anyhow!("mp3: {e}"))?;
    b.set_sample_rate(sr).map_err(|e| anyhow::anyhow!("mp3: {sr} Гц — {e}"))?;
    b.set_brate(Bitrate::Kbps64).map_err(|e| anyhow::anyhow!("mp3: {e}"))?;
    b.set_quality(Quality::Good).map_err(|e| anyhow::anyhow!("mp3: {e}"))?;
    let mut enc = b.build().map_err(|e| anyhow::anyhow!("mp3: {e}"))?;

    let pcm = to_i16(x);
    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));
    let n = enc.encode(MonoPcm(&pcm), out.spare_capacity_mut()).map_err(|e| anyhow::anyhow!("mp3: {e}"))?;
    unsafe { out.set_len(out.len() + n) };
    let n = enc.flush::<FlushNoGap>(out.spare_capacity_mut()).map_err(|e| anyhow::anyhow!("mp3: {e}"))?;
    unsafe { out.set_len(out.len() + n) };
    Ok(out)
}

// ---------------------------------------------------------------------- flac

fn flac(x: &[f32], sr: u32) -> anyhow::Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let samples: Vec<i32> = to_i16(x).into_iter().map(i32::from).collect();
    let config = flacenc::config::Encoder::default().into_verified().map_err(|e| anyhow::anyhow!("flac: {e:?}"))?;
    let source = flacenc::source::MemSource::from_samples(&samples, 1, 16, sr as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac: {e:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream.write(&mut sink).map_err(|e| anyhow::anyhow!("flac: {e:?}"))?;
    Ok(sink.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Синус на 24 кГц: проверяем, что получается не пустота и с нужной меткой.
    fn tone(seconds: f64, sr: u32) -> Vec<f32> {
        (0..(seconds * sr as f64) as usize)
            .map(|i| (i as f64 * 440.0 * 2.0 * std::f64::consts::PI / sr as f64).sin() as f32 * 0.5)
            .collect()
    }

    #[test]
    fn opus_is_ogg_with_head() {
        let data = opus(&tone(0.5, 24000), 24000).expect("opus");
        assert_eq!(&data[..4], b"OggS");
        assert!(data.windows(8).any(|w| w == b"OpusHead"));
        assert!(data.len() > 500, "слишком коротко: {}", data.len());
    }

    #[test]
    fn mp3_and_flac_have_their_marks() {
        let mp3 = mp3(&tone(0.5, 24000), 24000).expect("mp3");
        assert!(mp3.len() > 500);
        assert!(mp3[0] == 0xFF || &mp3[..3] == b"ID3", "не похоже на mp3");
        let flac = flac(&tone(0.5, 24000), 24000).expect("flac");
        assert_eq!(&flac[..4], b"fLaC");
    }
}
