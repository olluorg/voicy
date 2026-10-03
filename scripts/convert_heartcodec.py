"""HeartCodec → три ONNX-графа для родного движка HeartMuLa (native/heartmula.rs).

    ~/src/heartlib/.venv/bin/python scripts/convert_heartcodec.py CKPT OUT [--check]

Кодек превращает коды HeartMuLa (8 кодбуков, 12.5 Гц) в стерео 48 кГц окнами
по 29.76 с: условия из кодов → flow matching (DiT, 10 шагов Эйлера с CFG 1.25)
→ латент 25 Гц → свёрточный декодер. В ONNX уходят три части с постоянными
формами окна; шаги Эйлера, CFG, случайный шум и склейка окон с перекрытием —
забота движка, как и в heartlib (heartcodec/models/decoder.py):

- heartcodec_cond.onnx    codes int64 [1, 8, 372]       → mu [1, 744, 512]
- heartcodec_dit.onnx     x [2, 744, 1024], t [2]       → v  [2, 744, 256]
  (и heartcodec_dit.fp16.onnx — то же в fp16, входы и выход тоже fp16)
- heartcodec_decode.onnx  latent [2, 128, L]            → wav [2, 1920·L]

С --check каждая часть сверяется с PyTorch на случайном входе.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from heartlib.heartcodec.modeling_heartcodec import HeartCodec

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
check = "--check" in sys.argv
out.mkdir(parents=True, exist_ok=True)

WINDOW_CODES = int(29.76 * 12.5)   # 372
WINDOW_LATENT = int(29.76 * 25)    # 744
WINDOW_SAMPLES = int(29.76 * 48000)

codec = HeartCodec.from_pretrained(str(ckpt / "HeartCodec-oss"), torch_dtype=torch.float32).eval()
fm = codec.decoder.flow_matching
sq = codec.decoder.scalar_model


class Cond(torch.nn.Module):
    """Коды → условия DiT, как в начале inference_codes."""
    def __init__(self):
        super().__init__()
        self.vq, self.lin = fm.vq_embed, fm.cond_feature_emb

    def forward(self, codes):
        e = self.vq.get_output_from_indices(codes.transpose(1, 2))
        e = self.lin(e)
        return torch.repeat_interleave(e, 2, dim=1)       # interpolate ×2, nearest


class Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.est = fm.estimator

    def forward(self, x, t):
        return self.est(x, timestep=t)


class Decode(torch.nn.Module):
    """Длина по времени — переменная: движок декодирует окно кусками (активации
    на всём окне — 4.7 ГБ, кусками по 186 кадров — 1.9 ГБ). Свёртки каузальные,
    кроме первой («заглядывание» на 2 кадра), так что кусок с запасом 32 кадра
    слева и 4 справа совпадает с целым окном до 1e-3."""
    def __init__(self):
        super().__init__()
        self.sq = sq

    def forward(self, latent):
        return self.sq.decode(latent).squeeze(1)


def export(m, args, names, outs, path, axes=None):
    """Граф больше 2 ГБ torch выгружает с весами в сотнях отдельных файлов: экспорт идёт
    во временный каталог, а оттуда граф пересохраняется с весами в одном .data."""
    import tempfile
    import onnx
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        raw = Path(tmp) / path.name
        torch.onnx.export(m, args, str(raw), input_names=names, output_names=outs, opset_version=18,
                          dynamo=False, do_constant_folding=True, dynamic_axes=axes)
        model = onnx.load(str(raw))
        big = sum(t.ByteSize() for t in model.graph.initializer) > 1.5e9
        onnx.save(model, str(path), save_as_external_data=big, all_tensors_to_one_file=True,
                  location=path.name + ".data")
    print(f"{path.name}: {sum(f.stat().st_size for f in path.parent.glob(path.name + '*')) / 1e9:.2f} GB", flush=True)


def compare(path, m, feeds):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = s.run(None, {k: v.numpy() for k, v in feeds.items()})[0]
    with torch.no_grad():
        ref = m(*feeds.values()).numpy()
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
    print(f"  {path.name}: max rel err {err:.2e}", flush=True)


with torch.no_grad():
    codes = torch.randint(0, 8192, (1, 8, WINDOW_CODES), dtype=torch.int64)
    export(Cond(), (codes,), ["codes"], ["mu"], out / "heartcodec_cond.onnx")
    x = torch.randn(2, WINDOW_LATENT, 1024)
    t = torch.tensor([0.3, 0.3])
    export(Dit(), (x, t), ["x", "t"], ["v"], out / "heartcodec_dit.onnx")
    lat = torch.randn(2, 128, WINDOW_LATENT) * 0.5
    export(Decode(), (lat,), ["latent"], ["wav"], out / "heartcodec_decode.onnx",
           axes={"latent": {2: "frames"}, "wav": {1: "samples"}})
    # нулевое условие для безусловной ветви и пустых кадров
    np.save(out / "heartcodec_zero_cond.npy", fm.zero_cond_embedding1.detach().float().numpy())

# DiT в fp16 — 3 ГБ вместо 5.9: в fp32 рядом с бэкбоном он не помещается на 10 ГБ.
# Входы и выход тоже fp16: с keep_io_types=True конвертер оставляет ветвь
# эмбеддинга времени в f32, и она упирается в Gemm с весами fp16.
import onnx
from onnxconverter_common import float16
m16 = float16.convert_float_to_float16(onnx.load(str(out / "heartcodec_dit.onnx")), keep_io_types=False,
                                       disable_shape_infer=True)
onnx.save(m16, str(out / "heartcodec_dit.fp16.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
          location="heartcodec_dit.fp16.onnx.data")
print("heartcodec_dit.fp16.onnx: готово", flush=True)

if check:
    compare(out / "heartcodec_cond.onnx", Cond(), {"codes": codes})
    compare(out / "heartcodec_dit.onnx", Dit(), {"x": x, "t": t})
    compare(out / "heartcodec_decode.onnx", Decode(), {"latent": lat})
print("готово:", out, flush=True)
