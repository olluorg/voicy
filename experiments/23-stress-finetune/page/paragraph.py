"""Абзац из трудных слов: исходная Qwen, v3 без знаков, v3 со знаками по RUAccent.

Озвучивается по предложениям (как делит сервер) и склеивается с паузой 250 мс.
Детектор прогоняется по каждому предложению — для сводки, не для решения.
"""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("FT_OUT", "lora_v3")
sys.path.insert(0, "scripts")
import numpy as np, soundfile as sf, torch
import stress_baseline as sb
import stress_ft_eval as ev

PARAGRAPH = [
    "Большая часть проверок выполняется при сборке, поэтому этот вопрос стоит разобрать подробно.",
    "Мастер быстро починил замок на двери, а старинный замок стоит на высоком холме.",
    "Мы договорились созвониться в среду и начали звонить поставщикам.",
    "Именно долговременным хранением сообщений занимается этот сервис.",
    "Он запрещает удалять элементы, у которых нет ключа.",
    "Красивее всего город выглядит ранним утром, когда жалюзи в офисах ещё закрыты.",
    "Врач посоветовал облегчить нагрузку, а мы углубили анализ и уточнили выводы.",
]
OUT = os.path.join(sb.WORK, "eval", "paragraph")
PAUSE = 0.25


def main():
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    os.makedirs(OUT, exist_ok=True)
    asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe_words(x, sr):
        import librosa
        y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        segs, _ = asr.transcribe(y, language="ru", beam_size=5, word_timestamps=True)
        return [{"word": w.word, "start": w.start, "end": w.end} for s in segs for w in (s.words or [])]

    sb.transcribe_words = transcribe_words
    acc, silero_say, templates_in_context = sb.setup()
    tts = Qwen3TTSModel.from_pretrained(ev.BASE, device_map="cuda:0", dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(ev.OUT, "adapter.json")))
    inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
                                       target_modules=cfg["target_modules"]), tts.model.talker.model)
    set_peft_model_state_dict(tts.model.talker.model, load_file(os.path.join(ev.OUT, "adapter.safetensors")))
    tts.model.eval()

    def adapter(on):
        for m in tts.model.talker.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(on)

    conds = {"base": (False, False), "v3-plain": (True, False), "v3-marked": (True, True)}
    audio = {c: [] for c in conds}
    flagged = {c: [0, 0] for c in conds}
    marked_text = []
    for s in PARAGRAPH:
        marked = acc.process_all(s)
        expected = sb.words_of(marked)
        tpl = templates_in_context(expected)
        marked_text.append(ev.mark_all(marked))
        for c, (on, with_marks) in conds.items():
            adapter(on)
            torch.manual_seed(1)
            w, sr = tts.generate_voice_clone(text=ev.mark_all(marked) if with_marks else s, language="Russian",
                                             ref_audio=ev.VOICE, ref_text=ev.VOICE_TEXT)
            x = np.asarray(w[0], dtype=np.float32)
            audio[c] += [x, np.zeros(int(PAUSE * sr), dtype=np.float32)]
            for _, bad in sb.verdicts(sb.detect(x, sr, expected, tpl), [e for _, e in expected], ev.MARGIN).values():
                flagged[c][0] += bad; flagged[c][1] += 1
        print("  ", s[:60], flush=True)
    for c, parts in audio.items():
        sf.write(os.path.join(OUT, f"{c}.wav"), np.concatenate(parts), sr)
    json.dump({"text": " ".join(PARAGRAPH), "marked": " ".join(marked_text), "flagged": flagged},
              open(os.path.join(OUT, "paragraph.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("размеченный текст:", " ".join(marked_text))
    for c, (b, n) in flagged.items():
        print(f"{c:10} детектор: не там {b} из {n}")


def whole():
    """Весь абзац одним вызовом — как озвучил бы сервер (он делит только после 1500 символов)."""
    from qwen_tts import Qwen3TTSModel
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(OUT, "paragraph.json"), encoding="utf-8"))
    tts = Qwen3TTSModel.from_pretrained(ev.BASE, device_map="cuda:0", dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(ev.OUT, "adapter.json")))
    inject_adapter_in_model(LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0,
                                       target_modules=cfg["target_modules"]), tts.model.talker.model)
    set_peft_model_state_dict(tts.model.talker.model, load_file(os.path.join(ev.OUT, "adapter.safetensors")))
    tts.model.eval()
    for c, on, text in (("base", False, meta["text"]), ("v3-plain", True, meta["text"]),
                        ("v3-marked", True, meta["marked"])):
        for m in tts.model.talker.model.modules():
            if isinstance(m, BaseTunerLayer):
                m.enable_adapters(on)
        torch.manual_seed(1)
        w, sr = tts.generate_voice_clone(text=text, language="Russian", ref_audio=ev.VOICE, ref_text=ev.VOICE_TEXT)
        x = np.asarray(w[0], dtype=np.float32)
        sf.write(os.path.join(OUT, f"whole_{c}.wav"), x, sr)
        print(f"{c:10} {len(x) / sr:.1f} с", flush=True)


if __name__ == "__main__":
    whole() if sys.argv[1:] == ["whole"] else main()
