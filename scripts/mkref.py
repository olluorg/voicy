import sys, wave, torch
sys.stdout.reconfigure(encoding="utf-8")
m = torch.package.PackageImporter("v4_ru.pt").load_pickle("tts_models", "model")
REF = "Сегодня разберём, как устроены индексы в базе данных и почему запрос иногда идёт мимо них."
for spk in ("eugene", "xenia"):
    a = m.apply_tts(text=REF, speaker=spk, sample_rate=48000, put_accent=True, put_yo=True)
    with wave.open(f"ref_{spk}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes((a*32767).to(torch.int16).numpy().tobytes())
    print(f"ref_{spk}.wav  {len(a)/48000:.1f}s")
open("ref_text.txt","w",encoding="utf-8").write(REF)
