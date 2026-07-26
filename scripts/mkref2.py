import sys, wave, torch
sys.stdout.reconfigure(encoding="utf-8")
REF = ("Сегодня разберём, как устроены индексы в базе данных, "
       "почему запрос иногда идёт мимо них, и что с этим делать на практике.")
open("ref_text2.txt","w",encoding="utf-8").write(REF)

from vosk_tts import Model, Synth
synth = Synth(Model(model_name="vosk-model-tts-ru-0.9-multi"))
synth.synth(REF, "ref_vosk.wav", speaker_id=4)
with wave.open("ref_vosk.wav") as w: print(f"ref_vosk.wav   {w.getnframes()/w.getframerate():.1f}s")

m = torch.package.PackageImporter("v5_ru.pt").load_pickle("tts_models", "model")
a = m.apply_tts(text=REF, speaker="eugene", sample_rate=48000, put_accent=True, put_yo=True)
with wave.open("ref_silero5.wav","wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes((a*32767).to(torch.int16).numpy().tobytes())
print(f"ref_silero5.wav {len(a)/48000:.1f}s")
