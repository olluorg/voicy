"""Training data for teaching Qwen3-TTS to obey stress marks.

Source: Russian LibriSpeech (OpenSLR 96, public domain LibriVox audiobooks),
the Hugging Face copy istupakov/russian_librispeech. Readers of audiobooks stress
normatively almost everywhere, so RUAccent's marks on the original punctuated
text match what was actually said closely enough to learn from.

Every example is laid out the way voicy calls the model, in the ICL cloning mode:
the reference is another utterance of the same reader with its plain transcript,
the target is the utterance with marks. Marks are the combining acute U+0301
after the stressed vowel, a single token in Qwen's vocabulary. Each word is
marked with probability MARK_P, so the model keeps its own judgement for
unmarked words and text without marks reads as before.

Three stages, the middle one a separate process per corpus shard. Encoding grows
the process by about 1.3 GB per 900 utterances and never gives it back, and WSL
here has 16 GB: a single process over the whole selection would not finish, and
the first version of this script took WSL down. A process per shard returns its
memory on exit, and a finished shard is not redone.

    python scripts/stress_ft_data.py            # всё по порядку
    python scripts/stress_ft_data.py encode <shard.parquet>

Output: work/ds/{train,val}.jsonl with codes for target and reference, and 24 kHz
wavs for the speaker encoder.
"""
import glob, io, json, os, random, re, subprocess, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from stress_baseline import load_ruaccent, V, WORK

ACUTE = "́"
MARK_P = 0.7
PER_READER_H = 2.5  # в корпусе 5 дикторов, двое из них — 85% времени; выравниваем
VAL_PER_READER_H = 0.25
DS = os.path.join(WORK, "ds")
ITEMS = os.path.join(DS, "items.json")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots")


def mark(accented: str, rng: random.Random) -> str:
    """RUAccent's «+» before the vowel → U+0301 after it, on a random subset of words.

    One-vowel words and words with ё carry no ambiguity and stay unmarked.
    """
    def one(m):
        w = m.group(0)
        plain = w.replace("+", "")
        if "+" not in w or "ё" in plain.lower() or sum(ch in V for ch in plain.lower()) < 2:
            return plain
        if rng.random() >= MARK_P:
            return plain
        return re.sub(r"\+(.)", lambda v: v.group(1) + ACUTE, w)
    return re.sub(r"[А-Яа-яЁё+\-]+", one, accented)


def shards():
    return sorted(glob.glob(os.path.join(WORK, "ruls", "train-*.parquet"))) + [os.path.join(WORK, "ruls", "test.parquet")]


def select():
    """Only the light columns: the audio of the whole corpus is 11 GB."""
    import pyarrow.parquet as pq
    rng = random.Random(11)
    pool = {}
    for sh in shards():
        split = "val" if sh.endswith("test.parquet") else "train"
        cols = ["duration", "audio_filepath", "score", "text_no_preprocessing"]
        for row, r in enumerate(pq.read_table(sh, columns=cols).to_pylist()):
            if 3.0 <= r["duration"] <= 14.0 and r["score"] > -2.5:
                r.update(reader=r["audio_filepath"].split("/")[1], split=split,
                         shard=os.path.basename(sh), row=row)
                pool.setdefault((split, r["reader"]), []).append(r)
    items = []
    for (split, reader), rs in sorted(pool.items()):
        rng.shuffle(rs)
        cap = (PER_READER_H if split == "train" else VAL_PER_READER_H) * 3600
        got = 0.0
        for r in rs:
            if got >= cap:
                break
            items.append(r); got += r["duration"]
    for split in ("train", "val"):
        sel = [r for r in items if r["split"] == split]
        print(f"{split}: {len(sel)} фраз, {sum(r['duration'] for r in sel) / 3600:.1f} ч, "
              f"{len({r['reader'] for r in sel})} дикторов", flush=True)
    json.dump(items, open(ITEMS, "w", encoding="utf-8"), ensure_ascii=False)


def wav_path(key):
    return os.path.join(DS, "wav", re.sub(r"[^A-Za-z0-9_]", "_", key) + ".wav")


def encode(shard):
    import pyarrow.parquet as pq, librosa
    from qwen_tts import Qwen3TTSTokenizer
    out = os.path.join(DS, "codes", shard.replace(".parquet", ".npz"))
    mine = {r["row"]: r for r in json.load(open(ITEMS, encoding="utf-8")) if r["shard"] == shard}
    if not mine:
        return
    tok = Qwen3TTSTokenizer.from_pretrained(os.path.join(SNAP, os.listdir(SNAP)[0], "speech_tokenizer"),
                                            device_map="cuda:0")
    codes, batch = {}, []

    def flush():
        enc = tok.encode([x for _, x in batch], sr=24000)
        for (key, _), c in zip(batch, enc.audio_codes):
            codes[key] = c.cpu().numpy().astype(np.int16)
        batch.clear()

    base = 0
    for rb in pq.ParquetFile(os.path.join(WORK, "ruls", shard)).iter_batches(batch_size=64, columns=["audio"]):
        col = rb.column(0)
        for i in range(rb.num_rows):
            r = mine.get(base + i)
            if r is None:
                continue
            x, sr = sf.read(io.BytesIO(col[i].as_py()["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            x = librosa.resample(x, orig_sr=sr, target_sr=24000)
            sf.write(wav_path(r["audio_filepath"]), x, 24000)
            batch.append((r["audio_filepath"], x))
            if len(batch) == 16:
                flush()
        base += rb.num_rows
    if batch:
        flush()
    np.savez(out + ".tmp.npz", **{k.replace("/", "|"): v for k, v in codes.items()})
    os.replace(out + ".tmp.npz", out)
    print(f"  {shard}: {len(codes)} фраз", flush=True)


def assemble():
    rng = random.Random(11)
    items = json.load(open(ITEMS, encoding="utf-8"))
    codes = {}
    for f in glob.glob(os.path.join(DS, "codes", "*.npz")):
        with np.load(f) as z:
            codes.update({k.replace("|", "/"): z[k] for k in z.files})
    acc = load_ruaccent()
    by_reader = {}
    for r in items:
        by_reader.setdefault(r["reader"], []).append(r)
    out = {s: open(os.path.join(DS, f"{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "val")}
    marked_words = total_words = 0
    for r in items:
        others = [o for o in by_reader[r["reader"]]
                  if o is not r and 4.0 <= o["duration"] <= 10.0 and o["audio_filepath"] in codes]
        if not others or r["audio_filepath"] not in codes:
            continue
        ref = rng.choice(others)
        text = mark(acc.process_all(r["text_no_preprocessing"]), rng)
        marked_words += text.count(ACUTE)
        total_words += len(re.findall(r"[А-Яа-яЁё]+", text))
        out[r["split"]].write(json.dumps({
            "text": text, "codes": codes[r["audio_filepath"]].tolist(),
            "ref_text": ref["text_no_preprocessing"], "ref_codes": codes[ref["audio_filepath"]].tolist(),
            "ref_audio": wav_path(ref["audio_filepath"]), "reader": r["reader"],
        }, ensure_ascii=False) + "\n")
    for f in out.values():
        f.close()
    print(f"размечено {marked_words} слов из {total_words}")


def main():
    os.makedirs(os.path.join(DS, "wav"), exist_ok=True)
    os.makedirs(os.path.join(DS, "codes"), exist_ok=True)
    if not os.path.exists(ITEMS):
        select()
    for sh in shards():
        name = os.path.basename(sh)
        if os.path.exists(os.path.join(DS, "codes", name.replace(".parquet", ".npz"))):
            continue
        subprocess.run([sys.executable, __file__, "encode", name], check=True)
    assemble()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "encode":
        encode(sys.argv[2])
    else:
        main()
