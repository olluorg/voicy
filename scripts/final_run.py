"""Winning config on the full prepared samples: RL-V2 + best human reference."""
import json, time, sys, pathlib, warnings, itertools
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, torch, torchaudio
def _load_wav(p,*a,**k):
    d,sr=sf.read(str(p),dtype="float32",always_2d=True); return torch.from_numpy(d.T),sr
torchaudio.load=_load_wav
from onnxruntime import InferenceSession as _I
_o=_I.run
def _r(self,n,f,*a,**k):
    m={i.name for i in self.get_inputs()}-set(f)
    if m and "input_ids" in f: f={**f,**{x:np.zeros_like(f["input_ids"]) for x in m}}
    return _o(self,n,f,*a,**k)
_I.run=_r
from huggingface_hub import hf_hub_download
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

CFG=dict(dim=1024,depth=22,heads=16,ff_mult=2,text_dim=512,conv_layers=4)
REF="lv/ref_ys.wav"
OUT=pathlib.Path("final"); OUT.mkdir(exist_ok=True)
acc=RUAccent(); acc.load(omograph_model_size="turbo3.1",use_dictionary=True,tiny_mode=False)
voc=load_vocoder().to("cuda")
model=load_model(DiT,CFG,hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2",filename="espeech_tts_rlv2.pt"),
                 vocab_file=hf_hub_download(repo_id="ESpeech/ESpeech-TTS-1_RL-V2",filename="vocab.txt")).to("cuda")
_rt=json.load(open("lv/ref_texts.json",encoding="utf-8"))
ref_txt=next(v for k,v in _rt.items() if "ref_ys" in k)
ref_a,ref_t=preprocess_ref_audio_text(REF,acc.process_all(ref_txt))
res=[]
for s,(nfe,speed) in itertools.product(json.load(open("samples.json",encoding="utf-8")),((48,1.0),(48,0.9))):
    torch.manual_seed(1234); t0=time.perf_counter()
    w,sr,_=infer_process(ref_a,ref_t,acc.process_all(s["audio"]),model,voc,
                         cross_fade_duration=0.15,nfe_step=nfe,speed=speed)
    wall=time.perf_counter()-t0
    name=f"{s['id']}__final__rlv2-ys-sp{speed}.wav"
    sf.write(str(OUT/name),w,sr); d=len(w)/sr
    res.append({"id":s["id"],"speed":speed,"nfe":nfe,"file":name,"audio_s":round(d,2),
                "wall_s":round(wall,2),"rtf":round(d/wall,2)})
    print(f"{name:44} {d:6.1f}s / {wall:5.1f}s  ×{d/wall:.1f}",flush=True)
json.dump(res,open("results_final.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
