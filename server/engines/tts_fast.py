"""The code predictor without `generate()`, replayed from CUDA graphs.

Qwen3-TTS makes one audio frame (1/12.6 s) in two stages: the talker, a 28-layer
model, predicts the first codebook, and the code predictor, a 5-layer one,
predicts the other fifteen, one after another. The model calls the predictor
through transformers' `generate()` — a full generation loop, with its setup,
logits processors and a fresh cache — 170 times for a 13.5-second phrase.

Profiling showed where the time goes: the card sits at 36% while one CPU core
is saturated. Each of the fifteen predictor steps is a handful of tiny kernels,
and launching them from Python costs more than running them.

So the fifteen steps are done here by a plain loop with a static cache, and each
step's forward pass is captured once into a CUDA graph and replayed: one launch
instead of a few hundred. On an RTX 3080 a frame drops from 140 to 51–57 ms:
synthesis goes from ×0.55 to ×1.4–1.5 of real time.

Sampling is the same as in `generate()` — temperature, then top-k, then a
multinomial draw. The tokens are not bit-for-bit the same: the static cache
attends over sixteen slots with a mask, the original over a growing cache, and
the two use different attention kernels. Logits differ by one or two bf16 units,
which flips a token where two candidates are nearly tied — 70% of frames came
out identical. Intelligibility does not change: 30 phrases read back by Whisper
gave 2.1% CER both ways, all of it one phrase where "восемьдесят" was written
as "80" (scripts/tts_fast_eval.py).

Only the configuration the model actually uses is handled: batch of one,
sampling with top_p = 1. Anything else falls through to the original `generate`.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


class FastCodePredictor:
    def __init__(self, cp, use_graphs: bool = True):
        from transformers.cache_utils import StaticCache

        self.cp = cp
        self.original = cp.generate
        self.steps = cp.config.num_code_groups - 1              # 15
        param = next(cp.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.use_graphs = use_graphs and self.device.type == "cuda"
        max_len = self.steps + 1                                 # 2 на входе + 14 шагов
        self.cache = StaticCache(config=cp.config, max_cache_len=max_len)

        talker_hidden = cp.small_to_mtp_projection.in_features
        self.embeds_in = torch.zeros(1, 2, talker_hidden, device=self.device, dtype=self.dtype)
        self.ids_in = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # позиции и маски для каждого шага заранее: при записи графа
        # transformers не должен ничего вычислять на процессоре
        kv = torch.arange(max_len, device=self.device)
        self.positions, self.masks = [], []
        for step in range(self.steps):
            pos = (torch.arange(0, 2, device=self.device) if step == 0
                   else torch.tensor([step + 1], device=self.device))
            self.positions.append(pos)
            mask = (kv[None, :] <= pos[:, None])[None, None]    # (1, 1, q, max_len)
            self.masks.append({"full_attention": mask})
        self.logits: list[torch.Tensor | None] = [None] * self.steps
        self.graphs: list[torch.cuda.CUDAGraph] = []

    # ------------------------------------------------------------------ проход

    @torch.no_grad()
    def _forward(self, step: int) -> torch.Tensor:
        cp = self.cp
        if step == 0:
            emb = self.embeds_in
        else:
            emb = cp.model.get_input_embeddings()[step - 1](self.ids_in)
        emb = cp.small_to_mtp_projection(emb)
        out = cp.model(inputs_embeds=emb, attention_mask=self.masks[step],
                       position_ids=self.positions[step][None],
                       past_key_values=self.cache, use_cache=True,
                       cache_position=self.positions[step])
        # голова по всем позициям, как в generate(): иначе другой размер матрицы
        # может выбрать другое ядро и разойтись в последнем знаке bf16
        return cp.lm_head[step](out.last_hidden_state)[:, -1, :]

    def capture(self) -> None:
        """Record the fifteen graphs. Safe to call more than once."""
        if not self.use_graphs or self.graphs:
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):                                   # прогрев: ленивые буферы кэша
                for step in range(self.steps):
                    self._forward(step)
        torch.cuda.current_stream().wait_stream(stream)
        pool = None
        for step in range(self.steps):
            g = torch.cuda.CUDAGraph()
            # thread_local: распознаватель в другом потоке может в это время
            # работать на той же карте, и глобальный режим записи сломал бы ему вызов
            with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
                self.logits[step] = self._forward(step)
            pool = g.pool()
            self.graphs.append(g)

    # --------------------------------------------------------------- генерация

    @staticmethod
    def _sample(logits: torch.Tensor, do_sample: bool, top_k: int | None,
                temperature: float | None) -> torch.Tensor:
        scores = logits.to(torch.float32)
        if not do_sample:
            return scores.argmax(dim=-1, keepdim=True)
        if temperature is not None and temperature != 1.0:
            scores = scores / temperature
        if top_k:
            k = min(top_k, scores.size(-1))
            kth = torch.topk(scores, k)[0][..., -1, None]
            scores = scores.masked_fill(scores < kth, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate(self, inputs_embeds=None, max_new_tokens=None, do_sample=None, top_p=None,
                 top_k=None, temperature=None, **kwargs):
        usual = (inputs_embeds is not None and inputs_embeds.shape[0] == 1
                 and inputs_embeds.shape[1] == 2 and max_new_tokens == self.steps
                 and (top_p is None or top_p >= 1.0))
        if not usual:
            return self.original(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, top_p=top_p, top_k=top_k,
                                 temperature=temperature, **kwargs)
        self.capture()
        self.embeds_in.copy_(inputs_embeds)
        tokens = []
        for step in range(self.steps):
            if self.graphs:
                self.graphs[step].replay()
                logits = self.logits[step]
            else:
                logits = self._forward(step)
            tok = self._sample(logits, bool(do_sample), top_k, temperature)
            tokens.append(tok)
            if step + 1 < self.steps:
                self.ids_in.copy_(tok)
        return SimpleNamespace(sequences=torch.cat(tokens, dim=1))


def install(model, use_graphs: bool = True) -> FastCodePredictor:
    """Route the talker's calls to `code_predictor.generate` through the fast path."""
    inner = getattr(model, "model", model)
    cp = inner.talker.code_predictor
    fast = FastCodePredictor(cp, use_graphs=use_graphs)
    cp.generate = fast.generate
    return fast
