"""OpenCompass model ``DllmCFGOC``: the single-block decoder of this repo, scored generatively.

Multiple choice (GPQA / ARC-C / PIQA / MMLU) is scored the way Future_dLLM's eval_oc does it:
the OpenCompass templates, postprocessors and evaluators are used as shipped, and only the
model is ours. Prompt handling follows Future_dLLM's per-family wrappers (themselves
Sparse-dLLM's), because the two families disagree on it:

                Dream (DreamFutureOC)            LLaDA (LLaDAFutureOC)
    prompt      chat template, one user turn     raw string, no template
    decode      keeps specials, cut at first EOS skip_special_tokens=True
    stop words  dropped                          applied

A PromptList is joined into one string first, and prompts are left-truncated to
``max_seq_len``. The family and mask token come from the checkpoint config.

Generation is ``irrelevant_cfg.generate``: one block of ``max_out_len`` tokens in as many
steps, with or without the negative-template CFG. ``sampler="dream"`` uses Dream's sampler
(``alg`` / ``temperature`` / ``top_p``), ``sampler="llada"`` LLaDA's low_confidence remasking.
Anything left unset takes the family default (FAMILY_DEFAULTS): Dream entropy, temperature 0.2,
top_p 0.95, w 0.5; LLaDA low_confidence, temperature 0, w 1.0.
Every prompt is decoded from ``set_seed(seed)``, as in the lm-eval model, so a rerun
reproduces it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from opencompass.models.base import BaseModel

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Per-family decoding defaults, the same as eval/lm_eval_model.py.
FAMILY_DEFAULTS = {
    "dream": dict(sampler="dream", alg="entropy", temperature=0.2, top_p=0.95, w=0.5),
    "llada": dict(sampler="llada", alg="entropy", temperature=0.0, top_p=1.0, w=1.0),
}

# config name -> negative template (None = vanilla decoding); same names as eval/.
CONFIGS = {"vanilla": None, "neg_only_norel": "no_relevance",
           "neg_only_norel_notail": "no_relevance_notail"}


def _convert_base_messages(inputs):
    """PromptList -> plain string, roles dropped (as DreamFutureOC / Sparse-dLLM)."""
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            outputs.append(''.join(item['prompt'] for item in _input))
    return outputs


class DllmCFGOC(BaseModel):

    def __init__(
        self,
        path: str = "",
        config: str = "vanilla",
        w: Optional[float] = None,
        max_seq_len: int = 2048,
        sampler: Optional[str] = None,
        alg: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: int = 1234,
        meta_template: Optional[dict] = None,
    ):
        # mmengine parses the config lazily, so an empty path comes from the env.
        path = path or os.environ.get("MODEL_PATH", "")
        if not path:
            raise ValueError("set MODEL_PATH or pass path= in the config")
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}")
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template)

        from transformers import AutoModel, AutoTokenizer

        from irrelevant_cfg import MASK_ID

        self._cfg_name = config
        self._neg = CONFIGS[config]
        self._seed = int(seed)

        self.model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to("cuda").eval()
        self._dream = self.model.config.model_type.lower() == "dream"
        self._mask_id = int(self.model.config.mask_token_id) if self._dream else MASK_ID
        d = FAMILY_DEFAULTS["dream" if self._dream else "llada"]
        self._w = float(d["w"] if w is None else w)
        self._sampler = str(d["sampler"] if sampler is None else sampler)
        self._alg = str(d["alg"] if alg is None else alg)
        self._temperature = float(d["temperature"] if temperature is None else temperature)
        self._top_p = float(d["top_p"] if top_p is None else top_p)
        if self._sampler not in ("dream", "llada"):
            raise ValueError("sampler must be 'dream' or 'llada'")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default
        print(f"[DllmCFGOC] family={'dream' if self._dream else 'llada'} config={config} "
              f"neg={self._neg} w={self._w} sampler={self._sampler} alg={self._alg} "
              f"temperature={self._temperature} top_p={self._top_p} seed={self._seed} "
              f"max_seq_len={max_seq_len} mask_id={self._mask_id}", flush=True)

    def _encode(self, text: str, max_length: int):
        if self._dream:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                              add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 stopping_criteria: List[str] = []) -> List[str]:
        from irrelevant_cfg import generate, set_seed

        gen_length = int(max_out_len)
        eos = self.tokenizer.eos_token
        outputs = []
        for text in _convert_base_messages(inputs):
            set_seed(self._seed)
            ids = self._encode(text, self.max_seq_len)["input_ids"].to("cuda")
            out = generate(self.model, self.tokenizer, ids, gen_length=gen_length,
                           neg_template=self._neg, guidance_scale=self._w,
                           temperature=self._temperature, sampler=self._sampler,
                           top_p=self._top_p, alg=self._alg,
                           mask_id=self._mask_id, dream=self._dream)
            tokens = out[0, ids.shape[1]:].tolist()
            if self._dream:
                text_out = self.tokenizer.decode(tokens)
                text_out = text_out.split(eos)[0] if eos else text_out
            else:
                text_out = self.tokenizer.decode(tokens, skip_special_tokens=True)
                for stop in stopping_criteria:
                    text_out = text_out.split(stop)[0]
            outputs.append(text_out)
        return outputs


DreamCFGOC = DllmCFGOC   # the earlier name, kept for eval_dream_mc_cfg.py
