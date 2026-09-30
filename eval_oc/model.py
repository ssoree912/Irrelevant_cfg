"""OpenCompass model ``DreamCFGOC``: the single-block decoder of this repo, scored generatively.

Multiple choice (GPQA / ARC-C / PIQA) is scored the way Future_dLLM's eval_oc does it: the
OpenCompass templates, postprocessors and evaluators are used as shipped, and only the model
is ours. Prompt handling follows Future_dLLM's ``DreamFutureOC`` (itself Sparse-dLLM's
Dream-Instruct wrapper): a PromptList is joined into one string, put in a single user turn
of the chat template, left-truncated to ``max_seq_len``, and the output is decoded with
special tokens kept and cut at the first EOS.

Generation is ``irrelevant_cfg.generate``: one block of ``max_out_len`` tokens in as many
steps, Dream's sampler (``alg`` / ``temperature`` / ``top_p``), with or without the
negative-template CFG. Every prompt is decoded from ``set_seed(seed)``, as in the lm-eval
model, so a rerun reproduces it.
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

# config name -> negative template (None = vanilla decoding); same names as eval/.
CONFIGS = {"vanilla": None, "neg_only_norel": "no_relevance"}


def _convert_base_messages(inputs):
    """PromptList -> plain string, roles dropped (as DreamFutureOC / Sparse-dLLM)."""
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            outputs.append(''.join(item['prompt'] for item in _input))
    return outputs


class DreamCFGOC(BaseModel):

    def __init__(
        self,
        path: str = "",
        config: str = "vanilla",
        w: float = 1.0,
        max_seq_len: int = 2048,
        alg: str = "entropy",
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int = 1234,
        meta_template: Optional[dict] = None,
    ):
        # mmengine parses the config lazily, so the path arrives empty and comes from the env.
        path = path or os.environ.get("MODEL_PATH", "")
        if not path:
            raise ValueError("set MODEL_PATH or pass path= in the config")
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}")
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template)

        from transformers import AutoModel, AutoTokenizer

        self._cfg_name = config
        self._neg = CONFIGS[config]
        self._w = float(w)
        self._alg = str(alg)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._seed = int(seed)

        self.model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to("cuda").eval()
        if self.model.config.model_type.lower() != "dream":
            raise ValueError("DreamCFGOC expects a Dream checkpoint")
        self._mask_id = int(self.model.config.mask_token_id)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default
        print(f"[DreamCFGOC] config={config} neg={self._neg} w={self._w} alg={self._alg} "
              f"temperature={self._temperature} top_p={self._top_p} seed={self._seed} "
              f"max_seq_len={max_seq_len} mask_id={self._mask_id}", flush=True)

    def _encode(self, text: str, max_length: int):
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return self.tokenizer(chat, return_tensors="pt", truncation=True,
                              add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        from irrelevant_cfg import generate, set_seed

        gen_length = int(max_out_len)
        eos = self.tokenizer.eos_token
        outputs = []
        for text in _convert_base_messages(inputs):
            set_seed(self._seed)
            ids = self._encode(text, self.max_seq_len)["input_ids"].to("cuda")
            out = generate(self.model, self.tokenizer, ids, gen_length=gen_length,
                           neg_template=self._neg, guidance_scale=self._w,
                           temperature=self._temperature, sampler="dream",
                           top_p=self._top_p, alg=self._alg,
                           mask_id=self._mask_id, dream=True)
            text_out = self.tokenizer.decode(out[0, ids.shape[1]:].tolist())
            outputs.append(text_out.split(eos)[0] if eos else text_out)
        return outputs
