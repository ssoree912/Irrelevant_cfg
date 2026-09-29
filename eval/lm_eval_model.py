"""lm-eval model ``LLaDA_cfg``: Future_dLLM-style harness, single-block cache-free decoding.

Works for LLaDA and Dream; the family and mask token come from the checkpoint's config.

Request handling mirrors Future_dLLM's eval/lm_eval_model.FutureDLLM.generate_until
(add_bos, left truncation, decode with specials skipped, cut at every ``until``), so task
yamls, prompts, chat-template flags and scoring stay the harness's own. Generation is
always one block of ``gen_length`` (default 128) in ``gen_length`` steps, ignoring the
yaml's max_gen_toks.

    python eval/lm_eval_model.py --model LLaDA_cfg \
        --model_args "pretrained=<dir>,config=vanilla|neg_only_norel" ...

LLADA_CFG_SHARD="i/n" makes this process decode only requests with index % n == i
(the rest return "" and are not stored) into LLADA_CFG_STORE.shard<i>, so n processes
can run in parallel; a final unsharded run then replays every shard file and scores.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irrelevant_cfg import MASK_ID, generate, set_seed  # noqa: E402

# config name -> negative template (None = vanilla decoding)
CONFIGS = {"vanilla": None, "neg_only_norel": "no_relevance"}


@register_model("LLaDA_cfg")
class LLaDACFG(HFLM):
    def __init__(self, pretrained, config="vanilla", gen_length=128, max_seq_len=4096,
                 w=1.0, seed=1234, **kwargs):
        from transformers import AutoModel
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}")
        self._neg = CONFIGS[config]
        self._cfg_name = config
        self._gen_length = int(gen_length)
        self._max_seq_len = int(max_seq_len)
        self._w = float(w)
        self._seed = int(seed)
        model = AutoModel.from_pretrained(str(pretrained), trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).to("cuda").eval()
        self._dream = model.config.model_type.lower() == "dream"
        self._mask_id = int(model.config.mask_token_id) if self._dream else MASK_ID
        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(pretrained=model, **kwargs)
        print(f"[LLaDA_cfg] config={config} neg={self._neg} w={self._w} "
              f"gen_length=steps=block_length={self._gen_length} cache=off "
              f"family={'dream' if self._dream else 'llada'} mask_id={self._mask_id}", flush=True)

    def loglikelihood(self, requests, disable_tqdm=False):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm=False):
        from tqdm import tqdm
        # One fsynced line per answer (LLADA_CFG_STORE), replayed on restart.
        # Shards write <store>.shard<i> so parallel appends never share a file; every
        # process replays the main store plus all shard files.
        store_path = os.environ.get("LLADA_CFG_STORE", "")
        shard, n_shards = map(int, os.environ.get("LLADA_CFG_SHARD", "0/1").split("/"))
        done = {}
        if store_path:
            paths = [store_path] + sorted(glob.glob(glob.escape(store_path) + ".shard*"))
            for path in (p for p in paths if os.path.exists(p)):
                with open(path) as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        done[rec["key"]] = rec["text"]
            if n_shards > 1:
                store_path = f"{store_path}.shard{shard}"
        store = open(store_path, "a") if store_path else None
        results = []
        for index, request in enumerate(tqdm(requests, disable=disable_tqdm or self.rank != 0,
                            desc=f"LLaDA_cfg {self._cfg_name} shard {shard}/{n_shards}")):
            context, raw_kwargs = request.args
            request_body = context + repr(sorted(raw_kwargs.items()))
            key = hashlib.md5((request_body + self._cfg_name + str(self._gen_length)).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                continue
            if index % n_shards != shard:
                results.append("")
                continue
            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=self._max_seq_len - self._gen_length)
            prompt = context_enc.to(self.device)
            set_seed(self._seed)
            stats = {}
            started = time.perf_counter()
            out = generate(self.model, self.tokenizer, prompt, gen_length=self._gen_length,
                           neg_template=self._neg, guidance_scale=self._w, temperature=0.0,
                           guidance_scope="all_positive", stats=stats,
                           mask_id=self._mask_id, dream=self._dream)
            seconds = time.perf_counter() - started
            tokens = out[0, prompt.shape[1]:].tolist()
            assert len(tokens) == self._gen_length and self._mask_id not in tokens
            raw_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text = raw_text
            for term in raw_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            results.append(text)
            if store is not None:
                store.write(json.dumps({"key": key, "doc_id": request.doc_id, "text": text,
                                        "raw": raw_text, "tokens": tokens, "seconds": seconds,
                                        "stats": stats, "prompt_tokens": prompt.shape[1]}) + "\n")
                store.flush()
                os.fsync(store.fileno())
        if store is not None:
            store.close()
        return results


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
