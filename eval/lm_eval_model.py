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
import random
import re
import sys
import time
from pathlib import Path

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irrelevant_cfg import MASK_ID, generate, set_seed  # noqa: E402
from irrelevant_cfg.templates import HEAD_SENTENCES  # noqa: E402

# Per-family decoding defaults, used for any of sampler / alg / temperature / top_p / w
# not given in model_args.
FAMILY_DEFAULTS = {
    "dream": dict(sampler="dream", alg="entropy", temperature=0.2, top_p=0.95, w=0.5),
    "llada": dict(sampler="llada", alg="entropy", temperature=0.0, top_p=1.0, w=1.0),
}

# config name -> negative template (None = vanilla decoding)
CONFIGS = {"vanilla": None, "neg_only_norel": "no_relevance",
           "neg_only_norel_notail": "no_relevance_notail",
           "neg_only_norel_notail3": "no_relevance_notail3",
           "neg_only_norel_head": "no_relevance_head",
           **{f"neg_{name}": name for name in HEAD_SENTENCES}, "id_opposite": None,
           "neg_numswap": None, "neg_recipe": "recipe", "neg_gibberish": "gibberish",
           "neg_repeat": "repeat", "neg_blank": "blank", "neg_refusal": "refusal",
           "neg_lazy": "lazy", "neg_story_fairy": "story_fairy",
           "neg_story_mystery": "story_mystery", "neg_story_scifi": "story_scifi",
           "neg_story_paraphrase": "story_paraphrase"}

OPPOSITE_PREFIX = "Always respond with the opposite of what you're asked. You never get it right.\n\n"


def swap_numbers(text):
    """Counterfactual prompt: every digit run replaced by a different one of the same length
    (no new leading zero), seeded by the text so the negative is reproducible."""
    rng = random.Random(hashlib.md5(text.encode()).hexdigest())

    def repl(m):
        old = m.group(0)
        while True:
            new = "".join(rng.choice("123456789" if i == 0 and len(old) > 1 else "0123456789")
                          for i in range(len(old)))
            if new != old:
                return new

    return re.sub(r"\d+", repl, text)


# config -> (negative-branch prompt from the context, combine formula). id_opposite is
# Instructive Decoding (arXiv:2311.00233): z - w * z~ with w as its epsilon (default 0.3).
# neg_numswap is the same problem with its numbers changed, under the usual CFG formula.
NEG_PROMPTS = {
    "id_opposite": (lambda context: OPPOSITE_PREFIX + context, "id"),
    "neg_numswap": (swap_numbers, "cfg"),
}
ID_DEFAULT_EPS = 0.3


@register_model("LLaDA_cfg")
class LLaDACFG(HFLM):
    def __init__(self, pretrained, config="vanilla", gen_length=128, max_seq_len=4096,
                 w=None, seed=1234, sampler=None, temperature=None, top_p=None, alg=None,
                 postprocess="default", **kwargs):
        from transformers import AutoModel
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}")
        self._neg = CONFIGS[config]
        self._neg_prompt_fn, self._combine = NEG_PROMPTS.get(config, (None, "cfg"))
        self._cfg_name = config
        self._gen_length = int(gen_length)
        self._max_seq_len = int(max_seq_len)
        self._seed = int(seed)
        model = AutoModel.from_pretrained(str(pretrained), trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).to("cuda").eval()
        self._dream = model.config.model_type.lower() == "dream"
        self._mask_id = int(model.config.mask_token_id) if self._dream else MASK_ID
        d = FAMILY_DEFAULTS["dream" if self._dream else "llada"]
        default_w = ID_DEFAULT_EPS if self._combine == "id" else d["w"]
        self._w = float(default_w if w is None else w)
        self._sampler = str(d["sampler"] if sampler is None else sampler)
        self._temperature = float(d["temperature"] if temperature is None else temperature)
        self._top_p = float(d["top_p"] if top_p is None else top_p)
        self._alg = str(d["alg"] if alg is None else alg)
        if postprocess not in ("default", "ti"):
            raise ValueError("postprocess must be 'default' or 'ti'")
        self._postprocess = postprocess
        # Resume key suffix; the LLaDA sampler keeps the original (empty) suffix.
        self._decode_tag = ("" if self._sampler == "llada" else
                            f"{self._sampler}:{self._alg}:{self._temperature}:{self._top_p}")
        if self._postprocess != "default":
            self._decode_tag += f":post-{self._postprocess}"
        if (self._neg is not None or self._neg_prompt_fn is not None) and self._w != 1.0:
            self._decode_tag += f":w{self._w}"
        if self._max_seq_len != 4096:
            self._decode_tag += f":msl{self._max_seq_len}"
        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(pretrained=model, **kwargs)
        print(f"[LLaDA_cfg] config={config} neg={self._neg} "
              f"neg_prompt={config if self._neg_prompt_fn else None} combine={self._combine} "
              f"w={self._w} "
              f"gen_length=steps=block_length={self._gen_length} cache=off "
              f"family={'dream' if self._dream else 'llada'} mask_id={self._mask_id} "
              f"sampler={self._sampler} alg={self._alg} temperature={self._temperature} "
              f"top_p={self._top_p} seed={self._seed} postprocess={self._postprocess}", flush=True)

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
            key = hashlib.md5((request_body + self._cfg_name + str(self._gen_length)
                                  + self._decode_tag).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                continue
            if index % n_shards != shard:
                results.append("")
                continue
            neg_context = None if self._neg_prompt_fn is None else self._neg_prompt_fn(context)
            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
                if neg_context is not None:
                    neg_context = self.tokenizer.bos_token + neg_context
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=self._max_seq_len - self._gen_length)
            prompt = context_enc.to(self.device)
            neg_prompt = None
            if neg_context is not None:
                neg_enc, _ = self.tok_batch_encode(
                    [neg_context], truncation=self.truncation,
                    left_truncate_len=self._max_seq_len - self._gen_length)
                neg_prompt = neg_enc.to(self.device)
            set_seed(self._seed)
            stats = {}
            if neg_prompt is not None:
                stats["neg_prompt_differs"] = not torch.equal(neg_prompt, prompt)
            started = time.perf_counter()
            out = generate(self.model, self.tokenizer, prompt, gen_length=self._gen_length,
                           neg_template=self._neg, guidance_scale=self._w,
                           temperature=self._temperature if self._sampler == "dream" else 0.0,
                           guidance_scope="all_positive", stats=stats,
                           mask_id=self._mask_id, dream=self._dream, sampler=self._sampler,
                           top_p=self._top_p, alg=self._alg, neg_prompt=neg_prompt,
                           combine=self._combine)
            seconds = time.perf_counter() - started
            tokens = out[0, prompt.shape[1]:].tolist()
            assert len(tokens) == self._gen_length and self._mask_id not in tokens
            raw_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if self._postprocess == "ti":
                # Template-Infilling's ours_dream_verify: cut at the first EOS, apply the stop
                # strings with special tokens still in the text, then drop them.
                eos = self.tokenizer.eos_token_id
                kept = tokens[:tokens.index(eos)] if eos in tokens else tokens
                text = self.tokenizer.decode(kept, skip_special_tokens=False)
                for term in raw_kwargs.get("until") or []:
                    if term and term in text:
                        text = text.split(term)[0]
                text = self.tokenizer.decode(
                    self.tokenizer(text, add_special_tokens=False)["input_ids"],
                    skip_special_tokens=True)
            else:
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
