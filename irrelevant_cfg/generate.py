"""Single-block, cache-free LLaDA / Dream decoding with negative-template CFG.

The response is one block of ``gen_length`` tokens decoded in ``gen_length`` steps, so
exactly one token is committed per step. Every step runs a full forward over the whole
sequence (no KV / prefix / future cache).

With a negative template, two branches share one decoding state and differ only at the
negative's anchor positions. Each step evaluates both and combines

    L_guided = (1 + w) * L_pos - w * L_neg

in fp32; the guided logits pick both the token and the confidence that orders unmasking.
Committed tokens are written into both branches, so the trajectory stays shared.

Dream (``dream=True``) differs only in the forward: full bidirectional attention is
requested with ``attention_mask="full"`` and its logits are shifted right by one, since
Dream's position i predicts token i+1.

sampler="dream" swaps LLaDA's Gumbel-argmax + low-confidence pick for Dream's own
sample_tokens (temperature, then top-p, then Categorical; ``alg`` "entropy" ranks by
negative entropy, "maskgit_plus" by the sampled token's probability). The schedule is
unchanged: one token committed per step, except for ``alg="origin"``, which follows Dream's
_sample exactly: each step every masked position is committed independently with
probability 1 - s/t on timesteps linspace(1, eps, steps + 1), the last step commits all.
"""

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F

from .sampling import add_gumbel_noise, get_num_transfer_tokens
from .templates import MASK_ID, build_branches


def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep the first token above the threshold.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices,
                                                               sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def dream_sample_tokens(logits, temperature=0.0, top_p=None, alg="entropy"):
    """(confidence, token) per row, as Dream's generation_utils.sample_tokens.

    "origin" ranks nothing, so it returns maskgit_plus's confidence."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    if alg == "entropy":
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    elif alg not in ("maskgit_plus", "origin"):
        raise ValueError(f"unknown dream alg {alg!r}")
    return confidence, x0


def _logits(model, x, dream):
    if not dream:
        return model(x).logits
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    gen_length=128,
    neg_template=None,
    guidance_scale=1.0,
    pos_template=None,
    temperature=0.0,
    remasking="low_confidence",
    guidance_scope="all_positive",
    stats=None,
    mask_id=MASK_ID,
    dream=False,
    sampler="llada",
    top_p=None,
    alg="entropy",
    eps=1e-3,
):
    """Decode ``gen_length`` tokens after ``prompt`` (shape [1, prompt_len]).

    neg_template None or guidance_scale 0 is plain LLaDA decoding (one forward per step).

    guidance_scope: "all_positive" guides every response position; "shared_free" keeps
    the positive logits at the negative's anchor positions.

    stats (optional dict) receives the mean EOS probability over masked positions per
    branch ('eos_pos', 'eos_neg'), averaged over steps, to catch a branch collapsing to EOS.

    mask_id / dream: the model's mask token and whether to use Dream's shifted forward.

    sampler: "llada" (Gumbel-argmax, ``remasking`` confidence) or "dream"
    (dream_sample_tokens with ``temperature``, ``top_p``, ``alg``) on the guided logits.
    """
    if sampler not in ("llada", "dream"):
        raise ValueError(f"unknown sampler {sampler!r}")
    if guidance_scope not in ("all_positive", "shared_free"):
        raise ValueError(f"unknown guidance_scope {guidance_scope!r}")
    device = model.device
    use_cfg = guidance_scale != 0.0 and neg_template is not None

    x, x_neg, pos_spans, neg_spans = build_branches(
        tokenizer, prompt, gen_length, pos_template, neg_template if use_cfg else None, device,
        mask_id=mask_id)
    prompt_len = prompt.shape[1]

    # The negative's anchors must survive commits: without a positive template those
    # positions are free masks in x, and copying x's commits over them would make the
    # branches converge and L_pos - L_neg silently go to zero.
    neg_protect = None
    guidance_positions = torch.ones_like(x, dtype=torch.bool)
    guidance_positions[:, :prompt_len] = False
    if use_cfg:
        neg_protect = torch.zeros(x.shape, dtype=torch.bool, device=device)
        for start, end, _ in neg_spans:
            neg_protect[:, start:end] = True
        if guidance_scope == "shared_free":
            guidance_positions &= ~neg_protect

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    eos_pos_acc, eos_neg_acc, n_stat_steps = 0.0, 0.0, 0

    # Single block covering the whole response; one step per token.
    steps = gen_length
    block_end = x.shape[1]
    num_transfer_tokens = get_num_transfer_tokens(x[:, prompt_len:block_end] == mask_id, steps)
    origin = sampler == "dream" and alg == "origin"
    timesteps = torch.linspace(1, eps, steps + 1, device=device) if origin else None

    for step in range(steps):
        mask_index = (x == mask_id)
        if mask_index.sum() == 0:
            break

        if use_cfg:
            # Sequential batch-1 forwards: batch-2 packing is not bit-identical on cuBLAS.
            logits_pos = _logits(model, x, dream)
            logits_neg = _logits(model, x_neg, dream)
            w = guidance_scale
            logits = ((1.0 + w) * logits_pos.float() - w * logits_neg.float()).to(logits_pos.dtype)
            if guidance_scope == "shared_free":
                logits = torch.where(guidance_positions.unsqueeze(-1), logits, logits_pos)
            if stats is not None:
                p_pos = F.softmax(logits_pos.float(), dim=-1)[0, :, eos_id]
                p_neg = F.softmax(logits_neg.float(), dim=-1)[0, :, eos_id]
                m = mask_index[0] & guidance_positions[0]
                if m.any():
                    eos_pos_acc += p_pos[m].mean().item()
                    eos_neg_acc += p_neg[m].mean().item()
                    n_stat_steps += 1
        else:
            logits = _logits(model, x, dream)
            if stats is not None:
                p_pos = F.softmax(logits.float(), dim=-1)[0, :, eos_id]
                m = mask_index[0]
                if m.any():
                    eos_pos_acc += p_pos[m].mean().item()
                    n_stat_steps += 1

        if origin:
            t, s = timesteps[step], timesteps[step + 1]
            p_transfer = 1 - s / t if step < steps - 1 else 1
            mask_logits = logits[mask_index].float()
            picked = torch.rand(mask_logits.shape[0], device=device) < p_transfer
            x0 = x.clone()
            if picked.any():
                _, x0_picked = dream_sample_tokens(mask_logits[picked], temperature=temperature,
                                                   top_p=top_p, alg="maskgit_plus")
                x0_m = x0[mask_index]
                x0_m[picked] = x0_picked
                x0[mask_index] = x0_m
            transfer_index = torch.zeros_like(mask_index)
            transfer_index[mask_index] = picked
        elif sampler == "dream":
            conf_m, x0_m = dream_sample_tokens(logits[mask_index].float(), temperature=temperature,
                                               top_p=top_p, alg=alg)
            x0 = x.clone()
            x0[mask_index] = x0_m
            confidence = torch.full(x.shape, -np.inf, dtype=conf_m.dtype, device=x.device)
            confidence[mask_index] = conf_m
        else:
            x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

        if not origin:
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = min(num_transfer_tokens[j, step].item(), mask_index[j, :block_end].sum().item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j, :block_end], k=k)
                    transfer_index[j, select_index] = True

        x[transfer_index] = x0[transfer_index]
        if use_cfg:
            neg_transfer = transfer_index & ~neg_protect
            x_neg[neg_transfer] = x0[neg_transfer]

    if use_cfg:
        # Invariant: the branches differ only inside anchor spans.
        allowed = [(s, e) for s, e, _ in pos_spans] + [(s, e) for s, e, _ in neg_spans]
        for _, pos in (x != x_neg).nonzero().tolist():
            if not any(s <= pos < e for s, e in allowed):
                raise RuntimeError(f"branches diverged outside anchor spans at position {pos}")

    if stats is not None:
        if n_stat_steps:
            stats["eos_pos"] = eos_pos_acc / n_stat_steps
            stats["eos_neg"] = (eos_neg_acc / n_stat_steps) if use_cfg else None
            stats["n_steps"] = n_stat_steps
        stats["guidance_scope"] = guidance_scope

    return x
