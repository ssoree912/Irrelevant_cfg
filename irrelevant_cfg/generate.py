"""Single-block, cache-free LLaDA decoding with negative-template CFG.

The response is one block of ``gen_length`` tokens decoded in ``gen_length`` steps, so
exactly one token is committed per step. Every step runs a full forward over the whole
sequence (no KV / prefix / future cache).

With a negative template, two branches share one decoding state and differ only at the
negative's anchor positions. Each step evaluates both and combines

    L_guided = (1 + w) * L_pos - w * L_neg

in fp32; the guided logits pick both the token and the confidence that orders unmasking.
Committed tokens are written into both branches, so the trajectory stays shared.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .sampling import add_gumbel_noise, get_num_transfer_tokens
from .templates import MASK_ID, build_branches


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
):
    """Decode ``gen_length`` tokens after ``prompt`` (shape [1, prompt_len]).

    neg_template None or guidance_scale 0 is plain LLaDA decoding (one forward per step).

    guidance_scope: "all_positive" guides every response position; "shared_free" keeps
    the positive logits at the negative's anchor positions.

    stats (optional dict) receives the mean EOS probability over masked positions per
    branch ('eos_pos', 'eos_neg'), averaged over steps, to catch a branch collapsing to EOS.
    """
    if guidance_scope not in ("all_positive", "shared_free"):
        raise ValueError(f"unknown guidance_scope {guidance_scope!r}")
    device = model.device
    use_cfg = guidance_scale != 0.0 and neg_template is not None

    x, x_neg, pos_spans, neg_spans = build_branches(
        tokenizer, prompt, gen_length, pos_template, neg_template if use_cfg else None, device)
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
    num_transfer_tokens = get_num_transfer_tokens(x[:, prompt_len:block_end] == MASK_ID, steps)

    for step in range(steps):
        mask_index = (x == MASK_ID)
        if mask_index.sum() == 0:
            break

        if use_cfg:
            # Sequential batch-1 forwards: batch-2 packing is not bit-identical on cuBLAS.
            logits_pos = model(x).logits
            logits_neg = model(x_neg).logits
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
            logits = model(x).logits
            if stats is not None:
                p_pos = F.softmax(logits.float(), dim=-1)[0, :, eos_id]
                m = mask_index[0]
                if m.any():
                    eos_pos_acc += p_pos[m].mean().item()
                    n_stat_steps += 1

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
