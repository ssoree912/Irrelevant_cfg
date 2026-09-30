"""Anchor templates and the two-branch initial state for negative-template CFG.

A template is {offset in the response: text}. Its anchors are written into the masked
response before decoding and stay fixed. The positive branch of negative-only CFG has no
anchors; the negative branch carries an irrelevant-structure template.
"""

import torch

MASK_ID = 126336

# Answer anchor shared by the negatives below, byte-identical to the good template's.
# An answer-shaped anchor at offset 0 collapses LLaDA to all-EOS, so the negatives vary
# only the two reasoning-structure anchors and keep this one pinned.
PINNED_TAIL = "\nTherefore, the answer is:"


def _good_parts(gen_length):
    # Template Infilling's math_template1 (arXiv:2510.13870).
    return {
        0: "Let me work through this problem\n",
        gen_length // 4: "\nProceeding to the next logical step:\n",
        gen_length - 20: PINNED_TAIL,
    }


def _no_relevance_parts(gen_length):
    # Coherent structure from a different task (narrative), applied to a math question.
    return {
        0: "Let me introduce the main characters\n",
        gen_length // 4: "\nMoving on to the setting of the story:\n",
        gen_length - 20: PINNED_TAIL,
    }


def _no_relevance_notail_parts(gen_length):
    # no_relevance without the pinned answer anchor: nothing in the negative sits where the
    # final answer is written, so the guidance there does not push against answering.
    return {
        0: "Let me introduce the main characters\n",
        gen_length // 4: "\nMoving on to the setting of the story:\n",
    }


def _no_relevance_code_parts(gen_length):
    # Same manipulation with a code structure instead of a narrative one.
    return {
        0: "Let me import the required modules\n",
        gen_length // 4: "\nProceeding to initialize the object state:\n",
        gen_length - 20: PINNED_TAIL,
    }


def _no_coherence_parts(gen_length):
    # On-task roles in inverted dependency order: the chain starts at the conclusion.
    return {
        0: "Beginning from the conclusion already reached\n",
        gen_length // 4: "\nGoing back to the step that came before:\n",
        gen_length - 20: PINNED_TAIL,
    }


def _neutral_parts(gen_length):
    # Structure without reasoning semantics.
    return {
        0: "Here is the very first section\n",
        gen_length // 4: "\nHere is the next section of the content:\n",
        gen_length - 20: PINNED_TAIL,
    }


TEMPLATES = {
    "math_template1": _good_parts,
    "no_relevance": _no_relevance_parts,
    "no_relevance_notail": _no_relevance_notail_parts,
    "no_relevance_code": _no_relevance_code_parts,
    "no_coherence": _no_coherence_parts,
    "neutral_pinned": _neutral_parts,
}


def template_parts(name, gen_length):
    """{offset: text} for a template name; None / "none" means no anchors."""
    if name in (None, "none", ""):
        return {}
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[name](gen_length)


# Fake prefixes tried in order. Tokenizing an anchor on its own would treat its leading
# whitespace as sequence-initial, so a prefix is prepended and its tokens are dropped.
# "a" comes last for Qwen-style BPE (Dream), where the first three all merge with a
# leading newline; it never changes a tokenization an earlier prefix already accepts.
_ANCHOR_PREFIXES = (".", "\n", " ", "a")


def tokenize_anchor(tokenizer, text, device):
    """Token ids for an anchor as it would tokenize mid-sequence, with an exact round-trip.

    A prefix can merge with the anchor's first character (".Going" -> ".G", "oing"), so
    the prefix's own token count is stripped and the result must decode back to ``text``;
    otherwise the next prefix is tried, and if none round-trips this raises.
    """
    for prefix in _ANCHOR_PREFIXES:
        n_prefix = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        ids = tokenizer(prefix + text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        kept = ids[:, n_prefix:]
        if tokenizer.decode(kept[0]) == text:
            return kept.to(device)
    raise ValueError(f"no fake prefix tokenizes {text!r} losslessly (tried {_ANCHOR_PREFIXES})")


def anchor_spans(tokenizer, parts, prompt_len, gen_length, seq_len, device):
    """[(start, end, ids)] absolute spans of the anchors that fit in the response."""
    spans = []
    for position, text in parts.items():
        if position >= gen_length:
            continue
        ids = tokenize_anchor(tokenizer, text, device)
        start = prompt_len + position
        end = start + ids.shape[1]
        if end <= seq_len:
            spans.append((start, end, ids))
    return spans


def build_branches(tokenizer, prompt, gen_length, pos_template, neg_template, device,
                   mask_id=MASK_ID):
    """Initial x_pos / x_neg (prompt + masked response + anchors) and their anchor spans.

    Returns (x_pos, x_neg, pos_spans, neg_spans); x_neg and neg_spans are None when
    neg_template is None. When both branches have anchors they must occupy identical
    token spans, so the branches differ only in anchor content.
    """
    prompt_len = prompt.shape[1]
    seq_len = prompt_len + gen_length

    x_pos = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    x_pos[:, :prompt_len] = prompt
    pos_spans = anchor_spans(tokenizer, template_parts(pos_template, gen_length),
                             prompt_len, gen_length, seq_len, device)
    for start, end, ids in pos_spans:
        x_pos[:, start:end] = ids

    if neg_template is None:
        return x_pos, None, pos_spans, None

    neg_spans = anchor_spans(tokenizer, template_parts(neg_template, gen_length),
                             prompt_len, gen_length, seq_len, device)
    if pos_spans:
        if [(s, e) for s, e, _ in pos_spans] != [(s, e) for s, e, _ in neg_spans]:
            raise ValueError(f"anchor spans of {pos_template!r} and {neg_template!r} differ; "
                             "negative anchors must be token-length-matched")

    x_neg = x_pos.clone()
    for start, end, ids in neg_spans:
        x_neg[:, start:end] = ids
    return x_pos, x_neg, pos_spans, neg_spans
