"""CPU checks of the decoder with a stub model and a character-level stub tokenizer.

    python -m pytest tests/
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irrelevant_cfg import MASK_ID, generate  # noqa: E402
from irrelevant_cfg.templates import build_branches  # noqa: E402

VOCAB = MASK_ID + 1


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) for c in text]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class StubModel(torch.nn.Module):
    """Logits depend on every input token, so anchors change the output."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.randn(VOCAB, 16, generator=g)
        self.out = torch.randn(16, 256, generator=g)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, x):
        h = self.emb[x]
        h = h + h.mean(dim=1, keepdim=True)
        logits = torch.full((*x.shape, VOCAB), -1e4)
        logits[..., :256] = h @ self.out
        return SimpleNamespace(logits=logits)


def _prompt():
    return torch.tensor([[ord(c) for c in "What is 2+3?"]])


def test_vanilla_single_block_fills_every_position():
    tok, model = CharTokenizer(), StubModel()
    prompt = _prompt()
    out = generate(model, tok, prompt, gen_length=64, neg_template=None)
    assert out.shape == (1, prompt.shape[1] + 64)
    assert torch.equal(out[:, :prompt.shape[1]], prompt)
    assert not (out == MASK_ID).any()


def test_zero_scale_is_vanilla():
    tok, model = CharTokenizer(), StubModel()
    a = generate(model, tok, _prompt(), gen_length=64, neg_template=None)
    b = generate(model, tok, _prompt(), gen_length=64, neg_template="no_relevance",
                 guidance_scale=0.0)
    assert torch.equal(a, b)


def test_negative_only_cfg_changes_output_and_keeps_anchors_out_of_positive():
    tok, model = CharTokenizer(), StubModel()
    prompt = _prompt()
    a = generate(model, tok, prompt, gen_length=64, neg_template=None)
    b = generate(model, tok, prompt, gen_length=64, neg_template="no_relevance",
                 guidance_scale=1.0)
    assert not (b == MASK_ID).any()
    assert not torch.equal(a, b)
    x_pos, x_neg, pos_spans, neg_spans = build_branches(tok, prompt, 64, None, "no_relevance",
                                                        torch.device("cpu"))
    # Character-level ids make anchors long, so not all three fit in 64 tokens.
    assert pos_spans == [] and neg_spans
    assert (x_pos[:, prompt.shape[1]:] == MASK_ID).all()
    assert not torch.equal(x_pos, x_neg)


def test_one_token_committed_per_step():
    tok, model = CharTokenizer(), StubModel()
    prompt = _prompt()
    committed = []

    real_forward = model.forward

    def counting_forward(x):
        committed.append(int((x[:, prompt.shape[1]:] != MASK_ID).sum()))
        return real_forward(x)

    model.forward = counting_forward
    generate(model, tok, prompt, gen_length=32, neg_template=None)
    assert committed == list(range(32))
