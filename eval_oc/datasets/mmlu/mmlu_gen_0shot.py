"""MMLU, 0-shot: OpenCompass's mmlu_gen_79e572 (the config Future_dLLM's eval_dream_mmlu.py
uses) with its five dev-split examples removed.

Only the retriever changes (FixKRetriever -> ZeroRetriever); the prompt keeps the per-subject
header "The following are multiple choice questions (with answers) about <subject>." and the
"Answer:" cue, and scoring stays first_capital_postprocess + AccEvaluator. Full test split.
"""
from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import \
        mmlu_datasets as _mmlu_5shot

mmlu_datasets = [
    dict(d, infer_cfg=dict(d['infer_cfg'], retriever=dict(type=ZeroRetriever)))
    for d in _mmlu_5shot
]
