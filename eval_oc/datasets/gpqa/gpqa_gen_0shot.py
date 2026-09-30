"""GPQA diamond, 0-shot: gpqa_gen_5shot.py with the five in-context examples removed.

Prompt, reader, evaluator and postprocessor are the 5-shot config's; only the retriever
changes (FixKRetriever -> ZeroRetriever), so the ``</E>`` slot renders empty and the
prompt is the hint followed by the question.
"""
from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from .gpqa_gen_5shot import gpqa_datasets as _gpqa_5shot

gpqa_datasets = [
    dict(d, abbr=d['abbr'].replace('_5shot', '_0shot'),
         infer_cfg=dict(d['infer_cfg'], retriever=dict(type=ZeroRetriever)))
    for d in _gpqa_5shot
]
