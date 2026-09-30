"""MMLU (all 57 subjects, full test split, 0-shot) with the models of eval_mc_notail_0shot.py.

    CONFIG=eval_oc/configs/eval_mmlu_notail_0shot.py scripts/run_oc_mc.sh
"""
from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_gen_0shot import mmlu_datasets
    from .eval_mc_notail_0shot import eval, infer, models  # noqa: F401

datasets = mmlu_datasets
work_dir = "results/oc_mmlu_notail_0shot"
