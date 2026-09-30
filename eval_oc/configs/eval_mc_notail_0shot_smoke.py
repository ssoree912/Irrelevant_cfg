"""Two items per dataset (GPQA / ARC-C / PIQA / one MMLU subject), all four rows: a wiring check."""
from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_gen_0shot import mmlu_datasets
    from .eval_mc_notail_0shot import datasets as _mc, eval, infer, models  # noqa: F401

datasets = [dict(d, reader_cfg=dict(d["reader_cfg"], test_range="[0:2]"))
            for d in [*_mc, mmlu_datasets[0]]]
work_dir = "results/oc_mc_notail_0shot_smoke"
