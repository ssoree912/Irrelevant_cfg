"""Two items per dataset, vanilla and CFG w=1.0, in-process: a wiring check before the sweep."""
from mmengine.config import read_base

with read_base():
    from .eval_dream_mc_cfg import datasets, infer, eval, models  # noqa: F401

for _d in datasets:
    _d["reader_cfg"] = dict(_d["reader_cfg"], test_range="[0:2]")
models = models[:2]
work_dir = "results/oc_mc_smoke"
