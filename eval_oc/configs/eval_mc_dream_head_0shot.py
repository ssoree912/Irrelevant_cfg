"""OpenCompass multiple choice, 0-shot, 128 generated tokens, Dream only, one GPU:
vanilla vs no-tail CFG at w = 0.25 (a head_lazy row is kept commented out).

    Dream-v0-Instruct-7B  entropy, temperature 0.2, top_p 0.95

Datasets as in eval_mc_notail_0shot.py (GPQA diamond 0-shot, ARC-C test, PIQA, WinoGrande).
OC_DATASETS (comma-separated: gpqa, arc_c, piqa, winogrande; default all) picks a subset, so a
queue can run the shortest first; OC_SMOKE=1 keeps two questions per dataset.

    OC_DATASETS=gpqa CONFIG=eval_oc/configs/eval_mc_dream_head_0shot.py scripts/run_oc_mc.sh
"""
from mmengine.config import read_base

from eval_oc.model import DllmCFGOC
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets

    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_0shot import gpqa_datasets

# mmengine parses this file in lazy-import mode (an `import os` would be a lazy object) and
# dumps every module-level name, so the environment is read inline and never stored.
_by_name = {"gpqa": gpqa_datasets, "arc_c": ARC_c_datasets, "piqa": piqa_datasets,
            "winogrande": winogrande_datasets}
datasets = [d for n in __import__("os").environ.get("OC_DATASETS", ",".join(_by_name)).split(",")
            if n for d in _by_name[n]]
if __import__("os").environ.get("OC_SMOKE"):
    datasets = [dict(d, reader_cfg=dict(d["reader_cfg"], test_range="[0:2]")) for d in datasets]

_DREAM = __import__("os").environ.get("DREAM_PATH",
                                      "/mnt/srv/home/dlpch.345/dllm/model/Dream-v0-Instruct-7B")
_dream_dec = dict(sampler="dream", alg="entropy", temperature=0.2, top_p=0.95)

# (abbr, config, w)
_rows = [
    ("dream-vanilla", "vanilla", 0.25),
    ("dream-cfg-notail-w0.25", "neg_only_norel_notail", 0.25),
    # ("dream-cfg-head_lazy-w0.25", "neg_head_lazy", 0.25),  # dropped: weaker on countdown
]
# OC_ROWS (comma-separated configs, e.g. neg_only_norel_notail; default all) picks rows.
_rows = [r for r in _rows if r[1] in (__import__("os").environ.get("OC_ROWS")
                                      or ",".join(r[1] for r in _rows)).split(",")]

models = [
    dict(
        type=DllmCFGOC,
        abbr=abbr,
        path=_DREAM,
        config=config,
        w=w,
        max_seq_len=2048,
        seed=1234,
        max_out_len=128,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
        **_dream_dec,
    )
    for abbr, config, w in _rows
]

# One GPU: one task per model and dataset, run one at a time.
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLEvalTask)),
)

work_dir = "results/oc_mc_dream_head_0shot"
