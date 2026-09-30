"""OpenCompass multiple choice (GPQA diamond 5-shot, ARC-C test, PIQA) for Dream:
vanilla vs negative-template CFG (w=1.0, w=0.5).

Datasets are Future_dLLM's eval_oc MC suite (eval_dream_mc.py): PIQA from the installed
OpenCompass, GPQA 5-shot and ARC-C test vendored under eval_oc/datasets/. Decoding is
Dream's recommended sampler (entropy, temperature 0.2, top_p 0.95) in one block of
max_out_len = 128 tokens, 128 steps, like the lm-eval runs of this repo.

    scripts/run_oc_mc.sh
"""
from mmengine.config import read_base

from eval_oc.model import DreamCFGOC
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

# (abbr, config, w)
_rows = [
    ("dream-vanilla", "vanilla", 1.0),
    ("dream-cfg-w1.0", "neg_only_norel", 1.0),
    ("dream-cfg-w0.5", "neg_only_norel", 0.5),
]

models = [
    dict(
        type=DreamCFGOC,
        abbr=abbr,
        path="",                        # <- MODEL_PATH
        config=config,
        w=w,
        max_seq_len=2048,
        alg="entropy",
        temperature=0.2,
        top_p=0.95,
        seed=1234,
        max_out_len=128,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    for abbr, config, w in _rows
]

# Each dataset is split into 4 shards and run 4 at a time, one GPU each.
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLEvalTask)),
)

work_dir = "results/oc_mc"
