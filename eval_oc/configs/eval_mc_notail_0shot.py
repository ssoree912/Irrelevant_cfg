"""OpenCompass multiple choice, 0-shot, 128 generated tokens, vanilla vs no-tail CFG.

    Dream-v0-Instruct-7B  entropy, temperature 0.2, top_p 0.95   CFG w = 0.5
    LLaDA-8B-Instruct     low_confidence, temperature 0          CFG w = 1.0

The negative is no_relevance_notail. Datasets: GPQA diamond (0-shot), ARC-C test, PIQA;
MMLU has its own config (eval_mmlu_notail_0shot.py) that reuses these models.

    CONFIG=eval_oc/configs/eval_mc_notail_0shot.py scripts/run_oc_mc.sh
"""
from mmengine.config import read_base

from eval_oc.model import DllmCFGOC
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_0shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

_DREAM = "/workspace/dllm/Future_dLLM/model/Dream-v0-Instruct-7B"
_LLADA = "/workspace/dllm/Future_dLLM/model/LLaDA-8B-Instruct"
_dream_dec = dict(sampler="dream", alg="entropy", temperature=0.2, top_p=0.95)
_llada_dec = dict(sampler="llada", alg="entropy", temperature=0.0, top_p=1.0)

# (abbr, path, config, w, decoding)
_rows = [
    ("dream-vanilla", _DREAM, "vanilla", 0.5, _dream_dec),
    ("dream-cfg-notail-w0.5", _DREAM, "neg_only_norel_notail", 0.5, _dream_dec),
    ("llada-vanilla", _LLADA, "vanilla", 1.0, _llada_dec),
    ("llada-cfg-notail-w1.0", _LLADA, "neg_only_norel_notail", 1.0, _llada_dec),
]

models = [
    dict(
        type=DllmCFGOC,
        abbr=abbr,
        path=path,
        config=config,
        w=w,
        max_seq_len=2048,
        seed=1234,
        max_out_len=128,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
        **dec,
    )
    for abbr, path, config, w, dec in _rows
]

# Per model: every dataset split in 4, packed into 4 tasks (one model load each), 4 GPUs.
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLEvalTask)),
)

work_dir = "results/oc_mc_notail_0shot"
