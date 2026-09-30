#!/usr/bin/env bash
# OpenCompass multiple choice (GPQA / ARC-C / PIQA) on Dream: vanilla vs CFG.
#
#   scripts/run_oc_mc.sh [extra opencompass args, e.g. --debug or -r to reuse]
#
# Env:
#   MODEL_PATH         Dream-v0-Instruct-7B directory
#   OC_PYTHON          OpenCompass interpreter (default: /workspace/dllm/oc/ocenv/bin/python)
#   COMPASS_DATA_CACHE root holding data/ARC and data/piqa (default: /workspace/dllm/oc)
#   CONFIG             OpenCompass config (default: eval_oc/configs/eval_dream_mc_cfg.py)
# OpenCompass resolves every dataset path, GPQA's ./data/gpqa/ included, under COMPASS_DATA_CACHE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OC_PYTHON="${OC_PYTHON:-/workspace/dllm/oc/ocenv/bin/python}"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-/workspace/dllm/oc}"
export MODEL_PATH="${MODEL_PATH:-}"   # only for configs that leave path empty
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# nltk -> sqlite3 needs a newer libstdc++ than the system one.
export LD_LIBRARY_PATH="/opt/conda/envs/future-dllm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CONFIG="${CONFIG:-eval_oc/configs/eval_dream_mc_cfg.py}"

echo "config $CONFIG  model $MODEL_PATH  data $COMPASS_DATA_CACHE  gpus ${CUDA_VISIBLE_DEVICES:-all}"
"$OC_PYTHON" -m opencompass.cli.main "$CONFIG" "$@"
