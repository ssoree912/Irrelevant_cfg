#!/usr/bin/env bash
# Every dataset x config, shortest benchmark first, each run sharded over all GPUs.
#
#   scripts/run_all.sh
#
# Env: GPUS (default "0 1 2 3"), DATASETS, CONFIGS, plus everything run_eval.sh reads.
# Each run: one shard per GPU fills generations.jsonl, then an unsharded pass on the
# first GPU replays it and scores. Rerunning resumes from what is on disk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
read -r -a DATASET_LIST <<< "${DATASETS:-humaneval math500 mbpp gsm8k}"
read -r -a CONFIG_LIST <<< "${CONFIGS:-vanilla neg_only_norel}"
N="${#GPU_LIST[@]}"

for dataset in "${DATASET_LIST[@]}"; do
  for config in "${CONFIG_LIST[@]}"; do
    pids=()
    for i in "${!GPU_LIST[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" SHARD="$i/$N" \
        "$ROOT/scripts/run_eval.sh" "$dataset" "$config" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$ROOT/scripts/run_eval.sh" "$dataset" "$config"
  done
done
