#!/usr/bin/env bash
# lm-eval run with the single-block, cache-free LLaDA decoder (eval/lm_eval_model.py).
# Every task is 0-shot and generates 128 tokens: one block, 128 steps, one token per step.
#
#   scripts/run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel>
#
# Env:
#   MODEL_PATH  LLaDA-8B-Instruct or Dream-v0-Instruct-7B directory
#   DATA_DIR    parquet root in Future_dLLM's data/ layout (default: <repo>/data symlink)
#   PY          python interpreter with lm-eval 0.4.x (default: python)
#   LIMIT       evaluate only the first LIMIT examples
#   OUT_ROOT    results root (default: <repo>/results)
#   SHARD       "i/n": decode only every n-th request into the shared generations.jsonl
#               and skip scoring; run n shards, then once without SHARD to score
#
# Generations are appended to <out>/generations.jsonl as they finish; rerunning the
# same command resumes from there.
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel>}"
CONFIG="${2:?usage: run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/srv/home/dlpch.345/dllm/model/LLaDA-8B-Instruct}"
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data}" && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/results}"
GEN_LENGTH=128

# Future_dLLM's task set, all 0-shot. MBPP keeps its chat template; with no shots
# there is nothing for --fewshot_as_multiturn to split into turns.
CHAT_ARGS=()
SHOTS=(--num_fewshot 0)
case "$DATASET" in
  gsm8k)     TASK=local_gsm8k ;;
  math500)   TASK=local_math500 ;;
  humaneval) TASK=local_humaneval ;;
  mbpp)      TASK=local_mbpp; CHAT_ARGS=(--apply_chat_template) ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac
case "$CONFIG" in
  vanilla|neg_only_norel) ;;
  *) echo "unknown config: $CONFIG" >&2; exit 1 ;;
esac

MODEL_TAG="$(basename "$MODEL_PATH")"
OUT="$OUT_ROOT/$MODEL_TAG/$DATASET/${CONFIG}_len${GEN_LENGTH}${LIMIT:+_limit$LIMIT}"
TASKS_DIR="$OUT/tasks"
mkdir -p "$TASKS_DIR"
cp "$ROOT"/eval/tasks/local_*.py "$TASKS_DIR/"
for y in "$ROOT"/eval/tasks/*.yaml; do
  sed "s|DATA_DIR|$DATA_DIR|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

export LLADA_CFG_STORE="$OUT/generations.jsonl"
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LLADA_CFG_SHARD="${SHARD:-0/1}"
RUN_OUT="$OUT/out"
[ -n "${SHARD:-}" ] && RUN_OUT="$OUT/shard_out/${SHARD%/*}"

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "$MODEL_TAG $DATASET config=$CONFIG gen_length=$GEN_LENGTH samples=${LIMIT:-all} shard=${SHARD:-none} -> $OUT"
cd "$ROOT"
"$PY" eval/lm_eval_model.py \
  --model LLaDA_cfg \
  --model_args "pretrained=$MODEL_PATH,config=$CONFIG,gen_length=$GEN_LENGTH" \
  --tasks "$TASK" "${SHOTS[@]}" \
  --include_path "$TASKS_DIR" \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --log_samples \
  "${LIMIT_ARGS[@]}" \
  "${CHAT_ARGS[@]}" \
  --output_path "$RUN_OUT"
