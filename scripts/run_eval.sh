#!/usr/bin/env bash
# lm-eval run with the single-block, cache-free LLaDA decoder (eval/lm_eval_model.py).
# Every task is 0-shot and generates 128 tokens: one block, 128 steps, one token per step.
#
#   scripts/run_eval.sh <gsm8k|math500|humaneval|mbpp|ti_gsm8k|ti_math500|ti_humaneval> <vanilla|neg_only_norel|neg_only_norel_notail>
#
# ti_* are Template-Infilling's own tasks (gsm8k, hendrycks_math500, humaneval_instruct;
# eval/tasks_ti) with its answer post-processing (postprocess=ti).
#
# Env:
#   MODEL_PATH  LLaDA-8B-Instruct or Dream-v0-Instruct-7B directory
#   DATA_DIR    parquet root in Future_dLLM's data/ layout (default: <repo>/data symlink)
#   TI_DATA_DIR parquet root for the ti_* tasks (default: <repo>/data_ti)
#   LONGBENCH_DATA  LongBench jsonl dir for longbench_* (default: $DATA_DIR/longbench/data)
#   MAX_SEQ_LEN prompt + generation budget; prompts are left-truncated to MAX_SEQ_LEN - 128
#               (default 4096; other values tag the dir _msl<N>)
#   PY          python interpreter with lm-eval 0.4.x (default: python)
#   LIMIT       evaluate only the first LIMIT examples
#   OUT_ROOT    results root (default: <repo>/results)
#   DECODING    llada (default: greedy, low_confidence) or dream (Dream's sampler with
#               DREAM_ALG=entropy, DREAM_TEMPERATURE=0.2, DREAM_TOP_P=0.95)
#   W           guidance scale for neg_only_norel (default 1.0; other values tag the dir _w<W>)
#   SHARD       "i/n": decode only every n-th request into the shared generations.jsonl
#               and skip scoring; run n shards, then once without SHARD to score
#
# Generations are appended to <out>/generations.jsonl as they finish; rerunning the
# same command resumes from there.
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel|neg_only_norel_notail>}"
CONFIG="${2:?usage: run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel|neg_only_norel_notail>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/srv/home/dlpch.345/dllm/model/LLaDA-8B-Instruct}"
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
  ti_gsm8k|ti_math500|ti_humaneval) TASK="$DATASET" ;;
  longbench_*) TASK="$DATASET" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac
case "$CONFIG" in
  vanilla|neg_only_norel|neg_only_norel_notail) ;;
  *) echo "unknown config: $CONFIG" >&2; exit 1 ;;
esac

DECODING="${DECODING:-llada}"
DECODE_ARGS=""
DECODE_TAG=""
case "$DECODING" in
  llada) ;;
  dream) DREAM_ALG="${DREAM_ALG:-entropy}" DREAM_TEMPERATURE="${DREAM_TEMPERATURE:-0.2}"
         DREAM_TOP_P="${DREAM_TOP_P:-0.95}"
         DECODE_ARGS=",sampler=dream,alg=$DREAM_ALG,temperature=$DREAM_TEMPERATURE,top_p=$DREAM_TOP_P"
         DECODE_TAG="_dream_${DREAM_ALG}_t${DREAM_TEMPERATURE}_p${DREAM_TOP_P}" ;;
  *) echo "unknown decoding: $DECODING" >&2; exit 1 ;;
esac
W="${W:-1.0}"
W_TAG=""
if [ "$CONFIG" != vanilla ] && [ "$W" != 1.0 ]; then
  W_TAG="_w$W"
  DECODE_ARGS="$DECODE_ARGS,w=$W"
fi
[[ "$DATASET" == ti_* ]] && DECODE_ARGS="$DECODE_ARGS,postprocess=ti"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MSL_TAG=""
if [ "$MAX_SEQ_LEN" != 4096 ]; then
  MSL_TAG="_msl$MAX_SEQ_LEN"
  DECODE_ARGS="$DECODE_ARGS,max_seq_len=$MAX_SEQ_LEN"
fi
MODEL_TAG="$(basename "$MODEL_PATH")"
OUT="$OUT_ROOT/$MODEL_TAG/$DATASET/${CONFIG}_len${GEN_LENGTH}${DECODE_TAG}${W_TAG}${MSL_TAG}${LIMIT:+_limit$LIMIT}"
# Shards launched together each get their own copy, so concurrent cp/sed never race.
TASKS_DIR="$OUT/tasks${SHARD:+_shard${SHARD%/*}}"
mkdir -p "$TASKS_DIR"
if [[ "$DATASET" == longbench_* ]]; then
  DATA_DIR_LB="${DATA_DIR:-$ROOT/data}"
  LONGBENCH_DATA="$(cd "${LONGBENCH_DATA:-$DATA_DIR_LB/longbench/data}" && pwd)"
  cp "$ROOT"/eval/tasks_longbench/metrics.py "$TASKS_DIR/"
  for y in "$ROOT"/eval/tasks_longbench/*.yaml; do
    sed "s|LONGBENCH_DATA_DIR|$LONGBENCH_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
  done
elif [[ "$DATASET" == ti_* ]]; then
  TI_DATA_DIR="$(cd "${TI_DATA_DIR:-$ROOT/data_ti}" && pwd)"
  cp "$ROOT"/eval/tasks_ti/ti_*.py "$TASKS_DIR/"
  for y in "$ROOT"/eval/tasks_ti/*.yaml; do
    sed "s|TI_DATA_DIR|$TI_DATA_DIR|" "$y" > "$TASKS_DIR/$(basename "$y")"
  done
else
  DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data}" && pwd)"
  cp "$ROOT"/eval/tasks/local_*.py "$TASKS_DIR/"
  for y in "$ROOT"/eval/tasks/*.yaml; do
    sed "s|DATA_DIR|$DATA_DIR|" "$y" > "$TASKS_DIR/$(basename "$y")"
  done
fi

export LLADA_CFG_STORE="$OUT/generations.jsonl"
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LLADA_CFG_SHARD="${SHARD:-0/1}"
RUN_OUT="$OUT/out"
[ -n "${SHARD:-}" ] && RUN_OUT="$OUT/shard_out/${SHARD%/*}"

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

echo "$MODEL_TAG $DATASET config=$CONFIG gen_length=$GEN_LENGTH decoding=$DECODING w=$W samples=${LIMIT:-all} shard=${SHARD:-none} -> $OUT"
cd "$ROOT"
"$PY" eval/lm_eval_model.py \
  --model LLaDA_cfg \
  --model_args "pretrained=$MODEL_PATH,config=$CONFIG,gen_length=$GEN_LENGTH$DECODE_ARGS" \
  --tasks "$TASK" "${SHOTS[@]}" \
  --include_path "$TASKS_DIR" \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --log_samples \
  "${LIMIT_ARGS[@]}" \
  "${CHAT_ARGS[@]}" \
  --output_path "$RUN_OUT"
