# Irrelevant_cfg

LLaDA-8B-Instruct / Dream-v0-Instruct-7B에 **negative-template CFG**를 적용합니다. 정답과 무관한 구조의 템플릿(`no_relevance`, 서사 구조)을 negative branch에 두고, 매 스텝 아래처럼 logits를 섞습니다.

    L_guided = (1 + w) * L_pos - w * L_neg        (fp32로 계산, w = 1.0)

positive branch에는 앵커가 없습니다(negative-only CFG). 두 branch는 같은 디코딩 상태를 공유하고, negative 앵커 위치만 다릅니다.

## 디코딩 설정 (모든 태스크 공통)

| 항목 | 값 |
|---|---|
| gen length | 128 |
| block | 단일 블록 (block_length = 128) |
| steps | 128 (스텝당 1토큰 확정) |
| remasking | 기본: low_confidence, temperature 0 (`DECODING=dream`이면 Dream sampler, 아래 참고) |
| cache | 사용 안 함 (매 스텝 전체 시퀀스 forward) |

태스크 yaml의 `max_gen_toks`와 관계없이 항상 128 토큰을 생성합니다.

## 구성

```
irrelevant_cfg/   디코더 (외부 레포 import 없음)
  sampling.py     set_seed, gumbel noise, 스텝별 unmask 개수
  templates.py    앵커 템플릿, 앵커 토크나이즈, 두 branch 초기 상태
  generate.py     단일 블록 / 캐시 없는 CFG 디코딩
eval/
  lm_eval_model.py  lm-eval 모델 `LLaDA_cfg` (Future_dLLM의 generate_until 처리 방식)
  tasks/            Future_dLLM의 local task yaml/utils (gsm8k, math500, humaneval, mbpp)
  tasks_ti/         Template-Infilling의 task (ti_gsm8k, ti_math500, ti_humaneval)
eval_oc/            OpenCompass 모델 `DreamCFGOC` + 객관식 config (GPQA / ARC-C / PIQA)
scripts/run_eval.sh, run_all.sh, run_oc_mc.sh
data -> Future_dLLM/data  (로컬 parquet, 심볼릭 링크)
data_ti/                  ti_* task용 parquet (gitignore, HF에서 받아 SOURCE.json에 revision 기록)
tests/test_generate.py    stub 모델로 CPU에서 도는 디코더 테스트
```

## 실행

```bash
PY=/path/to/python scripts/run_eval.sh <dataset> <vanilla|neg_only_norel|neg_only_norel_notail>
```

- `vanilla`: CFG 없이 디코딩
- `neg_only_norel`: negative = `no_relevance`, w = `W` (기본 1.0)
- `neg_only_norel_notail`: negative = `no_relevance_notail` (끝의 답 앵커 `"\nTherefore, the answer is:"` 없음)

dataset: `gsm8k | math500 | humaneval | mbpp` (Future_dLLM local task) 또는
`ti_gsm8k | ti_math500 | ti_humaneval` (Template-Infilling의 gsm8k / hendrycks_math500 /
humaneval_instruct와 프롬프트·채점이 같고, TI의 후처리 `postprocess=ti`를 씁니다).

디코딩: `DECODING=llada`(기본, low_confidence, temperature 0) 또는 `DECODING=dream`
(Dream `sample_tokens`와 같은 sampler: `DREAM_ALG=entropy|maskgit_plus|origin`,
`DREAM_TEMPERATURE`, `DREAM_TOP_P`, 기본 entropy / 0.2 / 0.95). `origin`은 Dream `_sample`처럼
매 스텝 각 마스크 위치를 확률 1 - s/t로 확정합니다(TI의 vanilla 디코딩). CFG 세기는 `W`.

환경 변수: `MODEL_PATH`, `DATA_DIR`(기본은 `data` 심볼릭 링크), `TI_DATA_DIR`(기본 `data_ti`), `LIMIT`, `OUT_ROOT`, `SHARD`, `DECODING`, `DREAM_*`, `W`.
모델 종류(LLaDA / Dream)는 체크포인트 config에서 읽습니다. Dream은 mask id 151666, `attention_mask="full"`, 한 칸 shift한 logits를 씁니다.
결과는 `results/<model>/<dataset>/<config>_len128/`에 저장됩니다. 생성 결과는 `generations.jsonl`에 한 줄씩 기록되므로, 같은 명령을 다시 실행하면 이어서 돌아갑니다.

전체 데이터셋 x config를 짧은 벤치마크부터(humaneval → math500 → mbpp → gsm8k) GPU 여러 장에 샤딩해 돌리려면:

```bash
PY=... MODEL_PATH=... DATA_DIR=... GPUS="0 1 2 3" scripts/run_all.sh
```

각 실행은 GPU마다 샤드 하나가 `generations.jsonl.shard<i>`를 채우고(샤드마다 `tasks_shard<i>` 폴더를 따로 씀), 이어서 샤드 없이 한 번 더 돌려 전부 재생해 채점합니다. `DATASETS`, `CONFIGS`로 범위를 좁힐 수 있습니다.

태스크는 Future_dLLM의 local task를 쓰되, 모든 태스크를 0-shot으로 평가합니다(`--num_fewshot 0`, yaml의 `num_fewshot`도 0). chat template은 mbpp에만 적용합니다(`--apply_chat_template`).

OpenCompass 객관식(GPQA diamond 5-shot, ARC-C test, PIQA; Future_dLLM eval_oc와 같은 데이터셋 설정):

```bash
MODEL_PATH=.../Dream-v0-Instruct-7B scripts/run_oc_mc.sh
```

## 테스트

```bash
python -c "import sys; sys.path.insert(0,'tests'); import test_generate as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
# 또는 pytest tests/
```
