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
| remasking | low_confidence, temperature 0 |
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
scripts/run_eval.sh
data -> Future_dLLM/data  (로컬 parquet, 심볼릭 링크)
tests/test_generate.py    stub 모델로 CPU에서 도는 디코더 테스트
```

## 실행

```bash
PY=/path/to/python scripts/run_eval.sh <gsm8k|math500|humaneval|mbpp> <vanilla|neg_only_norel>
```

- `vanilla`: CFG 없이 기본 LLaDA 디코딩
- `neg_only_norel`: negative = `no_relevance`, w = 1.0

환경 변수: `MODEL_PATH`, `DATA_DIR`(기본은 `data` 심볼릭 링크), `LIMIT`, `OUT_ROOT`, `SHARD`.
모델 종류(LLaDA / Dream)는 체크포인트 config에서 읽습니다. Dream은 mask id 151666, `attention_mask="full"`, 한 칸 shift한 logits를 씁니다.
결과는 `results/<model>/<dataset>/<config>_len128/`에 저장됩니다. 생성 결과는 `generations.jsonl`에 한 줄씩 기록되므로, 같은 명령을 다시 실행하면 이어서 돌아갑니다.

전체 데이터셋 x config를 짧은 벤치마크부터(humaneval → math500 → mbpp → gsm8k) GPU 여러 장에 샤딩해 돌리려면:

```bash
PY=... MODEL_PATH=... DATA_DIR=... GPUS="0 1 2 3" scripts/run_all.sh
```

각 실행은 GPU마다 샤드 하나가 `generations.jsonl.shard<i>`를 채우고, 이어서 샤드 없이 한 번 더 돌려 전부 재생해 채점합니다.

few-shot / chat 설정은 Future_dLLM `scripts/run_eval.sh`와 같습니다: gsm8k 5-shot, mbpp 3-shot(`--apply_chat_template --fewshot_as_multiturn`), math500·humaneval 0-shot.

## 테스트

```bash
python -c "import sys; sys.path.insert(0,'tests'); import test_generate as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
# 또는 pytest tests/
```
