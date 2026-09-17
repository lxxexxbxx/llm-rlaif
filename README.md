# llm-rlaif

Gemma3-4B에 RLAIF 기반 PPO를 적용한 학습·평가 파이프라인

[![Policy Model](https://img.shields.io/badge/🤗%20Model-PPO%20Policy-yellow)](https://huggingface.co/lxxexxbxx/gemma3-4b-ko-rlaif-ppo)
[![Reward Model](https://img.shields.io/badge/🤗%20Model-Reward-yellow)](https://huggingface.co/lxxexxbxx/gemma3-4b-ko-rlaif-reward)
[![Base](https://img.shields.io/badge/base-gemma--3--4b--it-blue)](https://huggingface.co/google/gemma-3-4b-it)

---

## 개요

- 한국공학대학교 딥러닝응용 팀 프로젝트(2025-2)
- 3인 팀에서 PPO 트랙과 DPO 트랙을 분리해 진행, **본 저장소는 단독 담당한 PPO 트랙의 산출물**
- 목표: 4B 경량 모델 + 4-bit 양자화 환경에서 RLAIF 기반 보상 학습이 실제로 동작하는지 검증
- 결과: **PPO 적용 후 baseline 대비 성능 하락.** 원인을 보상 설계와 평가 데이터 가공 결함으로 규명

## 파이프라인

```
[1] 데이터 구축      TruthfulQA 한국어 번역 250 + KMMLU QA 변환 250
                     → 평가셋 50문항 분리
[2] RLAIF 데이터     Gemini 심판으로 Chosen/Rejected 분류 → 선호 쌍 500건
[3] 보상 모델 학습   Gemma3-4B + LoRA + score 헤드 (SEQ_CLS)
[4] PPO 학습         보상 모델 신호로 정책 모델 업데이트
[5] 평가             Gemini 2.5 Flash 심판, Accuracy / Conciseness 분리 채점
```

## 모델 구성

| 구성 요소 | 배포 위치 | 파일 |
|---|---|---|
| 정책 모델(Policy) | [🤗 gemma3-4b-ko-rlaif-ppo](https://huggingface.co/lxxexxbxx/gemma3-4b-ko-rlaif-ppo) | `adapter_model.safetensors` (131 MB) |
| 가치 헤드(Value) | 위와 동일 저장소 | `pytorch_model.bin` (7 kB) |
| 보상 모델(Reward) | [🤗 gemma3-4b-ko-rlaif-reward](https://huggingface.co/lxxexxbxx/gemma3-4b-ko-rlaif-reward) | `adapter_model.safetensors` (24 MB) |
| 참조 모델(Reference) | 별도 저장 없음 | LoRA 비활성화 base 모델로 대체 |

- TRL `AutoModelForCausalLMWithValueHead`는 정책과 가치가 백본을 공유 → 한 저장소에 함께 저장
- 모델 가중치는 GitHub 100 MB 제한으로 Hugging Face Hub에 분리 배포

## 학습 구성

| 항목 | 값 |
|---|---|
| Base | `google/gemma-3-4b-it` (4-bit 양자화, BitsAndBytes) |
| LoRA | `r=16`, `lora_alpha=32`, `lora_dropout=0.1` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Reward 모델 | `task_type=SEQ_CLS`, `modules_to_save=["score"]` |
| Framework | TRL `PPOTrainer` / `RewardTrainer` + PEFT |
| 학습 데이터 | RLAIF 선호 쌍 500건 |
| 환경 | RunPod RTX 3090 / Google Colab A100 |

### 구현 시 다룬 문제

- **Gemma 3 분류 헤드 미구성** — `AutoModelForSequenceClassification` 로드 시 `score` 헤드가 자동 생성되지 않아, 명시적으로 추가하고 `modules_to_save`에 포함
- **`token_type_ids` 요구** — chosen/rejected 배치 처리 시 해당 필드를 0으로 채워 전달하는 커스텀 콜레이터 `GemmaRewardCollator` 작성

## 평가 결과

- 평가셋 50문항 (TruthfulQA-ko 25 + KMMLU-QA 25)
- 심판: Gemini 2.5 Flash, `temperature=0` (학습 미사용 외부 LLM)
- Accuracy / Conciseness 분리 채점, 각 10점 만점

| 구성 | n | Accuracy | Conciseness | Accuracy = 0 |
|---|---|---|---|---|
| Base Gemma3-4B | 50 | 1.10 | 1.68 | 44 |
| **+ PPO** | 50 | **0.78** | **0.82** | **46** |

재현: `python scripts/analyze_results.py`

### 원인 분석

**1. 보상 해킹(Reward hacking)**

보상 모델이 "모른다"는 응답보다 근거 없이 풍부하게 설명하는 응답에 높은 점수를 부여했고,
정책 모델이 이 패턴을 학습하면서 할루시네이션이 증가했습니다.

**2. 평가 데이터 가공 결함**

KMMLU 객관식 문항을 QA 형식으로 변환하는 과정에서, 질문은 객관식 형태인데
정답으로는 보기 내용만 제공되는 논리적 불일치가 발생했습니다.
모델이 존재하지 않는 보기를 생성해 답하고 0점 처리된 사례가 다수 확인됩니다.

**3. Base 모델조차 1.10에 그친 이유**

심판이 "정답의 핵심 언급 여부"를 기준으로 삼는 반면 Gemma3-4B는 마크다운 불릿 기반
장문으로 응답해 핵심이 희석되었습니다. 즉 이 수치는 모델 성능뿐 아니라
**평가 설계의 한계를 함께 반영**합니다.

> DPO 트랙은 같은 팀의 다른 구성원이 담당했으며, 코드와 원본 결과 데이터를 보유하고 있지 않아
> 본 저장소에 포함하지 않았습니다.

## 실행 방법

### 준비

```bash
git clone https://github.com/lxxexxbxx/llm-rlaif.git
cd llm-rlaif

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # GEMINI_API_KEY, HF_TOKEN 입력
```

### 결과 집계 (CPU, 맥북에서 바로 실행 가능)

```bash
python scripts/analyze_results.py
```

### 노트북

```bash
jupyter lab
# notebooks/01_build_dataset.ipynb  데이터셋 구축
# notebooks/02_rlaif_ppo.ipynb      RLAIF → 보상 모델 → PPO → 평가
```

> **학습 노트북은 CUDA GPU가 필요합니다.** 4-bit 양자화(BitsAndBytes)가 CUDA 전용이라
> Apple Silicon(MPS)에서는 실행되지 않습니다. 맥북에서는 데이터 구축과 결과 집계까지만 가능하며,
> 학습은 RunPod·Colab 등 CUDA 환경에서 수행하세요.

경로는 `.env`로 덮어쓸 수 있습니다. 미설정 시 저장소 상대 경로(`data/`, `results/`, `models/`)를 사용합니다.

## 구조

```
llm-rlaif/
├── notebooks/
│   ├── 01_build_dataset.ipynb   TruthfulQA 번역 · KMMLU QA 변환 · 평가셋 분리
│   └── 02_rlaif_ppo.ipynb       RLAIF 생성 → 보상 모델 → PPO → 평가 · 시각화
├── scripts/
│   └── analyze_results.py       평가 JSON 집계 및 그래프 재생성 (GPU 불필요)
├── data/
│   ├── rlaif_dataset_final.json         선호 쌍 500건
│   ├── truthfulqa_ko_250.json
│   ├── kmmlu_qa_style_250.json
│   └── evaluation_dataset_*.json        평가셋 50 / tqa 25 / kmmlu 25
├── results/
│   ├── eval_base_4b.json        Base 모델 채점 결과
│   ├── eval_ppo.json            PPO 모델 채점 결과
│   └── charts/
└── docs/딥러닝응용_팀프로젝트_LLM강화학습(레포트).pdf
```

## 데이터 출처 및 라이선스

- Base 모델: [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) — [Gemma Terms of Use](https://ai.google.dev/gemma/terms) 적용
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) (Apache-2.0) 한국어 번역본
- [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) — HAERAE-HUB
- 본 저장소는 LoRA 어댑터만 배포하며 base 모델 가중치를 재배포하지 않습니다
