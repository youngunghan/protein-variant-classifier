# Configuration

> **범위:** 학습 스크립트와 score 분석 스크립트의 CLI 옵션. 데이터 의미는 `data-format.md`, 평가 해석은 `evaluation-validity.md`를 본다.
> **대상:** 개발자, 실험 실행자.
> **상태:** 구현 반영 — 기준일 2026-06-14.

## 1. Training CLI

실행 파일: [train_esm_classifier.py](../../code/train_esm_classifier.py) `build_parser()`

### 1.1 데이터 입력

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--train_csv` | 없음 | 학습 CSV 경로. `--use_mock_data`가 아니면 필수 |
| `--val_csv` | 없음 | 검증 CSV 경로. 없으면 validation metric을 계산하지 않음 |
| `--wt_col` | `wt_seq` | Wild-type sequence 컬럼 |
| `--mut_col` | `mut_seq` | Mutant sequence 컬럼 |
| `--label_col` | `label` | `0 = LOF`, `1 = GOF` label 컬럼 |
| `--position_col` | 없음 | optional 1-based 변이 위치 컬럼. 지정하면 해당 residue의 per-position embedding 사용 |
| `--use_mock_data` | false | smoke test용 synthetic 데이터 사용 |

### 1.2 모델과 학습

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model_name` | `facebook/esm2_t33_650M_UR50D` | Hugging Face ESM model |
| `--freeze_backbone` / `--no-freeze_backbone` | true | ESM backbone freeze 여부 |
| `--embedding_cache` | `auto` | `auto`는 frozen backbone일 때 ESM feature를 precompute 후 재사용, `on`은 강제, `off`는 비활성화 |
| `--embedding_cache_dir` | 없음 | cache 저장 디렉터리. 없으면 `output_dir/embedding_cache` 사용 |
| `--class_weight` | `auto` | `auto`는 train label 분포에서 class weight 계산, `none`은 미사용 |
| `--batch_size` | `8` | GPU당 batch size |
| `--max_len` | `1024` | tokenizer truncation 상한. 실제 padding은 batch 내 최장 sequence 길이에 맞춤 |
| `--epochs` | `10` | 학습 epoch 수 |
| `--lr` | `1e-4` | AdamW learning rate |
| `--num_workers` | `0` | DataLoader worker 수 |
| `--seed` | `13` | Python, PyTorch, NumPy(설치된 경우), DataLoader worker seed |

### 1.3 출력과 분산 학습

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--output_dir` | `runs/esm2_variant` | checkpoint/config/metrics 저장 경로 |
| `--backend` | `nccl` | DDP backend. CPU DDP는 `gloo` 사용 |
| `--local_rank` | `-1` | `torchrun`이 주입하는 local rank. 보통 직접 지정하지 않음 |

## 2. Score Analysis CLI

실행 파일: [analyze_scores.py](../../code/analyze_scores.py) `build_parser()`

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `file_path` | 없음 | score CSV 경로 |
| `--patient_col` | `Patient_ID` | 환자 또는 case 식별자 컬럼 |
| `--label_col` | `LABEL` | `0/1` label 컬럼 |
| `--score_cols` | `SCORE_A SCORE_B SCORE_C` | 비교할 score 컬럼 목록 |

## 3. Metric 정책

| 상황 | 동작 |
|---|---|
| validation label이 `0/1` 양쪽을 모두 포함 | accuracy, AUROC, AUPRC 계산 |
| validation label이 한 클래스만 포함 | AUROC/AUPRC는 `skipped`, loss/accuracy만 출력 |
| frozen backbone + embedding cache enabled | `train.pt`/`val.pt` feature cache를 metadata(model, max_len, records hash, pooling policy)와 함께 저장하고 classifier head만 학습 |
| score 분석에서 pathogenic variant가 없는 patient | Top-k ranking 분모에서 제외하고 제외 수 출력 |
| score 분석에서 분모가 0 | Top-k metric을 `skipped`로 출력 |
| score 분석에서 동점 score 발생 | Hit@k는 k번째 cutoff score 이상에 pathogenic variant가 있으면 hit로 계산 |
| score 분석 score 값이 `NaN`/`inf` | CSV 로딩 단계에서 거부 |
