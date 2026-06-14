# Quickstart

> **범위:** CSV로 ESM2 variant classifier를 한 번 학습하고 산출물을 확인하는 최소 경로.
> **대상:** 개발자, 실험 실행자.
> **상태:** 구현 반영 — 기준일 2026-06-14.

## 1. 환경 준비

```bash
conda create -n protein_classifier python=3.10
conda activate protein_classifier
pip install -r requirements.txt
```

## 2. CSV 준비

학습 CSV는 최소 세 컬럼을 가져야 합니다. 단일 amino-acid substitution 위치를 알고 있으면 `position` 컬럼을 추가하는 것을 권장합니다. Indel/복합 변이는 현재 position path가 아니라 별도 전처리나 CLS fallback으로 다룹니다.

```csv
wt_seq,mut_seq,position,label
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDILG,26,0
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDIKD,25,1
```

자세한 계약은 [§1 학습 CSV](../reference/data-format.md#1-학습-csv)를 봅니다.

## 3. 학습 실행

```bash
python code/train_esm_classifier.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --position_col position \
  --output_dir runs/esm2_variant_mvp \
  --batch_size 2 \
  --max_len 512 \
  --epochs 3
```

기본 설정은 frozen ESM backbone입니다. 따라서 학습 시작 시 `runs/esm2_variant_mvp/embedding_cache/`에 frozen ESM feature를 한 번 저장하고, epoch loop에서는 classifier head만 학습합니다. 디스크 cache 없이 매번 ESM forward를 실행하려면 `--embedding_cache off`를 추가합니다.

컬럼명이 기본값과 다르면 다음처럼 지정합니다.

```bash
python code/train_esm_classifier.py \
  --train_csv data/train.csv \
  --wt_col wild_type_sequence \
  --mut_col mutant_sequence \
  --position_col mutation_position \
  --label_col gof_label \
  --output_dir runs/custom_columns
```

## 4. 산출물 확인

```text
runs/esm2_variant_mvp/
├── best_checkpoint.pt
├── last_checkpoint.pt
├── config.json
├── metrics.json
└── embedding_cache/
    ├── train.pt
    └── val.pt
```

| 파일 | 의미 |
|---|---|
| `config.json` | 실행 인자, label mapping, 데이터 크기 |
| `metrics.json` | epoch별 train loss와 validation metric |
| `embedding_cache/*.pt` | frozen ESM feature cache와 metadata. `--embedding_cache off`일 때는 생성되지 않음 |
| `best_checkpoint.pt` | validation AUPRC 기준 best checkpoint, AUPRC 계산 불가 시 validation loss 기준 |
| `last_checkpoint.pt` | 마지막 epoch checkpoint |

## 5. Smoke Test

실제 CSV 없이 코드 경로만 확인하려면 mock 데이터를 명시적으로 사용합니다.

```bash
python code/train_esm_classifier.py \
  --use_mock_data \
  --epochs 1 \
  --batch_size 2 \
  --max_len 64 \
  --output_dir /tmp/pvc-smoke
```

이 경로도 Hugging Face model download가 필요합니다. 빠른 검증이 목적이면 `--model_name`에 작은 ESM 계열 모델을 지정합니다.

## 관련 문서

- [reference/configuration.md](../reference/configuration.md)
- [explanation/evaluation-validity.md](../explanation/evaluation-validity.md)
