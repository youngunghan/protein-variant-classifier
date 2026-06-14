# Protein Variant Classifier with ESM2

ESM2 기반 protein variant **GOF/LOF** binary classifier입니다. 현재 구현은 wild-type sequence와 mutant sequence를 각각 ESM2로 임베딩한 뒤, `wt`, `mut`, `mut - wt` feature를 classifier head에 넣어 `0 = LOF`, `1 = GOF`를 예측합니다. `position` 컬럼을 제공하면 whole-sequence CLS 대신 단일 치환 residue 위치의 per-position embedding을 사용합니다.

## Key Features

- **CSV 기반 학습**: `wt_seq`, `mut_seq`, `label` 컬럼을 가진 실제 CSV를 입력으로 사용합니다.
- **Variant-Position Embedding**: 1-based `position` 컬럼이 있으면 single-substitution residue의 `E_mut - E_wt`를 feature로 사용합니다.
- **Frozen Embedding Cache**: frozen ESM backbone이면 기본적으로 `output_dir/embedding_cache`에 feature를 precompute하고 classifier head만 학습합니다.
- **Efficient Batching**: cache 생성 전 sequence pass에는 tokenization cache와 batch 내 최장 길이 기준 dynamic padding을 사용합니다.
- **Class Imbalance Handling**: train label 분포에서 class weight를 자동 계산합니다.
- **Reproducible Outputs**: checkpoint, config, metrics JSON을 run directory에 저장합니다.
- **DDP Support**: `torchrun` 기반 PyTorch `DistributedDataParallel` 실행을 유지합니다.

## Installation

```bash
conda create -n protein_classifier python=3.10
conda activate protein_classifier
pip install -r requirements.txt
```

## Data Format

최소 CSV 스키마:

```csv
wt_seq,mut_seq,label
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDILG,0
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDIKD,1
```

- `wt_seq`: wild-type protein sequence
- `mut_seq`: mutant protein sequence
- `label`: `0 = LOF`, `1 = GOF`

권장 optional 컬럼:

- `position`: 1-based single-substitution residue 위치. CLI에서 `--position_col position`을 지정하면 ESM의 해당 residue embedding을 사용합니다. 없으면 CLS embedding으로 fallback합니다. Indel/복합 변이는 현재 position path에서 거부합니다.

컬럼명이 다르면 CLI에서 `--wt_col`, `--mut_col`, `--label_col`, `--position_col`로 지정합니다.

## Training

Single GPU/CPU:

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

`position` 컬럼이 없는 CSV라면 `--position_col position` 줄을 빼면 CLS embedding fallback으로 실행됩니다.
기본값인 `--freeze_backbone --embedding_cache auto`에서는 첫 epoch 전에 ESM feature를 `runs/esm2_variant_mvp/embedding_cache/{train,val}.pt`로 저장하고, 이후에는 head-only classifier를 학습합니다. 매 epoch ESM을 다시 forward하고 싶으면 `--embedding_cache off`를 지정합니다.

Distributed training:

```bash
torchrun --nproc_per_node=4 code/train_esm_classifier.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --output_dir runs/esm2_variant_ddp \
  --batch_size 8 \
  --max_len 1024 \
  --epochs 10
```

Smoke test with synthetic data:

```bash
python code/train_esm_classifier.py \
  --use_mock_data \
  --epochs 1 \
  --batch_size 2 \
  --max_len 64 \
  --output_dir /tmp/pvc-smoke
```

Outputs:

- `config.json`: CLI args, label mapping, dataset sizes
- `metrics.json`: epoch-level train/validation metrics
- `embedding_cache/train.pt`, `embedding_cache/val.pt`: frozen ESM feature cache when enabled
- `best_checkpoint.pt`: best checkpoint by validation AUPRC, falling back to validation loss
- `last_checkpoint.pt`: final epoch checkpoint

## Score Analysis

Evaluate external pathogenicity scores with global AUROC/AUPRC and patient-level Hit@k ranking:

```bash
python code/analyze_scores.py path/to/scores.csv \
  --patient_col Patient_ID \
  --label_col LABEL \
  --score_cols SCORE_A SCORE_B SCORE_C
```

## Documentation

See [docs/README.md](docs/README.md) for the project documentation index.

Important caveat: validation scores are only meaningful relative to the split used. If variants from the same protein, gene, family, or patient leak across train/validation, performance can be overestimated. See [docs/explanation/evaluation-validity.md](docs/explanation/evaluation-validity.md).

## License

MIT License
