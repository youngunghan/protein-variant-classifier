# ESM2 Position Feature Ablation - 2026-06-14

> **Scope:** Run additional 650M ESM2 synthetic experiments to compare per-residue `position_col` features against CLS fallback features, and verify frozen embedding cache reuse behavior.

## Result

Status: **passed with one implementation fix**

Findings:

- `position_col` per-residue features learned the synthetic mutation task quickly.
- CLS fallback had perfect ranking metrics on the two-row validation set, but needed more optimization to move the logits across the default 0.5 decision threshold.
- The first cache-reuse run revealed that the code still loaded the 650M ESM model before checking whether feature caches already existed.
- The cache path was fixed so valid caches are loaded before `EsmModel.from_pretrained()` is called.

## Fixture

Committed fixture files:

- `tests/fixtures/esm2_position_ablation_train.csv`
- `tests/fixtures/esm2_position_ablation_val.csv`

Run root for generated artifacts: `runs/esm2_650m_ablation_20260614`

Synthetic records:

| Variant | Label | Position | Count in train | Count in val |
|---|---:|---:|---:|---:|
| `D26G` | 0 | 26 | 8 | 1 |
| `L25K` | 1 | 25 | 4 | 1 |

This fixture is intentionally tiny. It tests feature-path behavior, not biological generalization.

## Commands

Position-aware run:

```bash
python code/train_esm_classifier.py \
  --train_csv tests/fixtures/esm2_position_ablation_train.csv \
  --val_csv tests/fixtures/esm2_position_ablation_val.csv \
  --position_col position \
  --epochs 20 \
  --batch_size 1 \
  --max_len 64 \
  --seed 13 \
  --output_dir runs/esm2_650m_ablation_20260614/position_seed13 \
  --embedding_cache on
```

CLS fallback run:

```bash
python code/train_esm_classifier.py \
  --train_csv tests/fixtures/esm2_position_ablation_train.csv \
  --val_csv tests/fixtures/esm2_position_ablation_val.csv \
  --epochs 20 \
  --batch_size 1 \
  --max_len 64 \
  --seed 13 \
  --output_dir runs/esm2_650m_ablation_20260614/cls_seed13 \
  --embedding_cache on
```

Cache reuse after fix:

```bash
python code/train_esm_classifier.py \
  --train_csv tests/fixtures/esm2_position_ablation_train.csv \
  --val_csv tests/fixtures/esm2_position_ablation_val.csv \
  --position_col position \
  --epochs 20 \
  --batch_size 1 \
  --max_len 64 \
  --seed 19 \
  --output_dir runs/esm2_650m_ablation_20260614/position_seed19_reuse_after_fix \
  --embedding_cache on \
  --embedding_cache_dir runs/esm2_650m_ablation_20260614/position_seed13/embedding_cache
```

CLS longer optimization run:

```bash
python code/train_esm_classifier.py \
  --train_csv tests/fixtures/esm2_position_ablation_train.csv \
  --val_csv tests/fixtures/esm2_position_ablation_val.csv \
  --epochs 100 \
  --batch_size 1 \
  --max_len 64 \
  --seed 13 \
  --lr 0.001 \
  --output_dir runs/esm2_650m_ablation_20260614/cls_seed13_lr1e3_epochs100_reuse \
  --embedding_cache on \
  --embedding_cache_dir runs/esm2_650m_ablation_20260614/cls_seed13/embedding_cache
```

## Metrics

| Run | Epoch | Train loss | Val loss | Val accuracy | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Position seed 13 | 1 | 0.5532766456 | 0.4009480625 | 1.0000 | 1.0000 | 1.0000 |
| Position seed 13 | 5 | 0.0628333815 | 0.0537395589 | 1.0000 | 1.0000 | 1.0000 |
| Position seed 13 | 20 | 0.0044850260 | 0.0044569053 | 1.0000 | 1.0000 | 1.0000 |
| Position seed 19, cache reuse after fix | 1 | 0.4901319183 | 0.3380338624 | 1.0000 | 1.0000 | 1.0000 |
| Position seed 19, cache reuse after fix | 5 | 0.0413713637 | 0.0418099137 | 1.0000 | 1.0000 | 1.0000 |
| Position seed 19, cache reuse after fix | 20 | 0.0034304156 | 0.0037881669 | 1.0000 | 1.0000 | 1.0000 |
| CLS seed 13 | 1 | 0.6966265465 | 0.7097883224 | 0.5000 | 1.0000 | 1.0000 |
| CLS seed 13 | 5 | 0.6420424655 | 0.7385011911 | 0.5000 | 1.0000 | 1.0000 |
| CLS seed 13 | 20 | 0.6088363156 | 0.7035805136 | 0.5000 | 1.0000 | 1.0000 |
| CLS seed 13, lr 1e-3 | 20 | 0.5222528564 | 0.6143892854 | 0.5000 | 1.0000 | 1.0000 |
| CLS seed 13, lr 1e-3 | 27 | 0.5187117346 | 0.5255483985 | 1.0000 | 1.0000 | 1.0000 |
| CLS seed 13, lr 1e-3 | 100 | 0.0307822810 | 0.0223531574 | 1.0000 | 1.0000 | 1.0000 |

Interpretation:

- Position-aware features move the synthetic fixture across the default 0.5 decision threshold much faster.
- CLS fallback is not a failed ranker in this toy setup: AUROC/AUPRC are already 1.0 because the two validation examples are correctly ordered.
- The CLS gap at 20 epochs is therefore a calibration/optimization-speed signal, not evidence of lower generalization capacity. With `lr=1e-3`, CLS reaches validation accuracy 1.0 at epoch 27 and low loss by epoch 100.
- With only two validation examples, neither run supports a biological generalization claim.

## Cache Reuse Behavior

| Run | Cache behavior | Pooler warning | Elapsed | Max RSS |
|---|---|---:|---:|---:|
| Position seed 13 | build cache | yes | 10.03s | 3,381,948 KB |
| Position seed 17 before fix | reuse cache but still load ESM model | yes | 8.43s | 3,381,868 KB |
| Position seed 19 after fix | reuse cache and skip ESM model load | no | 6.53s | 1,189,564 KB |
| CLS seed 13 lr 1e-3, 100 epochs after lazy-load fix | reuse cache and skip ESM model load | no | 14.12s | 1,189,584 KB |

The fix changes cache mode to compute `feature_dim` from `AutoConfig` first, then try cache metadata before instantiating `EsmModel`. If both train and validation caches are valid, the run trains the head without loading the 650M backbone. The tokenizer is also loaded lazily only when sequence precompute is needed.

## Code Change From Experiment

`code/train_esm_classifier.py` now:

- imports `AutoConfig`,
- adds `feature_dim_for_model(model_name)`,
- tries to load valid cached feature datasets before building the frozen ESM encoder,
- only instantiates `ESM2VariantClassifier` when one or more required cache files are missing or stale,
- loads the tokenizer lazily so cache-hit runs do not need sequence tokenization setup,
- validates that a provided `position` identifies a wt/mut token difference after tokenization,
- bumps the cache metadata version so caches built before the position-token guard are not silently reused.

## Conclusion

This ablation should be read as an implementation illustration, not a generalization proof:

- It illustrates the CLS-dilution mechanism: in this tiny fixture, the CLS fallback keeps the two examples correctly ranked but needs many more head-training steps to cross the default threshold.
- It is consistent with the domain argument for per-residue variant features, but it does not prove that position features generalize better on real GOF/LOF data.
- The stronger scientific rationale for position-aware features comes from the mutation-local nature of the task and from ESM variant-effect protocols that score mutated positions, not from this two-pattern synthetic ablation alone.
- Frozen feature caching is useful, and cache reuse should avoid loading the 650M model at all; the implementation now does that.

This is still not a real GOF/LOF performance result. The fixture has two repeated mutation patterns, so it proves feature-path behavior and implementation correctness, not generalization.
