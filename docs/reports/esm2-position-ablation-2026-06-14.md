# ESM2 Position Feature Ablation - 2026-06-14

> **Scope:** Run additional 650M ESM2 experiments on the assignment-provided tiny CSV fixture to exercise the per-residue `position_col` feature path, the CLS fallback path, and frozen embedding cache reuse behavior.

## Result

Status: **passed as an implementation smoke, not a controlled ablation**

Findings:

- `position_col` per-residue features crossed the default 0.5 decision threshold quickly on this assignment fixture.
- The validation rows are duplicated from train; all validation metrics below are in-sample only.
- CLS fallback had AUROC/AUPRC 1.0 on the two-row validation set, but its loss stayed near `ln(2)` at 20 epochs.
- The observed CLS behavior is consistent with a harder/noisier feature path, but this fixture does not isolate CLS dilution from learning-rate, conditioning, seed, or data-design effects.
- The first cache-reuse run revealed that the code still loaded the 650M ESM model before checking whether feature caches already existed.
- The cache path was fixed so valid caches are loaded before `EsmModel.from_pretrained()` is called.

## Fixture

Committed fixture files:

- `tests/fixtures/esm2_position_ablation_train.csv`
- `tests/fixtures/esm2_position_ablation_val.csv`

Run root for generated artifacts: `runs/esm2_650m_ablation_20260614`

Assignment fixture records:

| Variant | Label | Position | Count in train | Count in val |
|---|---:|---:|---:|---:|
| `D26G` | 0 | 26 | 8 | 1 |
| `L25K` | 1 | 25 | 4 | 1 |

This assignment-provided fixture is intentionally tiny. It tests feature-path behavior, not biological generalization. The validation rows are byte-for-byte duplicates of the two mutation patterns present in train, so `val` should be read as an in-sample smoke split.

Important confounders:

- `val` is a subset of train. Frozen ESM features for validation are therefore the same repeated feature vectors already seen during head training.
- The label is perfectly collinear with mutation pattern, position, original residue, mutant residue, and sequence context (`D26G @ 26 -> 0`, `L25K @ 25 -> 1`).
- The longer CLS run changes learning rate to `1e-3`, while the position runs remain at `1e-4`. It shows that CLS can eventually fit this assignment fixture, not a controlled feature-only comparison.
- With `n_val=2` and one positive/one negative, AUROC 1.0 is a single ordered pair, not statistical evidence of generalization.

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

- Position-aware features move this leaked, two-pattern assignment fixture across the default 0.5 decision threshold much faster at `lr=1e-4`.
- CLS fallback is not a failed ranker in this tiny setup: AUROC/AUPRC are 1.0 because the two validation examples are correctly ordered.
- However, CLS validation loss remains near `0.70` through 20 epochs, so the ranking metrics are near-degenerate here. They should not be read as capacity or calibration evidence.
- The `lr=1e-3` CLS run reaches validation accuracy 1.0 at epoch 27 and low loss by epoch 100, but it is not a controlled position-vs-CLS ablation because learning rate changed only for the CLS arm.
- With only two duplicated validation examples, neither run supports a biological generalization claim or a controlled claim that position pooling is empirically superior to CLS on real GOF/LOF data.

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
- adds lightweight model config metadata for feature dimension and resolved Hugging Face commit hash,
- tries to load valid cached feature datasets before building the frozen ESM encoder,
- only instantiates `ESM2VariantClassifier` when one or more required cache files are missing or stale,
- loads the tokenizer lazily so cache-hit runs do not need sequence tokenization setup,
- disables the unused ESM pooler when loading `EsmModel`,
- validates that a provided `position` identifies a wt/mut token difference after tokenization,
- enforces substitution-only CSV rows when `--position_col` is provided,
- stores requested revision, resolved model/tokenizer commit hash, and truncation policy in cache metadata,
- loads feature caches with `torch.load(..., weights_only=True)`,
- seeds explicit `RandomSampler` and `DistributedSampler`,
- bumps the cache metadata version so older feature caches are not silently reused.

## Conclusion

This ablation should be read as an implementation illustration, not a generalization proof:

- It is consistent with the CLS-dilution concern: in this tiny assignment fixture, the CLS fallback keeps the two examples ordered but needs many more head-training steps to cross the default threshold.
- It does not isolate that mechanism. Learning rate, feature conditioning, seed, and the tiny fixture design are all plausible explanations.
- It is consistent with the domain argument for per-residue variant features, but it does not prove that position features generalize better on real GOF/LOF data.
- The stronger scientific rationale for position-aware features comes from the mutation-local nature of the task and from ESM variant-effect protocols that score mutated positions, not from this two-pattern assignment fixture alone.
- Frozen feature caching is useful, and cache reuse should avoid loading the 650M model at all; the implementation now does that.

This is still not a real GOF/LOF performance result. The fixture has two repeated mutation patterns and validation rows duplicated from train, so it proves feature-path behavior and implementation correctness, not generalization.
