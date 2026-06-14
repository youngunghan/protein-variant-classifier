# ESM2 Official Comparison and Epoch Evaluation - 2026-06-14

> **Scope:** Compare this repo's ESM2 usage against Meta/Hugging Face ESM2 guidance, then judge whether the one-epoch smoke run was enough training or only an execution check.

## Sources Checked

- Meta FAIR ESM GitHub README: <https://github.com/facebookresearch/esm>
- Meta FAIR ESM variant prediction example: <https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py>
- Hugging Face model card for `facebook/esm2_t33_650M_UR50D`: <https://huggingface.co/facebook/esm2_t33_650M_UR50D>
- Hugging Face Transformers ESM docs: <https://huggingface.co/docs/transformers/en/model_doc/esm>

## Official Guidance Summary

1. **ESM2 is general-purpose, not variant-specialized only.**
   Meta describes ESM2 as a general-purpose protein language model that can be used to predict structure, function, and other protein properties from individual sequences. Hugging Face describes the same checkpoint family as suitable for fine-tuning on tasks that take protein sequences as input.

2. **Variant effect examples in Meta's repo are zero-shot MLM-logit scoring, not a supervised GOF/LOF classifier.**
   The official `examples/variant-prediction/predict.py` path loads an ESM-family model, sets `model.eval()`, computes token log probabilities from model logits, and supports `wt-marginals` / `masked-marginals` style scoring. That is a mutation-effect scoring protocol, not the same as training a classifier head on GOF/LOF labels.

3. **ESM-1v is the explicitly variant-effect-specialized model family; ESM2 is expected to be usable for variant prediction.**
   Meta's README lists ESM-1v as specialized for zero-shot variant effects and notes that ESM2 can also be used for variant prediction. For a supervised GOF/LOF label task, using ESM2 embeddings plus a supervised head is defensible, but the evaluation protocol must be separate from zero-shot DMS scoring claims.

4. **Using token embeddings is aligned with the ESM design.**
   Hugging Face's ESM docs describe ESM models as masked-language-model protein transformers and ESMFold as relying on token embeddings from the ESM2 stem. This repo's `last_hidden_state` feature extraction is therefore a normal use of the backbone. The smoke run's pooler warning was not material because this repo does not use the pooler output; the implementation now disables the unused pooler at load time.

## Current Repo Alignment

| Area | Current repo | Official comparison | Judgment |
|---|---|---|---|
| Backbone | `facebook/esm2_t33_650M_UR50D` through Transformers `EsmModel` | Same checkpoint family documented by HF/Meta | OK |
| Objective | Supervised GOF/LOF head over frozen ESM features | HF says ESM2 is suitable for fine-tuning/downstream tasks | OK, but it is not Meta's zero-shot variant script |
| Variant localization | Optional `position` column uses per-residue `last_hidden_state` | More appropriate for point mutations than whole-sequence CLS | Good design choice |
| No-position fallback | CLS embedding fallback | Usable as generic sequence representation, but mutation signal can be diluted | Acceptable fallback, weaker scientific signal |
| Frozen cache | Precompute fixed ESM features, train `FeatureOnlyClassifier` | Feature extraction / linear-probe style, not full fine-tuning | Efficient, but lower capacity than full fine-tuning |
| Evaluation | AUROC/AUPRC/accuracy on provided val split | Official docs do not define GOF/LOF protocol | Repo must keep leakage/imbalance caveats |

## Smoke Runs Reviewed

### 1. Fresh-cache 650M execution smoke

Run directory: `runs/esm2_650m_smoke_fresh_20260614`

| Item | Value |
|---|---:|
| Epochs | 1 |
| `position_col` | not used |
| Train examples | 12 mock rows from `--use_mock_data` |
| Validation examples | 2 mock rows from `--use_mock_data` |
| Train loss | 0.6966265465 |
| Validation loss | 0.7097883224 |
| Validation accuracy | 0.5000 |
| Validation AUROC | 1.0000 |
| Validation AUPRC | 1.0000 |
| Elapsed including fresh HF cache | 166.65 seconds |

Interpretation:

- This proves model download/cache, CUDA load, wt/mut forward, embedding cache, head training, metrics, and checkpoint writing.
- It does **not** prove classifier training quality. Accuracy is only 0.5 at the default threshold after one epoch, and AUROC/AUPRC are inflated by a two-row mock validation set.

### 2. Position-aware 20-epoch head-training smoke

Run directory: `runs/esm2_650m_position_epochs20_20260614`

This run used the same tiny assignment-provided mutation patterns but with explicit `position` values:

- LOF: `D26G`, label `0`, position `26`
- GOF: `L25K`, label `1`, position `25`

| Epoch | Train loss | Validation loss | Validation accuracy | AUROC | AUPRC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5532766456 | 0.4009480625 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 0.0628333815 | 0.0537395589 | 1.0000 | 1.0000 | 1.0000 |
| 20 | 0.0044850260 | 0.0044569053 | 1.0000 | 1.0000 | 1.0000 |

Interpretation:

- The classifier head can learn from cached per-position ESM features.
- The rapid loss collapse is expected because the assignment train/validation sets contain repeated versions of only two mutation patterns.
- This is a functional training check, not a biological generalization check.

## Was One Epoch Enough?

No, not for a real supervised GOF/LOF classifier.

The one-epoch run should be treated only as an integration smoke test. It verifies that the pipeline works with the real 650M checkpoint, but it does not establish convergence, calibration, or generalization.

For real data:

- Use `--position_col` whenever mutation positions are available.
- Keep `--embedding_cache auto` or `on` when the ESM backbone is frozen.
- Start with `--epochs 20` to `50` for head-only training because cached ESM features make epochs cheap.
- Select the checkpoint by validation AUPRC when both classes exist, as the current script already does.
- Stop based on validation loss/AUPRC, not training loss alone.
- Use protein/gene/family/patient-disjoint validation where possible; random row split is likely optimistic.

## Was It "Trained Correctly"?

Pipeline correctness: **yes for a smoke/integration check**.

The following paths were exercised successfully:

- Official 650M checkpoint download/cache.
- CUDA ESM forward over wild-type and mutant sequences.
- Frozen feature cache generation.
- Head-only classifier training.
- Position-aware per-residue feature path in the 20-epoch assignment-fixture run.
- Checkpoint and metrics writing.

Scientific training sufficiency: **not yet**.

The current runs do not answer whether the model learns GOF/LOF biology. They use repeated assignment examples and tiny validation sets. A valid training claim requires a larger disjoint labeled dataset, group-aware splits, and comparison to baselines such as simple substitution scores or ESM zero-shot log-likelihood ratios.

## Recommended Next Experiment

For the actual dataset, run:

```bash
python code/train_esm_classifier.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --position_col position \
  --output_dir runs/esm2_variant_real_position_head \
  --batch_size 1 \
  --max_len 512 \
  --epochs 30 \
  --embedding_cache on
```

Then inspect:

- validation AUPRC trend across epochs,
- validation loss minimum,
- calibration at the 0.5 threshold,
- class-specific confusion matrix,
- whether train/validation share proteins, genes, families, or patients.

If validation AUPRC plateaus early, fewer epochs are enough. If training loss falls while validation AUPRC falls or validation loss rises, stop earlier or add regularization. If the head underfits with enough real data, consider unfreezing the last ESM layers or using LoRA/adapters rather than full 650M fine-tuning on an 8GB GPU.
