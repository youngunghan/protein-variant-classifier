# protein-variant-classifier docs

ESM2 기반 GOF/LOF protein variant classifier의 실행, 데이터 계약, 설정, 평가 한계를 정리한 문서 허브입니다.

## 문서 목록 (Diátaxis)

### Tutorials

| 문서 | 설명 |
|---|---|
| [tutorials/quickstart.md](tutorials/quickstart.md) | CSV 파일로 1회 학습을 실행하고 산출물을 확인하는 happy path |

### Reference

| 문서 | 설명 |
|---|---|
| [reference/data-format.md](reference/data-format.md) | 학습 CSV와 score 분석 CSV의 컬럼 계약 |
| [reference/configuration.md](reference/configuration.md) | `train_esm_classifier.py`, `analyze_scores.py` CLI 옵션 |

### Explanation

| 문서 | 설명 |
|---|---|
| [explanation/evaluation-validity.md](explanation/evaluation-validity.md) | validation split, label imbalance, truncation 때문에 생기는 평가 해석 한계 |

### Reports

| 문서 | 설명 |
|---|---|
| [reports/esm2-650m-smoke-2026-06-14.md](reports/esm2-650m-smoke-2026-06-14.md) | 실제 ESM2 650M download/load/forward/cache/head-training smoke run 결과 |
| [reports/esm2-official-comparison-and-epoch-evaluation-2026-06-14.md](reports/esm2-official-comparison-and-epoch-evaluation-2026-06-14.md) | Meta/Hugging Face 공식 ESM2 자료와 현재 학습 경로 비교, epoch 판단 |
| [reports/esm2-position-ablation-2026-06-14.md](reports/esm2-position-ablation-2026-06-14.md) | `position_col`/CLS feature-path smoke, cache reuse 개선, 통제 ablation 한계 |
