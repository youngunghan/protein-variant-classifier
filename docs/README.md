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
