# Evaluation Validity

> **범위:** protein variant classifier 성능 수치를 해석할 때 필요한 split, imbalance, truncation 한계.
> **대상:** 모델 개발자, 결과 리뷰어.
> **상태:** 구현 반영 + 운영 주의 — 기준일 2026-06-14.

## 1. 현재 metric이 말하는 것

`train_esm_classifier.py`는 validation CSV가 있을 때 loss, accuracy, AUROC, AUPRC를 출력합니다. 이 값은 **제공된 validation split 기준 성능**입니다. 독립 데이터 일반화 성능이라고 쓰려면 split 기준이 그 주장을 뒷받침해야 합니다.

## 2. Split Leakage 위험

Protein variant 문제에서는 무작위 row split이 성능을 과대평가할 수 있습니다.

| 누수 단위 | 위험 |
|---|---|
| 같은 protein | 비슷한 wild-type context를 train과 validation이 공유 |
| 같은 gene/family | homologous sequence pattern을 공유 |
| 같은 patient/case | case-specific artifact를 공유 |
| 같은 source dataset | label policy나 measurement bias를 공유 |

가능하면 validation은 protein, gene, family, patient 중 문제 정의에 맞는 group 기준으로 분리합니다. MVP CLI는 이미 분리된 `--train_csv`, `--val_csv`를 받으며, split 생성 자체는 데이터 준비 단계의 책임입니다.

## 3. Label Imbalance

GOF/LOF 데이터는 한쪽 label이 적을 수 있습니다. 이 repo는 `--class_weight auto`로 train label 분포 기반 weighted cross entropy를 사용합니다.

해석상 주의:

| 지표 | 주의점 |
|---|---|
| accuracy | 다수 클래스 예측만으로 높아질 수 있음 |
| AUROC | ranking 전반을 보지만 positive가 적을 때 불확실성이 커짐 |
| AUPRC | positive class가 희소할 때 더 민감하므로 best checkpoint 기준으로 사용 |

validation set이 한 클래스만 포함하면 AUROC/AUPRC는 정의되지 않으므로 `skipped`로 처리합니다.

## 4. Sequence Truncation

ESM2 입력은 `--max_len`으로 잘립니다. `--position_col`을 제공하면 모델은 single-substitution 변이의 해당 residue token embedding을 사용하고, 그 token이 truncation으로 잘리면 명시적으로 에러를 냅니다. 현재 position-aware 경로는 insertion/deletion이나 복합 변이를 지원하지 않습니다. 위치 컬럼을 쓰지 않는 경우에는 CLS embedding으로 fallback하므로 단일 변이 신호가 whole-sequence representation에서 덜 직접적으로 표현될 수 있습니다. 이 가능성만으로 position feature의 일반화 우위를 입증할 수는 없고, split과 교란 통제가 필요합니다.

| 상황 | 권장 조치 |
|---|---|
| 변이가 N-terminal 근처 | 현재 full sequence truncation으로도 영향이 남을 가능성이 큼 |
| 변이가 `max_len` 이후 | variant-centered window 생성 검토 |
| 긴 protein이 많음 | `max_len` 분포와 truncation 비율을 별도 리포트 |

현재 구현은 full sequence를 tokenizer truncation에 맡기되, position-aware 학습에서는 잘린 변이 위치를 실패로 처리합니다. 긴 protein이 많으면 position-centered windowing을 데이터 전처리 단계나 별도 모델 입력 경로로 추가하는 편이 안전합니다.

## 5. 정직한 결과 표현

권장 표현:

- "제공한 validation CSV 기준 AUROC/AUPRC"
- "split 기준은 `<protein-disjoint | random-row | patient-disjoint>`"
- "같은 protein/gene/family가 train/validation에 섞였을 가능성이 있으면 일반화 성능은 별도 검증 필요"

피해야 할 표현:

- "임상 일반화 성능"
- "새 protein family에도 검증됨"
- "GOF/LOF를 일반적으로 구분함"

위 표현은 별도 외부 validation이나 group-disjoint 실험이 있을 때만 사용합니다.
