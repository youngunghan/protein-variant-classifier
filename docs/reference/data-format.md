# Data Format

> **범위:** 학습 CSV와 score 분석 CSV의 컬럼 계약. 모델 구조나 평가 해석은 다루지 않는다.
> **대상:** 데이터 준비자, 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-14.

## 1. 학습 CSV

기본 컬럼 계약:

| 컬럼 | 타입 | 필수 | 설명 |
|---|---:|:---:|---|
| `wt_seq` | string | 예 | Wild-type protein sequence |
| `mut_seq` | string | 예 | Mutant protein sequence |
| `label` | int | 예 | `0 = LOF`, `1 = GOF` |
| `position` | int | 아니오 | 1-based mutation residue position. `--position_col`로 지정하면 per-residue embedding을 사용 |

예시:

```csv
wt_seq,mut_seq,label
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDILG,0
MKTAYIAKQRQISFVKSHFSRQDILD,MKTAYIAKQRQISFVKSHFSRQDIKD,1
```

컬럼명이 다르면 [configuration.md §1.1](configuration.md#11-데이터-입력) 옵션으로 매핑합니다.

`position`은 1부터 시작하는 원본 protein sequence residue index입니다. 제공된 위치가 sequence 길이 밖이면 CSV 로딩에서 거부하고, `--max_len` truncation으로 해당 token이 잘리면 학습 중 명시적 에러를 냅니다. 위치 컬럼을 지정하지 않으면 모델은 CLS embedding으로 fallback합니다.

## 2. Label

| 값 | 의미 |
|---:|---|
| `0` | Loss-of-Function (LOF) |
| `1` | Gain-of-Function (GOF) |

문자열 label은 MVP에서 지원하지 않습니다. `LOF`/`GOF` 같은 값은 전처리 단계에서 `0`/`1`로 변환합니다.

## 3. Score 분석 CSV

`analyze_scores.py` 기본 컬럼 계약:

| 컬럼 | 타입 | 필수 | 설명 |
|---|---:|:---:|---|
| `Patient_ID` | string | 예 | 환자 또는 case 식별자 |
| `LABEL` | int | 예 | `0 = negative`, `1 = pathogenic` |
| `SCORE_A` | float | 예 | 비교할 predictor score |
| `SCORE_B` | float | 예 | 비교할 predictor score |
| `SCORE_C` | float | 예 | 비교할 predictor score |

`SCORE_*` 컬럼은 `--score_cols`로 원하는 개수만큼 지정할 수 있습니다. Score 값은 finite float이어야 하며 `NaN`, `inf`, `-inf`는 거부합니다.

## 4. 보관 정책

임상·과제 제공 데이터, 원본 PDF, 민감 자료는 문서에 복사하지 않습니다. 이 repo 문서는 스키마와 실행 계약만 기록하고, 실제 데이터는 로컬 경로나 접근 권한이 통제된 저장소에 둡니다.
