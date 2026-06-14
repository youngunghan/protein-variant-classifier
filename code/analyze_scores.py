from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics import binary_average_precision, binary_roc_auc, format_metric, has_both_binary_classes

DEFAULT_SCORE_COLS = ["SCORE_A", "SCORE_B", "SCORE_C"]


def parse_label(value: str, line_number: int, label_col: str) -> int:
    try:
        label = int(value)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: {label_col} must be 0 or 1, got {value!r}") from exc
    if label not in (0, 1):
        raise ValueError(f"Line {line_number}: {label_col} must be 0 or 1, got {value!r}")
    return label


def load_score_rows(
    file_path: str,
    patient_col: str,
    label_col: str,
    score_cols: list[str],
) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {patient_col, label_col, *score_cols}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{file_path} is missing required columns: {', '.join(missing)}")

        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            patient_id = (row.get(patient_col) or "").strip()
            if not patient_id:
                raise ValueError(f"Line {line_number}: {patient_col} must not be empty")

            parsed_scores: dict[str, float] = {}
            for score_col in score_cols:
                try:
                    score = float(row.get(score_col) or "")
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_number}: {score_col} must be numeric, got {row.get(score_col)!r}"
                    ) from exc
                if not math.isfinite(score):
                    raise ValueError(f"Line {line_number}: {score_col} must be finite, got {row.get(score_col)!r}")
                parsed_scores[score_col] = score

            rows.append(
                {
                    "patient_id": patient_id,
                    "label": parse_label(row.get(label_col) or "", line_number, label_col),
                    "scores": parsed_scores,
                }
            )

    if not rows:
        raise ValueError(f"{file_path} contains no score rows")
    return rows


def patient_hit_at_k(group: list[dict[str, Any]], score_col: str, k: int) -> bool:
    if not group:
        return False
    cutoff_index = min(k, len(group)) - 1
    cutoff_score = sorted((row["scores"][score_col] for row in group), reverse=True)[cutoff_index]
    return any(row["label"] == 1 and row["scores"][score_col] >= cutoff_score for row in group)


def analyze_scores(
    file_path: str,
    patient_col: str = "Patient_ID",
    label_col: str = "LABEL",
    score_cols: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    predictors = score_cols or DEFAULT_SCORE_COLS
    rows = load_score_rows(file_path, patient_col, label_col, predictors)
    labels = [row["label"] for row in rows]
    results: dict[str, dict[str, float | None]] = {}

    patient_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patient_groups[row["patient_id"]].append(row)

    print(f"Total patients: {len(patient_groups)}")
    print(f"Total variants: {len(rows)}")

    print("\n--- Global Metrics ---")
    for pred in predictors:
        scores = [row["scores"][pred] for row in rows]
        auroc = binary_roc_auc(labels, scores) if has_both_binary_classes(labels) else None
        auprc = binary_average_precision(labels, scores) if has_both_binary_classes(labels) else None
        results[pred] = {"AUROC": auroc, "AUPRC": auprc}
        print(f"{pred}: AUROC = {format_metric(auroc)}, AUPRC = {format_metric(auprc)}")

    print("\n--- Patient-Centric Metrics (Ranking) ---")
    ranking_results = {pred: {"Top-1": 0, "Top-5 Hit": 0} for pred in predictors}
    total_patients_with_pathogenic = 0
    skipped_patients = 0

    for group in patient_groups.values():
        if not any(row["label"] == 1 for row in group):
            skipped_patients += 1
            continue

        total_patients_with_pathogenic += 1
        for pred in predictors:
            if patient_hit_at_k(group, pred, 1):
                ranking_results[pred]["Top-1"] += 1
            if patient_hit_at_k(group, pred, 5):
                ranking_results[pred]["Top-5 Hit"] += 1

    print(f"Patients with at least one pathogenic variant: {total_patients_with_pathogenic}")
    print(f"Patients skipped because they have no pathogenic variant: {skipped_patients}")

    for pred in predictors:
        if total_patients_with_pathogenic == 0:
            top1_acc = None
            top5_hit = None
        else:
            top1_acc = ranking_results[pred]["Top-1"] / total_patients_with_pathogenic
            top5_hit = ranking_results[pred]["Top-5 Hit"] / total_patients_with_pathogenic
        results[pred]["Top-1 Accuracy"] = top1_acc
        results[pred]["Top-5 Hit"] = top5_hit
        print(
            f"{pred}: Top-1 Accuracy = {format_metric(top1_acc)}, "
            f"Top-5 Hit = {format_metric(top5_hit)}"
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze pathogenicity prediction scores")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing scores and labels")
    parser.add_argument("--patient_col", type=str, default="Patient_ID", help="Patient identifier column")
    parser.add_argument("--label_col", type=str, default="LABEL", help="Binary label column: 0=negative, 1=pathogenic")
    parser.add_argument("--score_cols", nargs="+", default=DEFAULT_SCORE_COLS, help="One or more score columns")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    analyze_scores(args.file_path, args.patient_col, args.label_col, args.score_cols)
