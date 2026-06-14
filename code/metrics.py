from __future__ import annotations

from typing import Iterable


def has_both_binary_classes(labels: Iterable[int]) -> bool:
    values = set(labels)
    return 0 in values and 1 in values


def binary_roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    """Compute binary AUROC with average ranks for tied scores."""
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores)]
    n_pos = sum(label == 1 for _, label in pairs)
    n_neg = sum(label == 0 for _, label in pairs)
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        positives_in_tie = sum(label == 1 for _, label in pairs[idx:end])
        rank_sum_pos += positives_in_tie * avg_rank
        idx = end

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_average_precision(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    pairs = sorted(
        [(float(score), int(label)) for label, score in zip(labels, scores)],
        key=lambda item: item[0],
        reverse=True,
    )
    n_pos = sum(label == 1 for _, label in pairs)
    if n_pos == 0:
        return None

    precision_sum = 0.0
    true_positives = 0
    seen = 0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1

        positives_in_tie = sum(label == 1 for _, label in pairs[idx:end])
        true_positives += positives_in_tie
        seen = end
        if positives_in_tie:
            precision_sum += positives_in_tie * (true_positives / seen)
        idx = end

    return precision_sum / n_pos


def format_metric(value: float | None) -> str:
    if value is None:
        return "skipped"
    return f"{value:.4f}"
