"""Stdlib metric implementations for benchmark summaries."""

from __future__ import annotations

from collections import defaultdict

from fair_mia.types import AttackRecord


def auc_roc(labels: list[bool], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    sorted_pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        tie_end = index
        while tie_end + 1 < len(sorted_pairs) and sorted_pairs[tie_end + 1][0] == sorted_pairs[index][0]:
            tie_end += 1
        average_rank = (index + 1 + tie_end + 1) / 2.0
        for tie_index in range(index, tie_end + 1):
            if sorted_pairs[tie_index][1]:
                rank_sum += average_rank
        index = tie_end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def tpr_at_fpr(labels: list[bool], scores: list[float], max_fpr: float) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    best_tpr = 0.0
    thresholds = sorted(set(scores), reverse=True)
    for threshold in thresholds:
        true_positive = sum(1 for label, score in zip(labels, scores) if label and score >= threshold)
        false_positive = sum(1 for label, score in zip(labels, scores) if not label and score >= threshold)
        fpr = false_positive / negatives
        if fpr <= max_fpr:
            best_tpr = max(best_tpr, true_positive / positives)
    return best_tpr


def best_threshold_accuracy(labels: list[bool], scores: list[float]) -> float:
    if not labels:
        return 0.0
    thresholds = sorted(set(scores), reverse=True)
    candidates = thresholds + [min(thresholds) - 1.0] if thresholds else [0.0]
    best = 0.0
    for threshold in candidates:
        correct = sum(1 for label, score in zip(labels, scores) if (score >= threshold) == label)
        best = max(best, correct / len(labels))
    return best


def best_threshold_balanced_accuracy(labels: list[bool], scores: list[float]) -> float:
    if not labels:
        return 0.0
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    thresholds = sorted(set(scores), reverse=True)
    candidates = thresholds + [min(thresholds) - 1.0] if thresholds else [0.0]
    best = 0.0
    for threshold in candidates:
        true_positive = sum(1 for label, score in zip(labels, scores) if label and score >= threshold)
        true_negative = sum(1 for label, score in zip(labels, scores) if not label and score < threshold)
        tpr = true_positive / positives
        tnr = true_negative / negatives
        best = max(best, (tpr + tnr) / 2.0)
    return best


def majority_class_accuracy(labels: list[bool]) -> float:
    if not labels:
        return 0.0
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    return max(positives, negatives) / len(labels)


def privacy_leakage_disparity(group_metrics: dict[str, float], group_a: str = "G0", group_b: str = "G1") -> float:
    return abs(group_metrics.get(group_a, 0.0) - group_metrics.get(group_b, 0.0))


def summarize_records(records: list[AttackRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[AttackRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.scenario, record.attack)].append(record)

    rows: list[dict[str, object]] = []
    for (scenario, attack), attack_records in sorted(grouped.items()):
        labels = [record.is_member for record in attack_records]
        scores = [record.membership_score for record in attack_records]
        row: dict[str, object] = {
            "scenario": scenario,
            "attack": attack,
            "scope": "all",
            "samples": len(attack_records),
            "auc_roc": round(auc_roc(labels, scores), 6),
            "tpr_at_1_fpr": round(tpr_at_fpr(labels, scores, 0.01), 6),
            "tpr_at_0_1_fpr": round(tpr_at_fpr(labels, scores, 0.001), 6),
            "accuracy": round(best_threshold_accuracy(labels, scores), 6),
            "balanced_accuracy": round(best_threshold_balanced_accuracy(labels, scores), 6),
            "majority_class_accuracy": round(majority_class_accuracy(labels), 6),
            "pld_auc_roc": "",
            "pld_accuracy": "",
        }
        rows.append(row)

        per_group_auc: dict[str, float] = {}
        per_group_accuracy: dict[str, float] = {}
        for group in sorted({record.group for record in attack_records}):
            group_records = [record for record in attack_records if record.group == group]
            group_labels = [record.is_member for record in group_records]
            group_scores = [record.membership_score for record in group_records]
            group_auc = auc_roc(group_labels, group_scores)
            group_accuracy = best_threshold_accuracy(group_labels, group_scores)
            per_group_auc[group] = group_auc
            per_group_accuracy[group] = group_accuracy
            rows.append(
                {
                    "scenario": scenario,
                    "attack": attack,
                    "scope": group,
                    "samples": len(group_records),
                    "auc_roc": round(group_auc, 6),
                    "tpr_at_1_fpr": round(tpr_at_fpr(group_labels, group_scores, 0.01), 6),
                    "tpr_at_0_1_fpr": round(tpr_at_fpr(group_labels, group_scores, 0.001), 6),
                    "accuracy": round(group_accuracy, 6),
                    "balanced_accuracy": round(best_threshold_balanced_accuracy(group_labels, group_scores), 6),
                    "majority_class_accuracy": round(majority_class_accuracy(group_labels), 6),
                    "pld_auc_roc": "",
                    "pld_accuracy": "",
                }
            )
        rows.append(
            {
                "scenario": scenario,
                "attack": attack,
                "scope": "gap:G0-G1",
                "samples": "",
                "auc_roc": "",
                "tpr_at_1_fpr": "",
                "tpr_at_0_1_fpr": "",
                "accuracy": "",
                "balanced_accuracy": "",
                "majority_class_accuracy": "",
                "pld_auc_roc": round(privacy_leakage_disparity(per_group_auc), 6),
                "pld_accuracy": round(privacy_leakage_disparity(per_group_accuracy), 6),
            }
        )
    return rows
