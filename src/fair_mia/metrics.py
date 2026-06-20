"""Dependency-free privacy, calibration, and subgroup metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from fair_mia.types import AttackRecord


def auc_roc(labels: list[bool], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    positives = sum(labels)
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
        rank_sum += average_rank * sum(1 for _, label in sorted_pairs[index : tie_end + 1] if label)
        index = tie_end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auc_pr(labels: list[bool], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in ordered:
        if label:
            true_positive += 1
        else:
            false_positive += 1
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def tpr_at_fpr(labels: list[bool], scores: list[float], max_fpr: float) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    best = 0.0
    for threshold in sorted(set(scores), reverse=True):
        tp, fp, _, _ = confusion(labels, scores, threshold)
        if fp / negatives <= max_fpr:
            best = max(best, tp / positives)
    return best


def fpr_at_tpr(labels: list[bool], scores: list[float], min_tpr: float) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    best = 1.0
    for threshold in sorted(set(scores), reverse=True):
        tp, fp, _, _ = confusion(labels, scores, threshold)
        if tp / positives >= min_tpr:
            best = min(best, fp / negatives)
    return best


def confusion(labels: list[bool], scores: list[float], threshold: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def select_threshold(labels: list[bool], scores: list[float]) -> float:
    if not scores:
        return 0.0
    candidates = sorted(set(scores), reverse=True) + [min(scores) - 1.0]
    return max(
        candidates,
        key=lambda threshold: (
            balanced_accuracy_at_threshold(labels, scores, threshold),
            accuracy_at_threshold(labels, scores, threshold),
            threshold,
        ),
    )


def accuracy_at_threshold(labels: list[bool], scores: list[float], threshold: float) -> float:
    if not labels:
        return 0.0
    tp, fp, tn, fn = confusion(labels, scores, threshold)
    return (tp + tn) / max(tp + fp + tn + fn, 1)


def balanced_accuracy_at_threshold(labels: list[bool], scores: list[float], threshold: float) -> float:
    tp, fp, tn, fn = confusion(labels, scores, threshold)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2.0


def best_threshold_accuracy(labels: list[bool], scores: list[float]) -> float:
    return accuracy_at_threshold(labels, scores, select_threshold(labels, scores))


def best_threshold_balanced_accuracy(labels: list[bool], scores: list[float]) -> float:
    return balanced_accuracy_at_threshold(labels, scores, select_threshold(labels, scores))


def majority_class_accuracy(labels: list[bool]) -> float:
    if not labels:
        return 0.0
    positives = sum(labels)
    return max(positives, len(labels) - positives) / len(labels)


def precision_recall_f1_mcc(
    labels: list[bool],
    scores: list[float],
    threshold: float,
) -> tuple[float, float, float, float]:
    tp, fp, tn, fn = confusion(labels, scores, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return precision, recall, f1, mcc


def _probabilities(scores: list[float]) -> list[float]:
    if not scores:
        return []
    ordered = sorted((score, index) for index, score in enumerate(scores))
    result = [0.0] * len(scores)
    denominator = max(len(scores) - 1, 1)
    for rank, (_, index) in enumerate(ordered):
        result[index] = rank / denominator
    return result


def brier_score(labels: list[bool], scores: list[float]) -> float:
    probabilities = _probabilities(scores)
    if not probabilities:
        return 0.0
    return sum((probability - float(label)) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(labels: list[bool], scores: list[float], bins: int = 10) -> float:
    probabilities = _probabilities(scores)
    if not probabilities:
        return 0.0
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if (
                lower <= probability <= upper
                if bin_index == bins - 1
                else lower <= probability < upper
            )
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(indices)
        accuracy = sum(float(labels[index]) for index in indices) / len(indices)
        error += len(indices) / len(labels) * abs(confidence - accuracy)
    return error


def privacy_leakage_disparity(
    group_metrics: dict[str, float],
    group_a: str = "G0",
    group_b: str = "G1",
) -> float:
    return abs(group_metrics.get(group_a, 0.0) - group_metrics.get(group_b, 0.0))


def _author_id(record: AttackRecord) -> str:
    return str(record.attributes.get("author_id") or record.sample_id.split(":w", 1)[0])


def clustered_bootstrap_auc(
    records: list[AttackRecord],
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    point = auc_roc([record.is_member for record in records], [record.membership_score for record in records])
    if replicates <= 0:
        return point, point
    by_author: dict[str, list[AttackRecord]] = defaultdict(list)
    for record in records:
        by_author[_author_id(record)].append(record)
    authors = sorted(by_author)
    if len(authors) < 2:
        return point, point
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled_authors = [rng.choice(authors) for _ in authors]
        sampled = [record for author in sampled_authors for record in by_author[author]]
        labels = [record.is_member for record in sampled]
        if not any(labels) or all(labels):
            continue
        values.append(auc_roc(labels, [record.membership_score for record in sampled]))
    if not values:
        return 0.5, 0.5
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _scope_definitions(records: list[AttackRecord]) -> list[tuple[str, str, Callable[[AttackRecord], bool]]]:
    definitions: list[tuple[str, str, Callable[[AttackRecord], bool]]] = [("all", "all", lambda _: True)]
    for group in sorted({record.group for record in records}):
        definitions.append(("gender", group, lambda record, value=group: record.group == value))
    for attribute in ("language", "variety"):
        for value in sorted({record.attributes.get(attribute, "") for record in records} - {""}):
            definitions.append(
                (attribute, value, lambda record, key=attribute, expected=value: record.attributes.get(key) == expected)
            )
    intersections = sorted(
        {
            (record.group, record.attributes.get("language", ""))
            for record in records
            if record.attributes.get("language")
        }
    )
    for group, language in intersections:
        definitions.append(
            (
                "gender_language",
                f"{group}|{language}",
                lambda record, expected_group=group, expected_language=language: (
                    record.group == expected_group and record.attributes.get("language") == expected_language
                ),
            )
        )
    return definitions


def summarize_records(
    records: list[AttackRecord],
    calibration_records: list[AttackRecord] | None = None,
    *,
    min_cell_members: int = 30,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
) -> list[dict[str, object]]:
    calibration_records = calibration_records or []
    grouped: dict[tuple[str, str], list[AttackRecord]] = defaultdict(list)
    calibration_grouped: dict[tuple[str, str], list[AttackRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.scenario, record.attack)].append(record)
    for record in calibration_records:
        calibration_grouped[(record.scenario, record.attack)].append(record)

    rows: list[dict[str, object]] = []
    for key, attack_records in sorted(grouped.items()):
        scenario, attack = key
        calibration_attack_records = calibration_grouped.get(key, [])
        gender_auc: dict[str, float] = {}
        dimension_auc: dict[str, dict[str, float]] = defaultdict(dict)
        for dimension, scope, predicate in _scope_definitions(attack_records):
            scoped = [record for record in attack_records if predicate(record)]
            member_count = sum(record.is_member for record in scoped)
            nonmember_count = len(scoped) - member_count
            suppressed = dimension != "all" and (
                member_count < min_cell_members or nonmember_count < min_cell_members
            )
            labels = [record.is_member for record in scoped]
            scores = [record.membership_score for record in scoped]
            calibration_scoped = [record for record in calibration_attack_records if predicate(record)]
            calibration_labels = [record.is_member for record in calibration_scoped]
            calibration_scores = [record.membership_score for record in calibration_scoped]
            threshold_source = "calibration" if calibration_scores else "oracle_test"
            threshold = select_threshold(
                calibration_labels if calibration_scores else labels,
                calibration_scores if calibration_scores else scores,
            )
            if suppressed or not scoped:
                row = _empty_metric_row(scenario, attack, dimension, scope, member_count, nonmember_count)
                row["suppressed"] = True
                rows.append(row)
                continue
            precision, recall, f1, mcc = precision_recall_f1_mcc(labels, scores, threshold)
            auc_value = auc_roc(labels, scores)
            ci_low, ci_high = clustered_bootstrap_auc(
                scoped,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            perplexity_values = [
                math.exp(min(float(record.diagnostics["target_loss"]), 20.0))
                for record in scoped
                if "target_loss" in record.diagnostics
            ]
            loss_values = [
                float(record.diagnostics["target_loss"])
                for record in scoped
                if "target_loss" in record.diagnostics
            ]
            row = {
                "scenario": scenario,
                "attack": attack,
                "dimension": dimension,
                "scope": scope,
                "samples": len(scoped),
                "members": member_count,
                "nonmembers": nonmember_count,
                "suppressed": False,
                "threshold": round(threshold, 8),
                "threshold_source": threshold_source,
                "auc_roc": round(auc_value, 6),
                "auc_ci_low": round(ci_low, 6),
                "auc_ci_high": round(ci_high, 6),
                "auc_pr": round(auc_pr(labels, scores), 6),
                "tpr_at_0_1_fpr": round(tpr_at_fpr(labels, scores, 0.001), 6),
                "tpr_at_1_fpr": round(tpr_at_fpr(labels, scores, 0.01), 6),
                "tpr_at_5_fpr": round(tpr_at_fpr(labels, scores, 0.05), 6),
                "fpr_at_90_tpr": round(fpr_at_tpr(labels, scores, 0.90), 6),
                "fpr_at_95_tpr": round(fpr_at_tpr(labels, scores, 0.95), 6),
                "fpr_at_99_tpr": round(fpr_at_tpr(labels, scores, 0.99), 6),
                "accuracy": round(accuracy_at_threshold(labels, scores, threshold), 6),
                "balanced_accuracy": round(balanced_accuracy_at_threshold(labels, scores, threshold), 6),
                "majority_class_accuracy": round(majority_class_accuracy(labels), 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "mcc": round(mcc, 6),
                "attack_advantage": round(2.0 * auc_value - 1.0, 6),
                "brier": round(brier_score(labels, scores), 6),
                "ece": round(expected_calibration_error(labels, scores), 6),
                "mean_loss": round(sum(loss_values) / len(loss_values), 6) if loss_values else "",
                "perplexity": round(sum(perplexity_values) / len(perplexity_values), 6) if perplexity_values else "",
                "pld_auc_roc": "",
                "macro_auc_roc": "",
                "worst_group_auc_roc": "",
                "max_pairwise_gap": "",
                "disparity_ratio": "",
            }
            if dimension == "gender":
                gender_auc[scope] = auc_value
            if dimension in {"gender", "language", "variety", "gender_language"}:
                dimension_auc[dimension][scope] = auc_value
            rows.append(row)
        for dimension in ("language", "variety", "gender_language"):
            values = dimension_auc.get(dimension, {})
            if not values:
                continue
            metric_values = list(values.values())
            summary_row = _empty_metric_row(
                scenario,
                attack,
                f"{dimension}_summary",
                f"summary:{dimension}",
                0,
                0,
            )
            summary_row.update(
                {
                    "suppressed": False,
                    "macro_auc_roc": round(sum(metric_values) / len(metric_values), 6),
                    "worst_group_auc_roc": round(min(metric_values), 6),
                    "max_pairwise_gap": round(max(metric_values) - min(metric_values), 6),
                    "disparity_ratio": round(min(metric_values) / max(metric_values), 6)
                    if max(metric_values)
                    else 0.0,
                }
            )
            rows.append(summary_row)
        if {"G0", "G1"}.issubset(gender_auc):
            rows.append(
                {
                    **_empty_metric_row(scenario, attack, "gender_gap", "gap:G0-G1", 0, 0),
                    "suppressed": False,
                    "pld_auc_roc": round(abs(gender_auc["G0"] - gender_auc["G1"]), 6),
                    "max_pairwise_gap": round(max(gender_auc.values()) - min(gender_auc.values()), 6),
                    "disparity_ratio": round(
                        min(gender_auc.values()) / max(gender_auc.values()), 6
                    ) if max(gender_auc.values()) else 0.0,
                    "macro_auc_roc": round(sum(gender_auc.values()) / len(gender_auc), 6),
                    "worst_group_auc_roc": round(min(gender_auc.values()), 6),
                }
            )
    return rows


def _empty_metric_row(
    scenario: str,
    attack: str,
    dimension: str,
    scope: str,
    members: int,
    nonmembers: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "attack": attack,
        "dimension": dimension,
        "scope": scope,
        "samples": members + nonmembers,
        "members": members,
        "nonmembers": nonmembers,
        "suppressed": False,
    }
    for field in (
        "threshold",
        "threshold_source",
        "auc_roc",
        "auc_ci_low",
        "auc_ci_high",
        "auc_pr",
        "tpr_at_0_1_fpr",
        "tpr_at_1_fpr",
        "tpr_at_5_fpr",
        "fpr_at_90_tpr",
        "fpr_at_95_tpr",
        "fpr_at_99_tpr",
        "accuracy",
        "balanced_accuracy",
        "majority_class_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "attack_advantage",
        "brier",
        "ece",
        "mean_loss",
        "perplexity",
        "pld_auc_roc",
        "macro_auc_roc",
        "worst_group_auc_roc",
        "max_pairwise_gap",
        "disparity_ratio",
    ):
        row[field] = ""
    return row
