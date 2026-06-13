"""N-gram overlap diagnostics for fuzzy member boundaries."""

from __future__ import annotations

from fair_mia.types import TextSample


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def overlap_ratio(left: str, right: str, n: int) -> float:
    left_ngrams = ngrams(left, n)
    right_ngrams = ngrams(right, n)
    if not left_ngrams and not right_ngrams:
        return 0.0
    denominator = min(len(left_ngrams), len(right_ngrams))
    if denominator == 0:
        return 0.0
    return len(left_ngrams & right_ngrams) / denominator


def build_overlap_report(samples: list[TextSample], n_values: tuple[int, ...] = (4, 7, 13), threshold: float = 0.8) -> dict[str, object]:
    members = [sample for sample in samples if sample.is_member]
    nonmembers = [sample for sample in samples if not sample.is_member]
    report: dict[str, object] = {"member_count": len(members), "nonmember_count": len(nonmembers), "n_grams": {}}

    n_reports: dict[str, object] = {}
    for n in n_values:
        max_ratios: list[float] = []
        flagged_pairs: list[dict[str, object]] = []
        for nonmember in nonmembers:
            best_ratio = 0.0
            best_member_id = None
            for member in members:
                ratio = overlap_ratio(nonmember.text, member.text, n)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_member_id = member.sample_id
            max_ratios.append(best_ratio)
            if best_ratio >= threshold and best_member_id is not None:
                flagged_pairs.append(
                    {
                        "nonmember_id": nonmember.sample_id,
                        "member_id": best_member_id,
                        "overlap": round(best_ratio, 6),
                    }
                )
        mean_max = sum(max_ratios) / len(max_ratios) if max_ratios else 0.0
        n_reports[str(n)] = {
            "mean_max_overlap": round(mean_max, 6),
            "max_overlap": round(max(max_ratios), 6) if max_ratios else 0.0,
            "pairs_above_threshold": len(flagged_pairs),
            "flagged_pairs": flagged_pairs,
        }
    report["n_grams"] = n_reports
    return report

