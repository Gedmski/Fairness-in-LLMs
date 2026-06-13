"""N-gram overlap diagnostics for fuzzy member boundaries."""

from __future__ import annotations

from fair_mia.types import TextSample


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    return ngrams_from_tokens(tokenize(text), n)


def ngrams_from_tokens(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def overlap_ratio(left: str, right: str, n: int) -> float:
    left_ngrams = ngrams(left, n)
    right_ngrams = ngrams(right, n)
    return overlap_ratio_from_ngrams(left_ngrams, right_ngrams)


def overlap_ratio_from_ngrams(left_ngrams: set[tuple[str, ...]], right_ngrams: set[tuple[str, ...]]) -> float:
    if not left_ngrams and not right_ngrams:
        return 0.0
    denominator = min(len(left_ngrams), len(right_ngrams))
    if denominator == 0:
        return 0.0
    return len(left_ngrams & right_ngrams) / denominator


def build_disabled_overlap_report(samples: list[TextSample]) -> dict[str, object]:
    members = [sample for sample in samples if sample.is_member]
    nonmembers = [sample for sample in samples if not sample.is_member]
    return {
        "status": "disabled",
        "sampled": False,
        "member_count": len(members),
        "nonmember_count": len(nonmembers),
        "evaluated_nonmembers": 0,
        "evaluated_members_per_nonmember": 0,
        "total_pairs_evaluated": 0,
        "n_grams": {},
    }


def build_overlap_report(
    samples: list[TextSample],
    *,
    enabled: bool = True,
    max_nonmembers: int = 250,
    max_members_per_nonmember: int = 250,
    n_values: tuple[int, ...] = (4, 7, 13),
    threshold: float = 0.8,
) -> dict[str, object]:
    if not enabled:
        return build_disabled_overlap_report(samples)
    if max_nonmembers <= 0:
        raise ValueError("max_nonmembers must be positive.")
    if max_members_per_nonmember <= 0:
        raise ValueError("max_members_per_nonmember must be positive.")

    members = sorted((sample for sample in samples if sample.is_member), key=lambda sample: sample.sample_id)
    nonmembers = sorted((sample for sample in samples if not sample.is_member), key=lambda sample: sample.sample_id)
    sampled_members = members[:max_members_per_nonmember]
    sampled_nonmembers = nonmembers[:max_nonmembers]

    sampled = len(sampled_members) < len(members) or len(sampled_nonmembers) < len(nonmembers)
    report: dict[str, object] = {
        "status": "complete",
        "sampled": sampled,
        "member_count": len(members),
        "nonmember_count": len(nonmembers),
        "evaluated_nonmembers": len(sampled_nonmembers),
        "evaluated_members_per_nonmember": len(sampled_members),
        "total_pairs_evaluated": len(sampled_nonmembers) * len(sampled_members),
        "n_grams": {},
    }

    member_tokens = {sample.sample_id: tokenize(sample.text) for sample in sampled_members}
    nonmember_tokens = {sample.sample_id: tokenize(sample.text) for sample in sampled_nonmembers}
    n_reports: dict[str, object] = {}
    for n in n_values:
        member_ngrams = {
            sample.sample_id: ngrams_from_tokens(member_tokens[sample.sample_id], n)
            for sample in sampled_members
        }
        nonmember_ngrams = {
            sample.sample_id: ngrams_from_tokens(nonmember_tokens[sample.sample_id], n)
            for sample in sampled_nonmembers
        }

        max_ratios: list[float] = []
        flagged_pairs: list[dict[str, object]] = []
        for nonmember in sampled_nonmembers:
            best_ratio = 0.0
            best_member_id = None
            nonmember_window = nonmember_ngrams[nonmember.sample_id]
            for member in sampled_members:
                ratio = overlap_ratio_from_ngrams(nonmember_window, member_ngrams[member.sample_id])
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
