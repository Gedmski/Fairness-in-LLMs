"""Min-K%++ reference-free token-distribution attack."""

from __future__ import annotations

import math

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class MinKPlusPlusAttack(MembershipInferenceAttack):
    name = "min_k_plus_plus"
    required_capabilities = frozenset(
        {
            LanguageModelAdapter.Capability.TOKEN_LOSSES,
            LanguageModelAdapter.Capability.TOKEN_DISTRIBUTIONS,
        }
    )

    def __init__(self, k_percent: float = 20.0) -> None:
        if not 0 < k_percent <= 100:
            raise ValueError("k_percent must be in (0, 100].")
        self.k_percent = k_percent

    def score(
        self,
        sample: TextSample,
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter | None = None,
    ) -> AttackScore:
        self.validate_models(target_model, reference_model)
        scores = target_model.score_tokens(sample.text, sample)
        if not scores.losses:
            return AttackScore(self.name, 0.0, 0.0, {"selected_tokens": 0})
        if len(scores.distribution_means) != len(scores.losses) or len(scores.distribution_stds) != len(scores.losses):
            raise ValueError("min_k_plus_plus requires per-token distribution means and standard deviations.")
        standardized_log_probs = [
            ((-loss) - (-mean_loss)) / max(std, 1e-8)
            for loss, mean_loss, std in zip(scores.losses, scores.distribution_means, scores.distribution_stds)
        ]
        selected_count = max(1, math.ceil(len(standardized_log_probs) * self.k_percent / 100.0))
        selected = sorted(standardized_log_probs)[:selected_count]
        membership_score = sum(selected) / len(selected)
        return AttackScore(
            self.name,
            raw_score=membership_score,
            membership_score=membership_score,
            diagnostics={
                "k_percent": self.k_percent,
                "selected_tokens": selected_count,
                "target_loss": scores.mean_loss,
            },
        )
