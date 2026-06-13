"""Min-k% probability style attack using highest-loss token subset."""

from __future__ import annotations

import math

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class MinKProbAttack(MembershipInferenceAttack):
    name = "min_k"

    def __init__(self, k_percent: float = 20.0) -> None:
        if k_percent <= 0 or k_percent > 100:
            raise ValueError("k_percent must be in (0, 100].")
        self.k_percent = k_percent

    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        token_scores = target_model.score_tokens(sample.text, sample)
        if not token_scores.losses:
            raw = 0.0
            selected = 0
        else:
            selected = max(1, math.ceil(len(token_scores.losses) * self.k_percent / 100.0))
            worst_losses = sorted(token_scores.losses, reverse=True)[:selected]
            raw = sum(worst_losses) / len(worst_losses)
        return AttackScore(
            self.name,
            raw_score=raw,
            membership_score=-raw,
            diagnostics={"k_percent": self.k_percent, "selected_tokens": selected},
        )

