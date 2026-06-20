"""LOSS attack."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class LossAttack(MembershipInferenceAttack):
    name = "loss"

    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        token_scores = target_model.score_tokens(sample.text, sample)
        raw = token_scores.mean_loss
        return AttackScore(
            self.name,
            raw_score=raw,
            membership_score=-raw,
            diagnostics={"token_count": len(token_scores.tokens), "target_loss": raw},
        )
