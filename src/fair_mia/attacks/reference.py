"""Reference-model calibrated attack."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class ReferenceAttack(MembershipInferenceAttack):
    name = "reference"

    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        if reference_model is None:
            raise ValueError("ReferenceAttack requires a reference model.")
        target_loss = target_model.score_tokens(sample.text, sample).mean_loss
        reference_loss = reference_model.score_tokens(sample.text, sample).mean_loss
        raw = target_loss - reference_loss
        return AttackScore(
            self.name,
            raw_score=raw,
            membership_score=-raw,
            diagnostics={"target_loss": target_loss, "reference_loss": reference_loss},
        )

