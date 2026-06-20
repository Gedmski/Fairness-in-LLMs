"""Relative Conditional Log-Likelihood (ReCaLL) attack."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class RecallAttack(MembershipInferenceAttack):
    name = "recall"
    expensive = True
    required_capabilities = frozenset(
        {
            LanguageModelAdapter.Capability.TOKEN_LOSSES,
            LanguageModelAdapter.Capability.CONDITIONAL_SCORING,
        }
    )

    def score(
        self,
        sample: TextSample,
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter | None = None,
    ) -> AttackScore:
        self.validate_models(target_model, reference_model)
        prefix = str(
            sample.metadata.get(
                "recall_prefix",
                "This passage is a held-out example from the same broad text domain.",
            )
        )
        plain_loss = target_model.score_tokens(sample.text, sample).mean_loss
        conditional_loss = target_model.score_conditional(prefix, sample.text, sample).mean_loss
        relative_change = conditional_loss - plain_loss
        return AttackScore(
            self.name,
            raw_score=relative_change,
            membership_score=relative_change,
            diagnostics={
                "plain_loss": plain_loss,
                "target_loss": plain_loss,
                "conditional_loss": conditional_loss,
                "prefix_chars": len(prefix),
            },
        )
