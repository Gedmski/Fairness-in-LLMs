"""Self-calibrated probabilistic-variation MIA."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class SpvMiaAttack(MembershipInferenceAttack):
    name = "spv_mia"
    expensive = True
    requires_reference_model = True
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
        assert reference_model is not None
        prompt = str(
            sample.metadata.get(
                "spv_prompt",
                "Continue with text that follows the same topic and writing style:",
            )
        )
        target_plain = target_model.score_tokens(sample.text, sample).mean_loss
        target_prompted = target_model.score_conditional(prompt, sample.text, sample).mean_loss
        reference_plain = reference_model.score_tokens(sample.text, sample).mean_loss
        reference_prompted = reference_model.score_conditional(prompt, sample.text, sample).mean_loss
        target_variation = target_prompted - target_plain
        reference_variation = reference_prompted - reference_plain
        score = reference_variation - target_variation
        return AttackScore(
            self.name,
            raw_score=score,
            membership_score=score,
            diagnostics={
                "target_variation": target_variation,
                "reference_variation": reference_variation,
                "target_loss": target_plain,
            },
        )
