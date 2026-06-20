"""Fine-tuning style offline scenario."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.defenses.base import Defense
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.scenarios.base import ProgressCallback, ScenarioRunner, score_samples_for_scenario
from fair_mia.types import AttackRecord, TextSample


class FineTuningScenarioRunner(ScenarioRunner):
    name = "finetuning"

    def run(
        self,
        samples: list[TextSample],
        attacks: list[MembershipInferenceAttack],
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter,
        defense: Defense,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AttackRecord]:
        return score_samples_for_scenario(
            self.name,
            samples,
            attacks,
            target_model,
            reference_model,
            defense,
            progress_callback=progress_callback,
        )
