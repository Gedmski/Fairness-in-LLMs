"""Scenario runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.defenses.base import Defense
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackRecord, TextSample


ProgressCallback = Callable[[int, int, str, str], None]


class ScenarioRunner(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        samples: list[TextSample],
        attacks: list[MembershipInferenceAttack],
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter,
        defense: Defense,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AttackRecord]:
        """Run scenario-specific attack evaluation."""


def score_samples_for_scenario(
    scenario_name: str,
    samples: list[TextSample],
    attacks: list[MembershipInferenceAttack],
    target_model: LanguageModelAdapter,
    reference_model: LanguageModelAdapter,
    defense: Defense,
    extra_diagnostics: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[AttackRecord]:
    records: list[AttackRecord] = []
    sample_total = len(samples)
    for sample_index, original_sample in enumerate(samples, start=1):
        sample = defense.apply(original_sample.with_scenario(scenario_name))
        for attack in attacks:
            score = attack.score(sample, target_model, reference_model)
            diagnostics = dict(extra_diagnostics or {})
            diagnostics.update(score.diagnostics)
            records.append(
                AttackRecord(
                    sample_id=sample.sample_id,
                    scenario=scenario_name,
                    attack=score.attack,
                    group=sample.group,
                    is_member=sample.is_member,
                    raw_score=score.raw_score,
                    membership_score=score.membership_score,
                    diagnostics=diagnostics,
                    attributes=dict(sample.attributes),
                )
            )
            if progress_callback is not None:
                progress_callback(sample_index, sample_total, attack.name, sample.sample_id)
    return records
