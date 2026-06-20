"""Base interface for membership inference attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class MembershipInferenceAttack(ABC):
    name: str
    required_capabilities: frozenset[LanguageModelAdapter.Capability] = frozenset(
        {LanguageModelAdapter.Capability.TOKEN_LOSSES}
    )
    requires_reference_model: bool = False
    expensive: bool = False

    def validate_models(
        self,
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter | None,
    ) -> None:
        target_model.require_capabilities(*self.required_capabilities)
        if self.requires_reference_model and reference_model is None:
            raise ValueError(f"{self.name} requires a reference model.")

    @abstractmethod
    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        """Return a score where larger membership_score means more likely to be a member."""
