"""Base interface for membership inference attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class MembershipInferenceAttack(ABC):
    name: str

    @abstractmethod
    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        """Return a score where larger membership_score means more likely to be a member."""

