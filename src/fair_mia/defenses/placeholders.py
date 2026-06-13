"""Explicit placeholders for defenses that require real training or retrieval controls."""

from __future__ import annotations

from fair_mia.defenses.base import Defense
from fair_mia.types import TextSample


class _UnavailableDefense(Defense):
    name = "unavailable"

    def apply(self, sample: TextSample) -> TextSample:
        raise NotImplementedError(f"{self.name} is a placeholder in the offline scaffold.")


class DifferentialPrivacyDefense(_UnavailableDefense):
    name = "differential_privacy"


class DropoutDefense(_UnavailableDefense):
    name = "dropout"


class L2RegularizationDefense(_UnavailableDefense):
    name = "l2_regularization"


class RetrievalFilteringDefense(_UnavailableDefense):
    name = "retrieval_filtering"

