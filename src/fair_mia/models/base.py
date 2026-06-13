"""Language-model adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fair_mia.types import TextSample, TokenScores


class LanguageModelAdapter(ABC):
    name: str

    @abstractmethod
    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        """Return per-token loss-like scores without mutating model state."""

