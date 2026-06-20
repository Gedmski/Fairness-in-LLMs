"""Language-model adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from fair_mia.types import TextSample, TokenScores


class LanguageModelAdapter(ABC):
    name: str

    class Capability(str, Enum):
        TOKEN_LOSSES = "token_losses"
        TOKEN_DISTRIBUTIONS = "token_distributions"
        CONDITIONAL_SCORING = "conditional_scoring"
        GENERATION = "generation"

    @property
    def capabilities(self) -> frozenset["LanguageModelAdapter.Capability"]:
        return frozenset({self.Capability.TOKEN_LOSSES})

    def require_capabilities(self, *required: "LanguageModelAdapter.Capability") -> None:
        missing = [capability.value for capability in required if capability not in self.capabilities]
        if missing:
            raise ValueError(f"Model adapter {self.name!r} does not support: {', '.join(missing)}")

    @abstractmethod
    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        """Return per-token loss-like scores without mutating model state."""

    def score_conditional(
        self,
        prefix: str,
        text: str,
        sample: TextSample | None = None,
    ) -> TokenScores:
        self.require_capabilities(self.Capability.CONDITIONAL_SCORING)
        return self.score_tokens(f"{prefix}\n{text}", sample)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int = 0,
    ) -> str:
        self.require_capabilities(self.Capability.GENERATION)
        raise NotImplementedError
