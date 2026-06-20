"""Deterministic fake language-model adapter for offline tests and demos."""

from __future__ import annotations

import hashlib
import math

from fair_mia.models.base import LanguageModelAdapter
from fair_mia.overlap import tokenize
from fair_mia.types import TextSample, TokenScores


class FakeLanguageModelAdapter(LanguageModelAdapter):
    def __init__(
        self,
        *,
        name: str,
        seed: int,
        member_bias: float = -0.35,
        group_bias: dict[str, float] | None = None,
    ) -> None:
        self.name = name
        self.seed = seed
        self.member_bias = member_bias
        self.group_bias = group_bias or {}

    @property
    def capabilities(self) -> frozenset[LanguageModelAdapter.Capability]:
        return frozenset(LanguageModelAdapter.Capability)

    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        tokens = tokenize(text)
        losses = [self._token_loss(token, index, sample) for index, token in enumerate(tokens)]
        means = [loss + 0.35 for loss in losses]
        stds = [0.4 + ((index % 5) * 0.03) for index in range(len(losses))]
        return TokenScores(tokens=tokens, losses=losses, distribution_means=means, distribution_stds=stds)

    def score_conditional(
        self,
        prefix: str,
        text: str,
        sample: TextSample | None = None,
    ) -> TokenScores:
        scores = self.score_tokens(text, sample)
        digest = hashlib.sha256(f"{self.seed}:{prefix}".encode("utf-8")).hexdigest()
        prefix_effect = (int(digest[:8], 16) / 0xFFFFFFFF - 0.5) * 0.08
        if sample is not None and sample.is_member:
            prefix_effect += 0.04
        losses = [max(loss + prefix_effect, 0.000001) for loss in scores.losses]
        return TokenScores(
            tokens=scores.tokens,
            losses=losses,
            distribution_means=scores.distribution_means,
            distribution_stds=scores.distribution_stds,
        )

    def generate(self, prompt: str, *, max_new_tokens: int, seed: int = 0) -> str:
        tokens = tokenize(prompt)
        if not tokens:
            return ""
        digest = hashlib.sha256(f"{self.seed}:{seed}:{prompt}".encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16) % len(tokens)
        rotated = tokens[offset:] + tokens[:offset]
        repeats = max(1, math.ceil(max_new_tokens / len(rotated)))
        return " ".join((rotated * repeats)[:max_new_tokens])

    def _token_loss(self, token: str, index: int, sample: TextSample | None) -> float:
        digest = hashlib.sha256(f"{self.seed}:{self.name}:{index}:{token}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        loss = 1.0 + bucket
        if sample is not None and sample.is_member:
            loss += self.member_bias
        if sample is not None:
            loss += self.group_bias.get(sample.group, 0.0)
        return max(loss, 0.000001)
