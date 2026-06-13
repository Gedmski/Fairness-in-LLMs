"""Deterministic fake language-model adapter for offline tests and demos."""

from __future__ import annotations

import hashlib

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

    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        tokens = tokenize(text)
        losses = [self._token_loss(token, index, sample) for index, token in enumerate(tokens)]
        return TokenScores(tokens=tokens, losses=losses)

    def _token_loss(self, token: str, index: int, sample: TextSample | None) -> float:
        digest = hashlib.sha256(f"{self.seed}:{self.name}:{index}:{token}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        loss = 1.0 + bucket
        if sample is not None and sample.is_member:
            loss += self.member_bias
        if sample is not None:
            loss += self.group_bias.get(sample.group, 0.0)
        return max(loss, 0.000001)

