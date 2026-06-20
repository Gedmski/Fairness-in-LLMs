"""Zlib complexity-calibrated attack."""

from __future__ import annotations

import zlib

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class ZlibEntropyAttack(MembershipInferenceAttack):
    name = "zlib"

    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        mean_loss = target_model.score_tokens(sample.text, sample).mean_loss
        compressed_size = max(len(zlib.compress(sample.text.encode("utf-8"))), 1)
        raw = mean_loss / compressed_size
        return AttackScore(
            self.name,
            raw_score=raw,
            membership_score=-raw,
            diagnostics={"compressed_size": compressed_size, "target_loss": mean_loss},
        )
