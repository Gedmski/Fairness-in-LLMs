"""Deterministic neighborhood curvature attack."""

from __future__ import annotations

import hashlib

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.overlap import tokenize
from fair_mia.types import AttackScore, TextSample


class NeighborhoodAttack(MembershipInferenceAttack):
    name = "neighborhood"

    def __init__(self, neighbors: int = 4) -> None:
        if neighbors <= 0:
            raise ValueError("neighbors must be positive.")
        self.neighbors = neighbors

    def score(self, sample: TextSample, target_model: LanguageModelAdapter, reference_model: LanguageModelAdapter | None = None) -> AttackScore:
        target_loss = target_model.score_tokens(sample.text, sample).mean_loss
        if hasattr(target_model, "multilingual_neighbors"):
            neighbor_texts = list(target_model.multilingual_neighbors(sample.text, self.neighbors))
        else:
            neighbor_texts = [self._perturb(sample.text, index) for index in range(self.neighbors)]
        neighbor_losses = [target_model.score_tokens(text, sample).mean_loss for text in neighbor_texts]
        mean_neighbor_loss = sum(neighbor_losses) / len(neighbor_losses)
        raw = target_loss - mean_neighbor_loss
        return AttackScore(
            self.name,
            raw_score=raw,
            membership_score=-raw,
            diagnostics={"neighbors": self.neighbors, "target_loss": target_loss, "mean_neighbor_loss": mean_neighbor_loss},
        )

    def _perturb(self, text: str, neighbor_index: int) -> str:
        tokens = tokenize(text)
        if not tokens:
            return text
        digest = hashlib.sha256(f"{neighbor_index}:{text}".encode("utf-8")).hexdigest()
        position = int(digest[:8], 16) % len(tokens)
        tokens[position] = f"{tokens[position]}_alt{neighbor_index}"
        return " ".join(tokens)
