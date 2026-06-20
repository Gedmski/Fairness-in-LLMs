"""Window-based comparison attack for fine-tuned language models."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import AttackScore, TextSample


class WindowBasedComparisonAttack(MembershipInferenceAttack):
    name = "wbc"
    requires_reference_model = True

    def __init__(self, window_sizes: tuple[int, ...] = (4, 8, 16, 32)) -> None:
        if not window_sizes or any(size <= 0 for size in window_sizes):
            raise ValueError("window_sizes must contain positive integers.")
        self.window_sizes = window_sizes

    def score(
        self,
        sample: TextSample,
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter | None = None,
    ) -> AttackScore:
        self.validate_models(target_model, reference_model)
        assert reference_model is not None
        target = target_model.score_tokens(sample.text, sample).losses
        reference = reference_model.score_tokens(sample.text, sample).losses
        count = min(len(target), len(reference))
        if count == 0:
            return AttackScore(self.name, 0.0, 0.0, {"votes": 0})
        differences = [reference[index] - target[index] for index in range(count)]
        votes: list[float] = []
        for size in self.window_sizes:
            if size > count:
                continue
            stride = max(1, size // 2)
            for start in range(0, count - size + 1, stride):
                window = differences[start : start + size]
                votes.append(1.0 if sum(window) / len(window) > 0 else -1.0)
        if not votes:
            votes = [1.0 if sum(differences) / len(differences) > 0 else -1.0]
        score = sum(votes) / len(votes)
        return AttackScore(
            self.name,
            raw_score=score,
            membership_score=score,
            diagnostics={
                "votes": len(votes),
                "window_sizes": list(self.window_sizes),
                "target_loss": sum(target[:count]) / count,
            },
        )
