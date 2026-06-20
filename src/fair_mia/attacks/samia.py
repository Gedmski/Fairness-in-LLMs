"""Generation-only sampling-based pseudo-likelihood attack."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.overlap import ngrams
from fair_mia.types import AttackScore, TextSample


class SamiaAttack(MembershipInferenceAttack):
    name = "samia"
    expensive = True
    required_capabilities = frozenset({LanguageModelAdapter.Capability.GENERATION})

    def __init__(self, generations: int = 3, ngram_size: int = 3, max_new_tokens: int = 48) -> None:
        self.generations = generations
        self.ngram_size = ngram_size
        self.max_new_tokens = max_new_tokens

    def score(
        self,
        sample: TextSample,
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter | None = None,
    ) -> AttackScore:
        self.validate_models(target_model, reference_model)
        words = sample.text.split()
        split = max(1, len(words) // 2)
        prompt = " ".join(words[:split])
        target_ngrams = ngrams(sample.text, self.ngram_size)
        similarities: list[float] = []
        for index in range(self.generations):
            generated = target_model.generate(prompt, max_new_tokens=self.max_new_tokens, seed=index)
            generated_ngrams = ngrams(generated, self.ngram_size)
            denominator = max(len(target_ngrams), 1)
            similarities.append(len(target_ngrams & generated_ngrams) / denominator)
        score = sum(similarities) / len(similarities) if similarities else 0.0
        return AttackScore(
            self.name,
            raw_score=score,
            membership_score=score,
            diagnostics={"generations": self.generations, "ngram_size": self.ngram_size},
        )
