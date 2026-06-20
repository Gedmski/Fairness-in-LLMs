"""RAG scenario placeholder until context-conditioned scoring is implemented."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.defenses.base import Defense
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.scenarios.base import ProgressCallback, ScenarioRunner
from fair_mia.types import AttackRecord, TextSample


class RagScenarioRunner(ScenarioRunner):
    name = "rag"
    UNSUPPORTED_MESSAGE = (
        "RAG scenario runs are currently unsupported because retrieved context is not injected into model scoring. "
        "Implement context-conditioned scoring before using rag results for benchmark claims."
    )

    def __init__(self, top_k: int = 2) -> None:
        self.top_k = top_k

    def run(
        self,
        samples: list[TextSample],
        attacks: list[MembershipInferenceAttack],
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter,
        defense: Defense,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AttackRecord]:
        raise NotImplementedError(self.UNSUPPORTED_MESSAGE)
