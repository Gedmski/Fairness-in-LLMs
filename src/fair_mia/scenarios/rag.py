"""Offline RAG scenario with a local in-memory retriever."""

from __future__ import annotations

from fair_mia.attacks.base import MembershipInferenceAttack
from fair_mia.defenses.base import Defense
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.overlap import tokenize
from fair_mia.scenarios.base import ScenarioRunner, score_samples_for_scenario
from fair_mia.types import AttackRecord, TextSample


class LocalRetriever:
    def __init__(self, corpus: list[TextSample], top_k: int = 2) -> None:
        self.corpus = corpus
        self.top_k = top_k

    def retrieve(self, query: TextSample) -> list[str]:
        query_tokens = set(tokenize(query.text))
        scored: list[tuple[int, str]] = []
        for candidate in self.corpus:
            if candidate.sample_id == query.sample_id:
                continue
            overlap = len(query_tokens & set(tokenize(candidate.text)))
            scored.append((overlap, candidate.sample_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [sample_id for _, sample_id in scored[: self.top_k]]


class RagScenarioRunner(ScenarioRunner):
    name = "rag"

    def __init__(self, top_k: int = 2) -> None:
        self.top_k = top_k

    def run(
        self,
        samples: list[TextSample],
        attacks: list[MembershipInferenceAttack],
        target_model: LanguageModelAdapter,
        reference_model: LanguageModelAdapter,
        defense: Defense,
    ) -> list[AttackRecord]:
        retriever = LocalRetriever(samples, top_k=self.top_k)
        records: list[AttackRecord] = []
        for sample in samples:
            retrieved_ids = retriever.retrieve(sample)
            sample_records = score_samples_for_scenario(
                self.name,
                [sample],
                attacks,
                target_model,
                reference_model,
                defense,
                extra_diagnostics={"retrieved_ids": retrieved_ids},
            )
            records.extend(sample_records)
        return records

