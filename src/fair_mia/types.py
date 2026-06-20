"""Shared dataclasses for the offline benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class TextSample:
    sample_id: str
    text: str
    is_member: bool
    group: str
    scenario: str
    metadata: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)

    def with_scenario(self, scenario: str) -> "TextSample":
        return replace(self, scenario=scenario)


@dataclass(frozen=True)
class TokenScores:
    tokens: list[str]
    losses: list[float]
    distribution_means: list[float] = field(default_factory=list)
    distribution_stds: list[float] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)


@dataclass(frozen=True)
class AttackScore:
    attack: str
    raw_score: float
    membership_score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackRecord:
    sample_id: str
    scenario: str
    attack: str
    group: str
    is_member: bool
    raw_score: float
    membership_score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    records: list[AttackRecord]
    run_dir: str
