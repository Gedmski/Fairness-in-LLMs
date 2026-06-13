"""Registries for attacks and scenarios."""

from __future__ import annotations

from fair_mia.attacks import LossAttack, MembershipInferenceAttack, MinKProbAttack, NeighborhoodAttack, ReferenceAttack, ZlibEntropyAttack


def build_attack_registry() -> dict[str, MembershipInferenceAttack]:
    attacks: list[MembershipInferenceAttack] = [
        LossAttack(),
        ReferenceAttack(),
        ZlibEntropyAttack(),
        MinKProbAttack(),
        NeighborhoodAttack(),
    ]
    return {attack.name: attack for attack in attacks}


def get_attacks(names: list[str]) -> list[MembershipInferenceAttack]:
    registry = build_attack_registry()
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown attack(s): {', '.join(unknown)}")
    return [registry[name] for name in names]

