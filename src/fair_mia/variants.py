"""Deterministic fine-tuning dataset variants and author-disjoint evaluation splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from fair_mia.data import load_jsonl_samples, write_jsonl_samples
from fair_mia.types import TextSample


TRAINING_VARIANTS = ("raw_full", "size_matched_random", "balanced")
EVALUATION_VARIANTS = ("native_raw", "balanced_audit")


def _author_id(sample: TextSample) -> str:
    return str(sample.metadata.get("author_id") or sample.attributes.get("author_id") or sample.sample_id.split(":w", 1)[0])


def _gender(sample: TextSample) -> str:
    return str(sample.attributes.get("gender") or sample.metadata.get("original_group") or sample.group)


def _language(sample: TextSample) -> str:
    return str(sample.attributes.get("language") or sample.metadata.get("lang") or "unknown")


def _cell(sample: TextSample) -> tuple[str, str]:
    return (_gender(sample), _language(sample))


def _with_membership(sample: TextSample, is_member: bool, variant: str) -> TextSample:
    metadata = dict(sample.metadata)
    metadata["training_variant"] = variant
    metadata["exposure_count"] = 1 if is_member else 0
    attributes = dict(sample.attributes)
    attributes["training_variant"] = variant
    attributes["author_id"] = _author_id(sample)
    return replace(sample, is_member=is_member, metadata=metadata, attributes=attributes)


def split_authors(
    samples: Iterable[TextSample],
    *,
    calibration_fraction: float = 0.2,
    seed: int,
) -> tuple[list[TextSample], list[TextSample]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    author_memberships: dict[str, set[bool]] = defaultdict(set)
    for sample in samples:
        author_memberships[_author_id(sample)].add(sample.is_member)
    ambiguous = sorted(author for author, values in author_memberships.items() if len(values) > 1)
    if ambiguous:
        raise ValueError(
            "Author IDs must be globally disjoint between member and nonmember pools; "
            f"found overlap for {ambiguous[:5]}"
        )
    by_cell: dict[tuple[str, str, bool], dict[str, list[TextSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_cell[(_gender(sample), _language(sample), sample.is_member)][_author_id(sample)].append(sample)

    calibration: list[TextSample] = []
    test: list[TextSample] = []
    for cell, authors in sorted(by_cell.items()):
        author_ids = sorted(authors)
        rng = random.Random(f"{seed}:{cell}")
        rng.shuffle(author_ids)
        calibration_count = max(1, round(len(author_ids) * calibration_fraction)) if len(author_ids) > 1 else 0
        calibration_authors = set(author_ids[:calibration_count])
        for author_id in author_ids:
            target = calibration if author_id in calibration_authors else test
            target.extend(authors[author_id])
    return calibration, test


def _balanced_sample(samples: list[TextSample], *, seed: int, cap_per_cell: int | None = None) -> list[TextSample]:
    buckets: dict[tuple[str, str], list[TextSample]] = defaultdict(list)
    for sample in samples:
        buckets[_cell(sample)].append(sample)
    nonempty = [values for values in buckets.values() if values]
    if not nonempty:
        return []
    limit = min(len(values) for values in nonempty)
    if cap_per_cell is not None:
        limit = min(limit, cap_per_cell)
    selected: list[TextSample] = []
    for cell, values in sorted(buckets.items()):
        rng = random.Random(f"{seed}:{cell}")
        candidates = list(values)
        rng.shuffle(candidates)
        selected.extend(candidates[:limit])
    return selected


def _native_nonmembers(nonmembers: list[TextSample], *, count: int, seed: int) -> list[TextSample]:
    candidates = list(nonmembers)
    random.Random(seed).shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def _audit_subset(samples: list[TextSample], *, seed: int, cap_per_cell: int) -> list[TextSample]:
    buckets: dict[tuple[str, str, bool], list[TextSample]] = defaultdict(list)
    for sample in samples:
        buckets[(_gender(sample), _language(sample), sample.is_member)].append(sample)
    selected: list[TextSample] = []
    for cell, values in sorted(buckets.items()):
        candidates = list(values)
        random.Random(f"audit:{seed}:{cell}").shuffle(candidates)
        selected.extend(candidates[:cap_per_cell])
    return selected


def _balanced_evaluation(
    members: list[TextSample],
    nonmembers: list[TextSample],
    *,
    seed: int,
) -> list[TextSample]:
    buckets: dict[tuple[str, str, bool], list[TextSample]] = defaultdict(list)
    for sample in members + nonmembers:
        buckets[(_gender(sample), _language(sample), sample.is_member)].append(sample)
    if not buckets:
        return []
    limit = min(len(values) for values in buckets.values())
    selected: list[TextSample] = []
    for cell, values in sorted(buckets.items()):
        candidates = list(values)
        random.Random(f"balanced-eval:{seed}:{cell}").shuffle(candidates)
        selected.extend(candidates[:limit])
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_variants(
    *,
    members_path: str | Path,
    nonmembers_path: str | Path,
    output_dir: str | Path,
    seed: int,
    calibration_fraction: float = 0.2,
    audit_cap_per_cell: int = 100,
) -> dict[str, object]:
    members_path = Path(members_path)
    nonmembers_path = Path(nonmembers_path)
    output_dir = Path(output_dir)
    members = load_jsonl_samples(members_path, is_member=True, default_scenario="finetuning")
    nonmembers = load_jsonl_samples(nonmembers_path, is_member=False, default_scenario="finetuning")
    balanced_members = _balanced_sample(members, seed=seed)
    random_members = list(members)
    random.Random(seed).shuffle(random_members)
    random_members = random_members[: len(balanced_members)]
    training_sets = {
        "raw_full": members,
        "size_matched_random": random_members,
        "balanced": balanced_members,
    }

    manifest: dict[str, object] = {
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "audit_cap_per_cell": audit_cap_per_cell,
        "sources": {
            "members": {"path": str(members_path), "sha256": _sha256(members_path)},
            "nonmembers": {"path": str(nonmembers_path), "sha256": _sha256(nonmembers_path)},
        },
        "variants": {},
    }
    for variant, selected_members in training_sets.items():
        variant_dir = output_dir / variant
        tagged_members = [_with_membership(sample, True, variant) for sample in selected_members]
        tagged_nonmembers = [_with_membership(sample, False, variant) for sample in nonmembers]
        write_jsonl_samples(tagged_members, variant_dir / "train_members.jsonl")

        native_nonmembers = _native_nonmembers(tagged_nonmembers, count=len(tagged_members), seed=seed)
        native_all = tagged_members + native_nonmembers
        balanced_all = _balanced_evaluation(tagged_members, tagged_nonmembers, seed=seed)

        eval_counts: dict[str, object] = {}
        for eval_name, eval_samples in {
            "native_raw": native_all,
            "balanced_audit": balanced_all,
        }.items():
            calibration, test = split_authors(
                eval_samples,
                calibration_fraction=calibration_fraction,
                seed=seed,
            )
            eval_dir = variant_dir / "evaluation" / eval_name
            for split_name, split_samples in {"calibration": calibration, "test": test}.items():
                write_jsonl_samples(
                    [sample for sample in split_samples if sample.is_member],
                    eval_dir / f"{split_name}_members.jsonl",
                )
                write_jsonl_samples(
                    [sample for sample in split_samples if not sample.is_member],
                    eval_dir / f"{split_name}_nonmembers.jsonl",
                )
            audit = _audit_subset(test, seed=seed, cap_per_cell=audit_cap_per_cell)
            write_jsonl_samples(
                [sample for sample in audit if sample.is_member],
                eval_dir / "audit_members.jsonl",
            )
            write_jsonl_samples(
                [sample for sample in audit if not sample.is_member],
                eval_dir / "audit_nonmembers.jsonl",
            )
            eval_counts[eval_name] = {
                "calibration": len(calibration),
                "test": len(test),
                "audit": len(audit),
            }

        manifest["variants"][variant] = {
            "training_samples": len(tagged_members),
            "training_authors": len({_author_id(sample) for sample in tagged_members}),
            "training_author_ids": sorted({_author_id(sample) for sample in tagged_members}),
            "selected_window_ids": sorted(sample.sample_id for sample in tagged_members),
            "token_count": {
                "minimum": min(
                    (int(sample.metadata.get("token_count", len(sample.text.split()))) for sample in tagged_members),
                    default=0,
                ),
                "maximum": max(
                    (int(sample.metadata.get("token_count", len(sample.text.split()))) for sample in tagged_members),
                    default=0,
                ),
            },
            "cells": dict(sorted(Counter(f"{_gender(sample)}:{_language(sample)}" for sample in tagged_members).items())),
            "evaluation": eval_counts,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
