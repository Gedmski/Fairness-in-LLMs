"""Dataset loading for local JSONL benchmark inputs."""

from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from fair_mia.types import TextSample


def load_jsonl_samples(path: str | Path, *, is_member: bool, default_scenario: str) -> list[TextSample]:
    samples: list[TextSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("sample_id") or f"{Path(path).stem}-{line_number}")
            text = str(payload["text"])
            group = str(payload["group"])
            scenario = str(payload.get("scenario", default_scenario))
            metadata = dict(payload.get("metadata", {}))
            samples.append(
                TextSample(
                    sample_id=sample_id,
                    text=text,
                    is_member=bool(payload.get("is_member", is_member)),
                    group=group,
                    scenario=scenario,
                    metadata=metadata,
                )
            )
    return samples


def validate_binary_groups(samples: Iterable[TextSample]) -> None:
    groups = sorted({sample.group for sample in samples})
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, found {groups}.")


def validate_samples(samples: Iterable[TextSample]) -> None:
    seen_ids: set[str] = set()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Dataset must contain at least one sample.")
    for sample in sample_list:
        if not sample.sample_id:
            raise ValueError("sample_id must not be empty.")
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if not sample.text.strip():
            raise ValueError(f"Sample {sample.sample_id} has empty text.")
        if sample.group not in {"G0", "G1"}:
            raise ValueError(f"Sample {sample.sample_id} has invalid group {sample.group!r}; expected G0 or G1.")
    validate_binary_groups(sample_list)


def load_benchmark_samples(members_path: str | Path, nonmembers_path: str | Path, *, default_scenario: str) -> list[TextSample]:
    members = load_jsonl_samples(members_path, is_member=True, default_scenario=default_scenario)
    nonmembers = load_jsonl_samples(nonmembers_path, is_member=False, default_scenario=default_scenario)
    samples = members + nonmembers
    validate_samples(samples)
    return samples


def write_jsonl_samples(samples: Iterable[TextSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "sample_id": sample.sample_id,
                "text": sample.text,
                "is_member": sample.is_member,
                "group": sample.group,
                "scenario": sample.scenario,
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def prepare_pan_from_csv(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    text_field: str,
    group_field: str,
    split_field: str,
    member_split: str,
    id_field: str = "sample_id",
    group0_value: str | None = None,
    group1_value: str | None = None,
) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("PAN input CSV is empty.")
    for field in [text_field, group_field, split_field]:
        if field not in rows[0]:
            raise ValueError(f"Required field not found in CSV: {field}")

    observed_groups = sorted({row[group_field] for row in rows})
    if group0_value is None or group1_value is None:
        if len(observed_groups) != 2:
            raise ValueError("Provide --group0-value and --group1-value when the input has more or fewer than two group values.")
        group0_value, group1_value = observed_groups

    samples: list[TextSample] = []
    for index, row in enumerate(rows, start=1):
        group_raw = row[group_field]
        if group_raw == group0_value:
            group = "G0"
        elif group_raw == group1_value:
            group = "G1"
        else:
            continue
        sample_id = row.get(id_field) or f"pan-{index}"
        samples.append(
            TextSample(
                sample_id=str(sample_id),
                text=row[text_field],
                is_member=row[split_field] == member_split,
                group=group,
                scenario="finetuning",
                metadata={"source": "pan", "original_group": group_raw, "split": row[split_field]},
            )
        )
    validate_samples(samples)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples([sample for sample in samples if sample.is_member], members_path)
    write_jsonl_samples([sample for sample in samples if not sample.is_member], nonmembers_path)
    return members_path, nonmembers_path


def prepare_pile_sample(
    *,
    output_dir: str | Path,
    max_members: int,
    max_nonmembers: int,
    cache_dir: str | Path | None = None,
    subset: str | None = None,
) -> tuple[Path, Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install research dependencies before preparing Pile samples: pip install -e .[research]") from exc

    dataset_kwargs = {
        "path": "EleutherAI/pile",
        "split": "train",
        "streaming": True,
    }
    if subset:
        dataset_kwargs["name"] = subset
    if cache_dir:
        dataset_kwargs["cache_dir"] = str(cache_dir)
    stream = load_dataset(**dataset_kwargs)
    members: list[TextSample] = []
    nonmembers: list[TextSample] = []
    for index, row in enumerate(stream):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        target = members if len(members) < max_members else nonmembers
        if len(nonmembers) >= max_nonmembers:
            break
        sample = TextSample(
            sample_id=f"pile-{index}",
            text=text[:4000],
            is_member=target is members,
            group="G0" if index % 2 == 0 else "G1",
            scenario="pretraining",
            metadata={"source": "pile", "synthetic_group_assignment": True},
        )
        target.append(sample)
    samples = members + nonmembers
    validate_samples(samples)
    output_dir = Path(output_dir)
    members_path = output_dir / "members.jsonl"
    nonmembers_path = output_dir / "nonmembers.jsonl"
    write_jsonl_samples(members, members_path)
    write_jsonl_samples(nonmembers, nonmembers_path)
    return members_path, nonmembers_path


def summarize_samples(samples: Iterable[TextSample]) -> dict[str, object]:
    sample_list = list(samples)
    by_group = Counter(sample.group for sample in sample_list)
    by_membership = Counter("member" if sample.is_member else "nonmember" for sample in sample_list)
    by_group_membership = Counter(
        f"{sample.group}:{'member' if sample.is_member else 'nonmember'}" for sample in sample_list
    )
    return {
        "total": len(sample_list),
        "groups": dict(sorted(by_group.items())),
        "membership": dict(sorted(by_membership.items())),
        "group_membership": dict(sorted(by_group_membership.items())),
    }
