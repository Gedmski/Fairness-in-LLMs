"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    members_path: Path
    nonmembers_path: Path
    default_scenario: str = "offline"
    calibration_members_path: Path | None = None
    calibration_nonmembers_path: Path | None = None
    evaluation_variant: str = "default"
    evaluation_tier: str = "full"


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "fake"
    name: str = "fake-target"
    member_bias: float = -0.35
    group_bias: dict[str, float] = field(default_factory=dict)
    reference_member_bias: float = -0.05
    reference_group_bias: dict[str, float] = field(default_factory=dict)
    model_id: str = "EleutherAI/pythia-160m"
    reference_model_id: str | None = None
    revision: str | None = None
    cache_dir: Path | None = None
    dtype: str = "auto"
    device_map: str = "auto"
    max_length: int = 512
    batch_size: int = 1
    local_files_only: bool = False
    trust_remote_code: bool = False
    adapter_path: Path | None = None
    load_in_4bit: bool = False
    perturbation_model_id: str = "FacebookAI/xlm-roberta-base"


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 2


@dataclass(frozen=True)
class OverlapConfig:
    enabled: bool = True
    max_nonmembers: int = 250
    max_members_per_nonmember: int = 250
    n_values: tuple[int, ...] = (4, 7, 13)
    threshold: float = 0.8


@dataclass(frozen=True)
class BenchmarkConfig:
    run_id: str
    seed: int
    outputs_dir: Path
    dataset: DatasetConfig
    attacks: list[str]
    scenarios: list[str]
    model: ModelConfig
    rag: RagConfig = field(default_factory=RagConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)

    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    dataset_raw = raw.get("dataset", {})
    model_raw = raw.get("model", {})
    rag_raw = raw.get("rag", {})
    overlap_raw = raw.get("overlap", {})

    def resolve_path(value: str) -> Path:
        path_value = Path(value)
        return path_value if path_value.is_absolute() else (base_dir / path_value)

    dataset = DatasetConfig(
        members_path=resolve_path(dataset_raw["members_path"]),
        nonmembers_path=resolve_path(dataset_raw["nonmembers_path"]),
        default_scenario=dataset_raw.get("default_scenario", "offline"),
        calibration_members_path=resolve_path(dataset_raw["calibration_members_path"])
        if dataset_raw.get("calibration_members_path")
        else None,
        calibration_nonmembers_path=resolve_path(dataset_raw["calibration_nonmembers_path"])
        if dataset_raw.get("calibration_nonmembers_path")
        else None,
        evaluation_variant=str(dataset_raw.get("evaluation_variant", "default")),
        evaluation_tier=str(dataset_raw.get("evaluation_tier", "full")),
    )
    model = ModelConfig(
        backend=model_raw.get("backend", "fake"),
        name=model_raw.get("name", "fake-target"),
        member_bias=float(model_raw.get("member_bias", -0.35)),
        group_bias={str(k): float(v) for k, v in model_raw.get("group_bias", {}).items()},
        reference_member_bias=float(model_raw.get("reference_member_bias", -0.05)),
        reference_group_bias={str(k): float(v) for k, v in model_raw.get("reference_group_bias", {}).items()},
        model_id=str(model_raw.get("model_id", "EleutherAI/pythia-160m")),
        reference_model_id=model_raw.get("reference_model_id"),
        revision=model_raw.get("revision"),
        cache_dir=resolve_path(model_raw["cache_dir"]) if model_raw.get("cache_dir") else None,
        dtype=str(model_raw.get("dtype", "auto")),
        device_map=str(model_raw.get("device_map", "auto")),
        max_length=int(model_raw.get("max_length", 512)),
        batch_size=int(model_raw.get("batch_size", 1)),
        local_files_only=bool(model_raw.get("local_files_only", False)),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        adapter_path=resolve_path(model_raw["adapter_path"]) if model_raw.get("adapter_path") else None,
        load_in_4bit=bool(model_raw.get("load_in_4bit", False)),
        perturbation_model_id=str(
            model_raw.get("perturbation_model_id", "FacebookAI/xlm-roberta-base")
        ),
    )
    rag = RagConfig(top_k=int(rag_raw.get("top_k", 2)))
    overlap = OverlapConfig(
        enabled=bool(overlap_raw.get("enabled", True)),
        max_nonmembers=int(overlap_raw.get("max_nonmembers", 250)),
        max_members_per_nonmember=int(overlap_raw.get("max_members_per_nonmember", 250)),
        n_values=tuple(int(value) for value in overlap_raw.get("n_values", [4, 7, 13])),
        threshold=float(overlap_raw.get("threshold", 0.8)),
    )
    config = BenchmarkConfig(
        run_id=str(raw.get("run_id", "offline_demo")),
        seed=int(raw.get("seed", 7)),
        outputs_dir=resolve_path(raw.get("outputs_dir", "outputs")),
        dataset=dataset,
        attacks=[str(name) for name in raw.get("attacks", [])],
        scenarios=[str(name) for name in raw.get("scenarios", [])],
        model=model,
        rag=rag,
        overlap=overlap,
        raw=raw,
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    if not config.attacks:
        raise ValueError("Config must include at least one attack.")
    if not config.scenarios:
        raise ValueError("Config must include at least one scenario.")
    if not config.dataset.members_path.exists():
        raise FileNotFoundError(f"Members file not found: {config.dataset.members_path}")
    if not config.dataset.nonmembers_path.exists():
        raise FileNotFoundError(f"Non-members file not found: {config.dataset.nonmembers_path}")
    if bool(config.dataset.calibration_members_path) != bool(config.dataset.calibration_nonmembers_path):
        raise ValueError("Both calibration_members_path and calibration_nonmembers_path must be provided together.")
    for calibration_path in (
        config.dataset.calibration_members_path,
        config.dataset.calibration_nonmembers_path,
    ):
        if calibration_path is not None and not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    if config.model.backend not in {"fake", "hf"}:
        raise ValueError("model.backend must be either 'fake' or 'hf'.")
    if config.model.max_length <= 0:
        raise ValueError("model.max_length must be positive.")
    if config.model.batch_size <= 0:
        raise ValueError("model.batch_size must be positive.")
    if config.model.adapter_path is not None and not config.model.adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {config.model.adapter_path}")
    if config.overlap.max_nonmembers <= 0:
        raise ValueError("overlap.max_nonmembers must be positive.")
    if config.overlap.max_members_per_nonmember <= 0:
        raise ValueError("overlap.max_members_per_nonmember must be positive.")
    if not config.overlap.n_values:
        raise ValueError("overlap.n_values must not be empty.")
    if any(value <= 0 for value in config.overlap.n_values):
        raise ValueError("overlap.n_values must contain only positive integers.")
    if not 0.0 <= config.overlap.threshold <= 1.0:
        raise ValueError("overlap.threshold must be between 0 and 1.")
