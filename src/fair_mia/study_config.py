"""YAML study configuration and controlled experiment expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any


SUPPORTED_STUDY_FIELDS = {
    "dataset",
    "model",
    "training_variant",
    "evaluation_variant",
    "seed",
    "epochs",
    "attacks",
    "full_attacks",
    "audit_attacks",
    "max_train_samples",
    "audit_cap_per_cell",
    "bootstrap_replicates",
    "skip_full_tier",
    "max_full_eval_samples",
    "max_audit_eval_samples",
    "max_calibration_samples",
}


@dataclass(frozen=True)
class DatasetStudySpec:
    alias: str
    members_path: str
    nonmembers_path: str
    variants_dir: str
    languages: tuple[str, ...] = ()
    enabled: bool = True
    historical_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelStudySpec:
    alias: str
    model_id: str
    reference_model_id: str | None = None
    cache_dir: str = "artifacts/models"
    trust_remote_code: bool = False
    gated: bool = False
    load_in_4bit: bool = False


@dataclass(frozen=True)
class FineTuningDefaults:
    method: str = "lora"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 2e-4
    max_length: int = 256
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    epochs: int = 4
    max_train_samples: int = 0


@dataclass(frozen=True)
class EvaluationDefaults:
    calibration_fraction: float = 0.2
    audit_cap_per_cell: int = 100
    min_cell_members: int = 30
    bootstrap_replicates: int = 1000
    full_attacks: tuple[str, ...] = (
        "loss",
        "reference",
        "zlib",
        "min_k",
        "neighborhood",
        "min_k_plus_plus",
        "wbc",
    )
    audit_attacks: tuple[str, ...] = (
        "loss",
        "reference",
        "zlib",
        "min_k",
        "neighborhood",
        "min_k_plus_plus",
        "wbc",
        "recall",
        "samia",
        "spv_mia",
    )


@dataclass(frozen=True)
class RuntimeDefaults:
    outputs_dir: str = "outputs"
    adapters_dir: str = "artifacts/adapters"
    gpu_ids: tuple[int, ...] = (0, 1)
    continue_on_error: bool = True
    local_files_only: bool = True


@dataclass(frozen=True)
class ReportingDefaults:
    generate_plots: bool = True


@dataclass(frozen=True)
class StudyDefinition:
    name: str
    description: str = ""
    enabled: bool = True
    historical: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedExperiment:
    study_name: str
    dataset: str
    model: str
    training_variant: str
    evaluation_variant: str
    seed: int
    epochs: int
    attacks: tuple[str, ...] = ()
    full_attacks: tuple[str, ...] = ()
    audit_attacks: tuple[str, ...] = ()
    max_train_samples: int = 0
    audit_cap_per_cell: int = 100
    bootstrap_replicates: int = 1000
    skip_full_tier: bool = False
    max_full_eval_samples: int = 0
    max_audit_eval_samples: int = 0
    max_calibration_samples: int = 0
    historical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyConfig:
    source_path: Path
    raw: dict[str, Any]
    datasets: dict[str, DatasetStudySpec]
    models: dict[str, ModelStudySpec]
    finetuning: FineTuningDefaults
    evaluation: EvaluationDefaults
    runtime: RuntimeDefaults
    reporting: ReportingDefaults
    defaults: dict[str, Any]
    studies: tuple[StudyDefinition, ...]


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = _parse_simple_yaml(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must contain a top-level mapping.")
    return payload


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the conservative YAML subset used by repository experiment configs."""

    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def scalar(value: str) -> Any:
        value = value.strip()
        if value in {"", "null", "Null", "NULL", "~"}:
            return None
        if value == "{}":
            return {}
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            return [] if not inner else [scalar(item) for item in inner.split(",")]
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        is_list = lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Invalid YAML indentation near: {content}")
            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                container.append(scalar(item))
                index += 1
                continue
            if content.startswith("- ") or ":" not in content:
                break
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                container[key] = scalar(value)
            elif index < len(lines) and lines[index][0] > indent:
                container[key], index = parse_block(index, lines[index][0])
            else:
                container[key] = {}
        return container, index

    if not lines:
        return {}
    payload, final_index = parse_block(0, lines[0][0])
    if final_index != len(lines) or not isinstance(payload, dict):
        raise ValueError("Unsupported YAML structure in experiment config.")
    return payload


def _require_sections(raw: dict[str, Any]) -> None:
    required = {
        "datasets",
        "models",
        "attacks",
        "finetuning",
        "evaluation",
        "runtime",
        "reporting",
        "defaults",
        "studies",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Study config is missing sections: {', '.join(missing)}")


def _tuple_values(raw: dict[str, Any], key: str) -> tuple[Any, ...]:
    value = raw.pop(key, ())
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list.")
    return tuple(value)


def load_study_config(path: str | Path) -> StudyConfig:
    source = Path(path)
    raw = _load_yaml_or_json(source)
    _require_sections(raw)
    datasets: dict[str, DatasetStudySpec] = {}
    for alias, value in raw["datasets"].items():
        options = dict(value)
        options["languages"] = _tuple_values(options, "languages")
        options["historical_outputs"] = _tuple_values(options, "historical_outputs")
        datasets[alias] = DatasetStudySpec(alias=alias, **options)
    models = {
        alias: ModelStudySpec(alias=alias, **dict(value))
        for alias, value in raw["models"].items()
    }
    finetuning = FineTuningDefaults(**dict(raw["finetuning"]))
    evaluation_raw = dict(raw["evaluation"])
    configured_attack_groups = raw["attacks"]
    evaluation_raw.setdefault("full_attacks", configured_attack_groups.get("full", ()))
    evaluation_raw.setdefault("audit_attacks", configured_attack_groups.get("audit", ()))
    evaluation_raw["full_attacks"] = tuple(evaluation_raw["full_attacks"])
    evaluation_raw["audit_attacks"] = tuple(evaluation_raw["audit_attacks"])
    evaluation = EvaluationDefaults(**evaluation_raw)
    runtime_raw = dict(raw["runtime"])
    runtime_raw["gpu_ids"] = tuple(runtime_raw.get("gpu_ids", (0, 1)))
    runtime = RuntimeDefaults(**runtime_raw)
    studies: list[StudyDefinition] = []
    for name, value in raw["studies"].items():
        options = dict(value)
        overrides = dict(options.pop("overrides", {}))
        sweep = dict(options.pop("sweep", {}))
        unknown = (set(overrides) | set(sweep)) - SUPPORTED_STUDY_FIELDS
        if unknown:
            raise ValueError(f"Study {name!r} uses unsupported fields: {', '.join(sorted(unknown))}")
        for field_name, values in sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Study {name!r} sweep {field_name!r} must be a non-empty list.")
        studies.append(
            StudyDefinition(
                name=name,
                overrides=overrides,
                sweep=sweep,
                **options,
            )
        )
    return StudyConfig(
        source_path=source,
        raw=raw,
        datasets=datasets,
        models=models,
        finetuning=finetuning,
        evaluation=evaluation,
        runtime=runtime,
        reporting=ReportingDefaults(**dict(raw["reporting"])),
        defaults=dict(raw["defaults"]),
        studies=tuple(studies),
    )


def resolve_experiments(config: StudyConfig, *, study_name: str | None = None) -> list[ResolvedExperiment]:
    experiments: list[ResolvedExperiment] = []
    for study in config.studies:
        if not study.enabled or (study_name and study.name != study_name):
            continue
        values = dict(config.defaults)
        values.update(study.overrides)
        fields = list(study.sweep)
        combinations = product(*(study.sweep[field] for field in fields)) if fields else [()]
        for combination in combinations:
            resolved = dict(values)
            resolved.update(dict(zip(fields, combination)))
            dataset = str(resolved["dataset"])
            model = str(resolved["model"])
            if dataset not in config.datasets:
                raise ValueError(f"Study {study.name!r} references unknown dataset {dataset!r}.")
            if model not in config.models:
                raise ValueError(f"Study {study.name!r} references unknown model {model!r}.")
            if not config.datasets[dataset].enabled:
                continue
            attacks = tuple(str(value) for value in resolved.get("attacks", ()))
            full_attacks = tuple(str(value) for value in resolved.get("full_attacks", ()))
            audit_attacks = tuple(str(value) for value in resolved.get("audit_attacks", ()))
            if attacks:
                if full_attacks or audit_attacks:
                    raise ValueError(
                        f"Study {study.name!r} cannot set both 'attacks' and per-tier attack overrides."
                    )
                full_attacks = attacks
                audit_attacks = attacks
            experiments.append(
                ResolvedExperiment(
                    study_name=study.name,
                    dataset=dataset,
                    model=model,
                    training_variant=str(resolved.get("training_variant", "balanced")),
                    evaluation_variant=str(resolved.get("evaluation_variant", "balanced_audit")),
                    seed=int(resolved.get("seed", 29)),
                    epochs=int(resolved.get("epochs", config.finetuning.epochs)),
                    attacks=attacks,
                    full_attacks=full_attacks,
                    audit_attacks=audit_attacks,
                    max_train_samples=int(
                        resolved.get("max_train_samples", config.finetuning.max_train_samples)
                    ),
                    audit_cap_per_cell=int(
                        resolved.get("audit_cap_per_cell", config.evaluation.audit_cap_per_cell)
                    ),
                    bootstrap_replicates=int(
                        resolved.get(
                            "bootstrap_replicates",
                            config.evaluation.bootstrap_replicates,
                        )
                    ),
                    skip_full_tier=bool(resolved.get("skip_full_tier", False)),
                    max_full_eval_samples=int(resolved.get("max_full_eval_samples", 0)),
                    max_audit_eval_samples=int(resolved.get("max_audit_eval_samples", 0)),
                    max_calibration_samples=int(resolved.get("max_calibration_samples", 0)),
                    historical=study.historical,
                )
            )
    if study_name and not experiments:
        raise ValueError(f"No enabled study named {study_name!r}.")
    return experiments


def stable_hash(payload: dict[str, Any], length: int = 16) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
