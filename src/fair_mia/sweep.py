"""Study-driven LoRA job planning, execution, resume, and aggregation."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fair_mia.study_config import (
    ResolvedExperiment,
    StudyConfig,
    load_study_config,
    resolve_experiments,
    stable_hash,
)
from fair_mia.data import load_jsonl_samples, write_jsonl_samples


_CONSOLE_LOCK = threading.Lock()


def _root_for(config: StudyConfig) -> Path:
    return config.source_path.parent.parent if config.source_path.parent.name == "configs" else Path.cwd()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _job(stage: str, payload: dict[str, Any], dependencies: list[str] | None = None) -> dict[str, Any]:
    identifier = stable_hash({"stage": stage, **payload})
    return {
        "job_hash": identifier,
        "stage": stage,
        "dependencies": dependencies or [],
        **payload,
    }


def _sample_key(sample: Any) -> tuple[str, str]:
    language = sample.attributes.get("language") or str(sample.metadata.get("language", "unknown"))
    return sample.group, language


def _materialize_capped_jsonl(
    source_path: str | Path,
    output_path: Path,
    *,
    is_member: bool,
    default_scenario: str,
    max_samples: int,
    seed: int,
) -> str:
    source = Path(source_path)
    if max_samples <= 0:
        return str(source)
    samples = load_jsonl_samples(source, is_member=is_member, default_scenario=default_scenario)
    if len(samples) <= max_samples:
        return str(source)
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[Any]] = {}
    for sample in samples:
        buckets.setdefault(_sample_key(sample), []).append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[Any] = []
    active = [key for key, bucket in buckets.items() if bucket]
    while active and len(selected) < max_samples:
        next_active: list[tuple[str, str]] = []
        rng.shuffle(active)
        for key in active:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_active.append(key)
            if len(selected) >= max_samples:
                break
        active = next_active
    write_jsonl_samples(selected, output_path)
    return str(output_path)


def build_execution_plan(config: StudyConfig, experiments: list[ResolvedExperiment]) -> dict[str, Any]:
    root = _root_for(config)
    jobs: dict[str, dict[str, Any]] = {}
    scoring_jobs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for experiment in experiments:
        dataset = config.datasets[experiment.dataset]
        model = config.models[experiment.model]
        if experiment.historical:
            historical = _job(
                "historical",
                {
                    "study_name": experiment.study_name,
                    "dataset": experiment.dataset,
                    "historical_outputs": [
                        str(_resolve(root, value)) for value in dataset.historical_outputs
                    ],
                },
            )
            jobs[historical["job_hash"]] = historical
            continue

        variants_dir = (
            _resolve(root, dataset.variants_dir)
            / f"seed_{experiment.seed}_audit_{experiment.audit_cap_per_cell}"
        )
        prepare = _job(
            "dataset_preparation",
            {
                "study_name": experiment.study_name,
                "dataset": experiment.dataset,
                "members_path": str(_resolve(root, dataset.members_path)),
                "nonmembers_path": str(_resolve(root, dataset.nonmembers_path)),
                "variants_dir": str(variants_dir),
                "seed": experiment.seed,
                "calibration_fraction": config.evaluation.calibration_fraction,
                "audit_cap_per_cell": experiment.audit_cap_per_cell,
            },
        )
        prepare["job_hash"] = stable_hash(
            {
                "stage": "dataset_preparation",
                "dataset": experiment.dataset,
                "members_path": prepare["members_path"],
                "nonmembers_path": prepare["nonmembers_path"],
                "variants_dir": prepare["variants_dir"],
                "seed": experiment.seed,
                "calibration_fraction": prepare["calibration_fraction"],
                "audit_cap_per_cell": prepare["audit_cap_per_cell"],
            }
        )
        jobs.setdefault(prepare["job_hash"], prepare)

        training_payload = {
            "study_name": experiment.study_name,
            "dataset": experiment.dataset,
            "model": experiment.model,
            "model_id": model.model_id,
            "reference_model_id": model.reference_model_id or model.model_id,
            "cache_dir": str(_resolve(root, model.cache_dir)),
            "training_variant": experiment.training_variant,
            "train_jsonl": str(variants_dir / experiment.training_variant / "train_members.jsonl"),
            "seed": experiment.seed,
            "epochs": experiment.epochs,
            "max_train_samples": experiment.max_train_samples,
            "load_in_4bit": model.load_in_4bit,
            "adapter_dir": str(
                _resolve(root, config.runtime.adapters_dir)
                / stable_hash(
                    {
                        "dataset": experiment.dataset,
                        "model": experiment.model,
                        "variant": experiment.training_variant,
                        "seed": experiment.seed,
                        "epochs": experiment.epochs,
                        "max_train_samples": experiment.max_train_samples,
                    }
                )
            ),
            "finetuning": {
                "rank": config.finetuning.rank,
                "alpha": config.finetuning.alpha,
                "dropout": config.finetuning.dropout,
                "learning_rate": config.finetuning.learning_rate,
                "max_length": config.finetuning.max_length,
                "per_device_train_batch_size": config.finetuning.per_device_train_batch_size,
                "gradient_accumulation_steps": config.finetuning.gradient_accumulation_steps,
                "gradient_checkpointing": config.finetuning.gradient_checkpointing,
            },
        }
        training = _job("training", training_payload, [prepare["job_hash"]])
        training["job_hash"] = stable_hash(
            {
                "stage": "training",
                "dataset": experiment.dataset,
                "model": experiment.model,
                "model_id": model.model_id,
                "training_variant": experiment.training_variant,
                "seed": experiment.seed,
                "epochs": experiment.epochs,
                "max_train_samples": experiment.max_train_samples,
                "load_in_4bit": model.load_in_4bit,
                "finetuning": training_payload["finetuning"],
            }
        )
        jobs.setdefault(training["job_hash"], training)

        for tier, default_attacks, members_name, nonmembers_name in (
            (
                "full",
                config.evaluation.full_attacks,
                "test_members.jsonl",
                "test_nonmembers.jsonl",
            ),
            (
                "audit",
                config.evaluation.audit_attacks,
                "audit_members.jsonl",
                "audit_nonmembers.jsonl",
            ),
        ):
            if tier == "full" and experiment.skip_full_tier:
                continue
            tier_attacks = experiment.full_attacks if tier == "full" else experiment.audit_attacks
            resolved_attacks = tuple(tier_attacks or experiment.attacks or default_attacks)
            key = (
                training["job_hash"],
                experiment.evaluation_variant,
                tier,
                experiment.audit_cap_per_cell,
                resolved_attacks,
                experiment.bootstrap_replicates,
                experiment.max_full_eval_samples if tier == "full" else experiment.max_audit_eval_samples,
                experiment.max_calibration_samples,
            )
            evaluation_dir = (
                variants_dir
                / experiment.training_variant
                / "evaluation"
                / experiment.evaluation_variant
            )
            if key in scoring_jobs:
                scoring = scoring_jobs[key]
            else:
                scoring = _job(
                    "primitive_scoring",
                    {
                    "study_name": experiment.study_name,
                    "dataset": experiment.dataset,
                    "model": experiment.model,
                    "model_id": model.model_id,
                    "reference_model_id": model.reference_model_id or model.model_id,
                    "cache_dir": str(_resolve(root, model.cache_dir)),
                    "adapter_dir": training["adapter_dir"],
                    "training_job_hash": training["job_hash"],
                    "training_variant": experiment.training_variant,
                    "evaluation_variant": experiment.evaluation_variant,
                    "evaluation_tier": tier,
                    "seed": experiment.seed,
                    "epochs": experiment.epochs,
                    "attacks": list(resolved_attacks),
                    "members_path": str(evaluation_dir / members_name),
                    "nonmembers_path": str(evaluation_dir / nonmembers_name),
                    "calibration_members_path": str(evaluation_dir / "calibration_members.jsonl"),
                    "calibration_nonmembers_path": str(evaluation_dir / "calibration_nonmembers.jsonl"),
                    "max_eval_samples": (
                        experiment.max_full_eval_samples
                        if tier == "full"
                        else experiment.max_audit_eval_samples
                    ),
                    "max_calibration_samples": experiment.max_calibration_samples,
                    "local_files_only": config.runtime.local_files_only,
                    "load_in_4bit": model.load_in_4bit,
                    "trust_remote_code": model.trust_remote_code,
                    "max_length": config.finetuning.max_length,
                    "min_cell_members": config.evaluation.min_cell_members,
                    "bootstrap_replicates": experiment.bootstrap_replicates,
                    },
                    [training["job_hash"]],
                )
                jobs[scoring["job_hash"]] = scoring
                scoring_jobs[key] = scoring
            evaluation = _job(
                "attack_evaluation",
                {
                    **{
                        key: value
                        for key, value in scoring.items()
                        if key not in {"job_hash", "stage", "dependencies"}
                    },
                    "study_name": experiment.study_name,
                    "primitive_job_hash": scoring["job_hash"],
                },
                [scoring["job_hash"]],
            )
            jobs[evaluation["job_hash"]] = evaluation

    report_dependencies = [
        job_hash
        for job_hash, job in jobs.items()
        if job["stage"] in {"attack_evaluation", "historical"}
    ]
    reporting = _job(
        "metrics_reporting",
        {"resolved_experiment_count": len(experiments)},
        report_dependencies,
    )
    jobs[reporting["job_hash"]] = reporting

    ordered = sorted(jobs.values(), key=lambda item: (
        {
            "dataset_preparation": 0,
            "training": 1,
            "primitive_scoring": 2,
            "attack_evaluation": 3,
            "historical": 4,
            "metrics_reporting": 5,
        }[item["stage"]],
        item["job_hash"],
    ))
    gpu_ids = config.runtime.gpu_ids
    for stage in ("training", "primitive_scoring"):
        stage_jobs = [job for job in ordered if job["stage"] == stage]
        for index, job in enumerate(stage_jobs):
            job["planned_gpu"] = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
    return {
        "config_path": str(config.source_path.resolve()),
        "config_hash": stable_hash(
            {
                "config": config.raw,
                "resolved_experiments": [experiment.to_dict() for experiment in experiments],
            },
            24,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_experiments": [experiment.to_dict() for experiment in experiments],
        "job_count": len(ordered),
        "training_job_count": sum(job["stage"] == "training" for job in ordered),
        "scoring_job_count": sum(job["stage"] == "primitive_scoring" for job in ordered),
        "gpu_ids": list(gpu_ids),
        "jobs": ordered,
        "estimated_artifacts": [
            "resolved_config.yaml",
            "execution_plan.json",
            "runs.jsonl",
            "failures.jsonl",
            "summary.csv",
            "metrics.csv",
            "comparisons.csv",
            "report.md",
            "plots/",
            "studies/",
            "jobs/",
        ],
    }


def render_plan(plan: dict[str, Any]) -> str:
    study_counts: dict[str, int] = {}
    for experiment in plan["resolved_experiments"]:
        study_counts[experiment["study_name"]] = study_counts.get(experiment["study_name"], 0) + 1
    lines = [
        f"Resolved {len(study_counts)} studies / {len(plan['resolved_experiments'])} experiments",
        f"Jobs: {plan['job_count']} total, {plan['training_job_count']} training, "
        f"{plan['scoring_job_count']} scoring",
        f"GPU workers: {', '.join(str(value) for value in plan['gpu_ids']) or 'CPU'}",
    ]
    lines.extend(f"- {name}: {count} experiment(s)" for name, count in sorted(study_counts.items()))
    return "\n".join(lines)


def _write_yaml_or_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _new_invocation_dir(config: StudyConfig, config_hash: str) -> Path:
    root = _resolve(_root_for(config), config.runtime.outputs_dir)
    now = datetime.now()
    invocation = f"{now.strftime('%H%M%S')}-{config_hash[:8]}"
    path = root / now.strftime("%Y-%m-%d") / invocation
    path.mkdir(parents=True, exist_ok=False)
    return path


def _find_resume_dir(config: StudyConfig, config_hash: str) -> Path | None:
    root = _resolve(_root_for(config), config.runtime.outputs_dir)
    candidates = sorted(root.glob(f"*/*-{config_hash[:8]}"), reverse=True)
    return candidates[0] if candidates else None


def _status_path(invocation_dir: Path, job_hash: str) -> Path:
    return invocation_dir / "jobs" / job_hash / "status.json"


def _completed(invocation_dir: Path, job_hash: str) -> bool:
    path = _status_path(invocation_dir, job_hash)
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "success"


def _job_status(invocation_dir: Path, job_hash: str) -> str:
    path = _status_path(invocation_dir, job_hash)
    if not path.exists():
        return "missing"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))


def execute_job(job: dict[str, Any], invocation_dir: str | Path, gpu_id: int | None = None) -> None:
    invocation_dir = Path(invocation_dir)
    job_dir = invocation_dir / "jobs" / job["job_hash"]
    job_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    status = {
        "job_hash": job["job_hash"],
        "stage": job["stage"],
        "study_name": job.get("study_name"),
        "gpu": gpu_id,
        "started_at": started.isoformat(),
        "status": "running",
        "job": job,
    }
    (job_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        if job["stage"] == "dataset_preparation":
            from fair_mia.variants import materialize_variants

            manifest = materialize_variants(
                members_path=job["members_path"],
                nonmembers_path=job["nonmembers_path"],
                output_dir=job["variants_dir"],
                seed=int(job["seed"]),
                calibration_fraction=float(job["calibration_fraction"]),
                audit_cap_per_cell=int(job["audit_cap_per_cell"]),
            )
            (job_dir / "dataset_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif job["stage"] == "training":
            from fair_mia.finetune import finetune_lora

            maximum = int(job["max_train_samples"])
            finetuning = job["finetuning"]
            finetune_lora(
                base_model_id=job["model_id"],
                train_jsonl=job["train_jsonl"],
                output_dir=job["adapter_dir"],
                cache_dir=job["cache_dir"],
                max_train_samples=maximum if maximum > 0 else 2**31 - 1,
                epochs=int(job["epochs"]),
                learning_rate=float(finetuning["learning_rate"]),
                max_length=int(finetuning["max_length"]),
                seed=int(job["seed"]),
                lora_rank=int(finetuning["rank"]),
                lora_alpha=int(finetuning["alpha"]),
                lora_dropout=float(finetuning["dropout"]),
                per_device_train_batch_size=int(finetuning["per_device_train_batch_size"]),
                gradient_accumulation_steps=int(finetuning["gradient_accumulation_steps"]),
                gradient_checkpointing=bool(finetuning["gradient_checkpointing"]),
                load_in_4bit=bool(job["load_in_4bit"]),
            )
            adapter_dir = Path(job["adapter_dir"])
            dataset_manifest = Path(job["train_jsonl"]).parents[1] / "dataset_manifest.json"
            if dataset_manifest.exists():
                shutil.copy2(dataset_manifest, adapter_dir / "dataset_manifest.json")
            (adapter_dir / "resolved_training_job.json").write_text(
                json.dumps(job, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif job["stage"] == "primitive_scoring":
            from fair_mia.cli import run_benchmark
            from fair_mia.config import load_config

            sampled_dir = job_dir / "sampled_inputs"
            sampled_dir.mkdir(parents=True, exist_ok=True)
            eval_seed = int(job["seed"]) + (0 if job["evaluation_tier"] == "full" else 10_000)
            calibration_seed = int(job["seed"]) + 20_000
            members_path = _materialize_capped_jsonl(
                job["members_path"],
                sampled_dir / "members.jsonl",
                is_member=True,
                default_scenario="finetuning",
                max_samples=int(job.get("max_eval_samples", 0)),
                seed=eval_seed,
            )
            nonmembers_path = _materialize_capped_jsonl(
                job["nonmembers_path"],
                sampled_dir / "nonmembers.jsonl",
                is_member=False,
                default_scenario="finetuning",
                max_samples=int(job.get("max_eval_samples", 0)),
                seed=eval_seed + 1,
            )
            calibration_members_path = _materialize_capped_jsonl(
                job["calibration_members_path"],
                sampled_dir / "calibration_members.jsonl",
                is_member=True,
                default_scenario="finetuning",
                max_samples=int(job.get("max_calibration_samples", 0)),
                seed=calibration_seed,
            )
            calibration_nonmembers_path = _materialize_capped_jsonl(
                job["calibration_nonmembers_path"],
                sampled_dir / "calibration_nonmembers.jsonl",
                is_member=False,
                default_scenario="finetuning",
                max_samples=int(job.get("max_calibration_samples", 0)),
                seed=calibration_seed + 1,
            )
            score_config = {
                "run_id": "score",
                "seed": job["seed"],
                "outputs_dir": str(job_dir),
                "dataset": {
                    "members_path": members_path,
                    "nonmembers_path": nonmembers_path,
                    "calibration_members_path": calibration_members_path,
                    "calibration_nonmembers_path": calibration_nonmembers_path,
                    "default_scenario": "finetuning",
                    "evaluation_variant": job["evaluation_variant"],
                    "evaluation_tier": job["evaluation_tier"],
                },
                "attacks": job["attacks"],
                "scenarios": ["finetuning"],
                "model": {
                    "backend": "hf",
                    "model_id": job["model_id"],
                    "adapter_path": job["adapter_dir"],
                    "reference_model_id": job["reference_model_id"],
                    "cache_dir": job["cache_dir"],
                    "dtype": "bfloat16",
                    "device_map": "auto",
                    "max_length": job["max_length"],
                    "local_files_only": job["local_files_only"],
                    "load_in_4bit": job["load_in_4bit"],
                    "trust_remote_code": job["trust_remote_code"],
                    "perturbation_model_id": "FacebookAI/xlm-roberta-base",
                },
                "evaluation": {
                    "min_cell_members": job["min_cell_members"],
                    "bootstrap_replicates": job["bootstrap_replicates"],
                },
                "overlap": {"enabled": True},
            }
            config_path = job_dir / "score_config.json"
            config_path.write_text(json.dumps(score_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run_benchmark(
                load_config(config_path),
                progress_path=job_dir / "progress.json",
                progress_label=(
                    f"{job['study_name']}:{job['evaluation_tier']}:{job['job_hash'][:8]}"
                ),
            )
        elif job["stage"] == "attack_evaluation":
            primitive_dir = invocation_dir / "jobs" / job["primitive_job_hash"] / "score"
            if not primitive_dir.exists():
                raise FileNotFoundError(f"Primitive scoring artifacts not found: {primitive_dir}")
            shutil.copytree(primitive_dir, job_dir / "score", dirs_exist_ok=True)
            (job_dir / "evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "primitive_job_hash": job["primitive_job_hash"],
                        "attacks": job["attacks"],
                        "evaluation_variant": job["evaluation_variant"],
                        "evaluation_tier": job["evaluation_tier"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        elif job["stage"] == "historical":
            indexed: list[str] = []
            for value in job["historical_outputs"]:
                path = Path(value)
                if path.exists():
                    indexed.append(str(path))
            (job_dir / "historical_index.json").write_text(
                json.dumps({"outputs": indexed}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif job["stage"] == "metrics_reporting":
            (job_dir / "reporting_manifest.json").write_text(
                json.dumps(
                    {
                        "resolved_experiment_count": job["resolved_experiment_count"],
                        "aggregation": "completed by parent runner after durable job status is written",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            raise ValueError(f"Unknown job stage: {job['stage']}")
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "failure_reason": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        (job_dir / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    status.update({"status": "success", "finished_at": datetime.now(timezone.utc).isoformat()})
    (job_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_job_file(job_file: str | Path, invocation_dir: str | Path, gpu_id: int | None) -> None:
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    execute_job(job, invocation_dir, gpu_id)


def _run_subprocess_job(
    job: dict[str, Any],
    invocation_dir: Path,
    gpu_id: int | None,
) -> tuple[str, bool, str]:
    job_dir = invocation_dir / "jobs" / job["job_hash"]
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "job.json"
    job_file.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "fair_mia.cli",
        "_execute-job",
        "--job-file",
        str(job_file),
        "--invocation-dir",
        str(invocation_dir),
    ]
    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    prefix = f"[{job['stage']} {job['job_hash'][:8]} gpu={gpu_id if gpu_id is not None else '-'}] "
    with _CONSOLE_LOCK:
        print(f"{prefix}started", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )

    def tee(source, log_handle, display) -> None:
        assert source is not None
        for line in iter(source.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            with _CONSOLE_LOCK:
                display.write(prefix + line)
                display.flush()
        source.close()

    with (
        (job_dir / "stdout.log").open("w", encoding="utf-8", buffering=1) as stdout_log,
        (job_dir / "stderr.log").open("w", encoding="utf-8", buffering=1) as stderr_log,
    ):
        stdout_thread = threading.Thread(
            target=tee,
            args=(process.stdout, stdout_log, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=tee,
            args=(process.stderr, stderr_log, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    error = ""
    if return_code != 0:
        error_lines = (job_dir / "stderr.log").read_text(encoding="utf-8").splitlines()
        error = "\n".join(error_lines[-20:])
    with _CONSOLE_LOCK:
        outcome = "success" if return_code == 0 else f"failed ({return_code})"
        print(f"{prefix}{outcome}", flush=True)
    return job["job_hash"], return_code == 0, error


def _run_stage(
    jobs: list[dict[str, Any]],
    invocation_dir: Path,
    gpu_ids: tuple[int, ...],
    *,
    resume: bool,
    retry_failed: bool,
    continue_on_error: bool,
) -> None:
    pending = _select_pending_jobs(
        jobs,
        invocation_dir,
        resume=resume,
        retry_failed=retry_failed,
    )
    if not pending:
        return
    worker_gpus = gpu_ids or (None,)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(worker_gpus)) as pool:
        futures = {
            pool.submit(_run_subprocess_job, job, invocation_dir, worker_gpus[index % len(worker_gpus)]): job
            for index, job in enumerate(pending)
        }
        for future in as_completed(futures):
            job_hash, success, error = future.result()
            if not success:
                failures.append(f"{job_hash}: {error}")
                if not continue_on_error:
                    for candidate in futures:
                        candidate.cancel()
                    break
    if failures and not continue_on_error:
        raise RuntimeError("Job stage failed:\n" + "\n".join(failures))


def _select_pending_jobs(
    jobs: list[dict[str, Any]],
    invocation_dir: Path,
    *,
    resume: bool,
    retry_failed: bool,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if not resume:
            pending.append(job)
            continue
        status = _job_status(invocation_dir, job["job_hash"])
        if status == "success":
            continue
        if status == "failed" and not retry_failed:
            continue
        pending.append(job)
    return pending


def run_sweep(
    config_path: str | Path,
    *,
    study_name: str | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> Path:
    config = load_study_config(config_path)
    experiments = resolve_experiments(config, study_name=study_name)
    plan = build_execution_plan(config, experiments)
    invocation_dir = _find_resume_dir(config, plan["config_hash"]) if resume else None
    if invocation_dir is None:
        invocation_dir = _new_invocation_dir(config, plan["config_hash"])
    _write_yaml_or_json(config.raw, invocation_dir / "resolved_config.yaml")
    (invocation_dir / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jobs_by_stage: dict[str, list[dict[str, Any]]] = {}
    for job in plan["jobs"]:
        jobs_by_stage.setdefault(job["stage"], []).append(job)
    for stage in (
        "dataset_preparation",
        "training",
        "primitive_scoring",
        "attack_evaluation",
        "historical",
        "metrics_reporting",
    ):
        _run_stage(
            jobs_by_stage.get(stage, []),
            invocation_dir,
            config.runtime.gpu_ids if stage in {"training", "primitive_scoring"} else (None,),
            resume=resume,
            retry_failed=retry_failed,
            continue_on_error=config.runtime.continue_on_error,
        )
    aggregate_run(invocation_dir)
    return invocation_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "execution_plan.json").read_text(encoding="utf-8"))
    metrics: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        status_path = _status_path(run_dir, job["job_hash"])
        if not status_path.exists():
            failures.append({**job, "status": "missing"})
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        runs.append(status)
        if status.get("status") != "success":
            failures.append(status)
            continue
        if job["stage"] != "attack_evaluation":
            continue
        summary_path = run_dir / "jobs" / job["job_hash"] / "score" / "summary.csv"
        for row in _read_csv(summary_path):
            metrics.append(
                {
                    "study_name": job["study_name"],
                    "job_hash": job["job_hash"],
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "training_variant": job["training_variant"],
                    "evaluation_variant": job["evaluation_variant"],
                    "evaluation_tier": job["evaluation_tier"],
                    "seed": job["seed"],
                    "epochs": job["epochs"],
                    **row,
                }
            )
    (run_dir / "runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in runs),
        encoding="utf-8",
    )
    (run_dir / "failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures),
        encoding="utf-8",
    )
    _write_csv(run_dir / "metrics.csv", metrics)
    summary = [
        row for row in metrics
        if row.get("dimension") == "all" and row.get("scope") == "all"
    ]
    _write_csv(run_dir / "summary.csv", summary)
    comparisons = _build_comparisons(summary)
    _write_csv(run_dir / "comparisons.csv", comparisons)
    _write_report(run_dir / "report.md", plan, summary, failures, comparisons)
    _generate_plots(run_dir / "plots", summary)

    for study_name in sorted({row.get("study_name", "") for row in metrics} | {
        job.get("study_name", "") for job in plan["jobs"]
    }):
        if not study_name:
            continue
        study_dir = run_dir / "studies" / study_name
        study_metrics = [row for row in metrics if row.get("study_name") == study_name]
        study_summary = [
            row for row in study_metrics
            if row.get("dimension") == "all" and row.get("scope") == "all"
        ]
        _write_csv(study_dir / "metrics.csv", study_metrics)
        _write_csv(study_dir / "summary.csv", study_summary)
        _write_csv(study_dir / "comparisons.csv", _build_comparisons(study_summary))
        study_runs = [
            row for row in runs
            if row.get("study_name") == study_name
            or row.get("job", {}).get("study_name") == study_name
        ]
        study_failures = [
            row for row in failures
            if row.get("study_name") == study_name
            or row.get("job", {}).get("study_name") == study_name
        ]
        (study_dir / "runs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in study_runs),
            encoding="utf-8",
        )
        (study_dir / "failures.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in study_failures),
            encoding="utf-8",
        )
        study_plan = {
            **plan,
            "jobs": [job for job in plan["jobs"] if job.get("study_name") == study_name],
            "resolved_experiments": [
                item
                for item in plan["resolved_experiments"]
                if item.get("study_name") == study_name
            ],
        }
        (study_dir / "resolved_plan.json").write_text(
            json.dumps(study_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_yaml_or_json(study_plan, study_dir / "resolved_config.yaml")
        _write_report(
            study_dir / "report.md",
            study_plan,
            study_summary,
            study_failures,
            _build_comparisons(study_summary),
        )
        _generate_plots(study_dir / "plots", study_summary)


def _build_comparisons(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary:
        key = (
            row.get("study_name"),
            row.get("dataset"),
            row.get("model"),
            row.get("seed"),
            row.get("epochs"),
            row.get("attack"),
            row.get("evaluation_tier"),
        )
        indexed[key + (row.get("training_variant"), row.get("evaluation_variant"))] = row
    comparisons: list[dict[str, Any]] = []
    for key, candidate in indexed.items():
        baseline = indexed.get(key[:-2] + ("raw_full", "native_raw"))
        if not baseline:
            continue
        try:
            delta = float(candidate["auc_roc"]) - float(baseline["auc_roc"])
        except (TypeError, ValueError):
            continue
        comparisons.append(
            {
                "study_name": key[0],
                "dataset": key[1],
                "model": key[2],
                "seed": key[3],
                "epochs": key[4],
                "attack": key[5],
                "evaluation_tier": key[6],
                "baseline_training_variant": "raw_full",
                "baseline_evaluation_variant": "native_raw",
                "candidate_training_variant": key[-2],
                "candidate_evaluation_variant": key[-1],
                "baseline_auc_roc": baseline["auc_roc"],
                "candidate_auc_roc": candidate["auc_roc"],
                "delta_auc_roc": round(delta, 6),
            }
        )
    return comparisons


def _write_report(
    path: Path,
    plan: dict[str, Any],
    summary: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fair MIA LoRA Experiment Report",
        "",
        f"- Resolved experiments: `{len(plan.get('resolved_experiments', []))}`",
        f"- Jobs: `{len(plan.get('jobs', []))}`",
        f"- Aggregate metric rows: `{len(summary)}`",
        f"- Failed or missing jobs: `{len(failures)}`",
        f"- Controlled raw-vs-balanced comparisons: `{len(comparisons)}`",
        "",
        "Results are descriptive until confidence intervals, multiple seeds, and cross-attack consistency are reviewed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_plots(path: Path, summary: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not summary:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped: dict[str, list[float]] = {}
    for row in summary:
        try:
            value = float(row["auc_roc"])
        except (KeyError, TypeError, ValueError):
            continue
        label = f"{row.get('training_variant')}→{row.get('evaluation_variant')}"
        grouped.setdefault(label, []).append(value)
    if not grouped:
        return
    labels = sorted(grouped)
    means = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    plt.figure(figsize=(max(8, len(labels) * 1.4), 5))
    plt.bar(labels, means, color="#315d8a")
    plt.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    plt.ylabel("Mean AUROC")
    plt.title("Mean MIA AUROC by training and evaluation distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path / "distribution_auc.png", dpi=200, bbox_inches="tight")
    plt.close()
