"""Command-line entry point for the offline benchmark scaffold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from fair_mia import __version__
from fair_mia.config import BenchmarkConfig, load_config
from fair_mia.data import (
    load_benchmark_samples,
    prepare_pan_from_csv,
    prepare_pan17_from_xml,
    prepare_pan17_multilingual,
    prepare_pan18_from_xml,
    prepare_pile_sample,
    summarize_samples,
)
from fair_mia.defenses import NoOpDefense
from fair_mia.models import FakeLanguageModelAdapter, HuggingFaceCausalLMAdapter
from fair_mia.models.huggingface import cache_hf_model
from fair_mia.overlap import build_overlap_report
from fair_mia.progress import ProgressReporter
from fair_mia.registry import build_attack_registry, get_attacks
from fair_mia.reporting import prepare_run_dir, write_json, write_results_jsonl, write_summary_csv
from fair_mia.scenarios import FineTuningScenarioRunner, PretrainingScenarioRunner, RagScenarioRunner
from fair_mia.types import AttackRecord, TextSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fair-mia", description="Offline fairness-aware MIA benchmark scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an offline benchmark.")
    run_parser.add_argument("--config", required=True, help="Path to JSON config.")

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspect configured local data.")
    inspect_parser.add_argument("--config", required=True, help="Path to JSON config.")

    subparsers.add_parser("list-attacks", help="List registered attacks.")

    cache_parser = subparsers.add_parser("cache-model", help="Explicitly download/cache a Hugging Face causal LM.")
    cache_parser.add_argument("--model-id", required=True)
    cache_parser.add_argument("--cache-dir", required=True)
    cache_parser.add_argument("--revision")
    cache_parser.add_argument("--trust-remote-code", action="store_true")
    cache_parser.add_argument("--task", choices=("causal", "masked"), default="causal")

    pan_parser = subparsers.add_parser("prepare-pan", help="Convert a user-provided PAN-style CSV into canonical JSONL.")
    pan_parser.add_argument("--input-file", help="CSV file with text, group, and split columns.")
    pan_parser.add_argument("--input-dir", help="Directory containing a PAN-style CSV file.")
    pan_parser.add_argument("--output-dir", required=True)
    pan_parser.add_argument("--text-field", default="text")
    pan_parser.add_argument("--group-field", default="gender")
    pan_parser.add_argument("--split-field", default="split")
    pan_parser.add_argument("--member-split", default="train")
    pan_parser.add_argument("--id-field", default="sample_id")
    pan_parser.add_argument("--group0-value")
    pan_parser.add_argument("--group1-value")

    pan17_parser = subparsers.add_parser(
        "prepare-pan17-xml",
        help="Convert PAN 2017 author-profiling XML plus truth.txt into canonical JSONL.",
    )
    pan17_parser.add_argument("--train-dir", required=True, help="PAN 2017 training directory or its language subdirectory.")
    pan17_parser.add_argument("--test-dir", required=True, help="PAN 2017 test directory or its language subdirectory.")
    pan17_parser.add_argument("--output-dir", required=True)
    pan17_parser.add_argument("--lang", default="en")
    pan17_parser.add_argument("--tokenizer-model-id", default="EleutherAI/pythia-160m")
    pan17_parser.add_argument("--cache-dir")
    pan17_parser.add_argument("--window-tokens", type=int, default=256)
    pan17_parser.add_argument("--max-windows-per-bucket", type=int, default=250)
    pan17_parser.add_argument("--seed", type=int, default=0)

    pile_parser = subparsers.add_parser("prepare-pile-sample", help="Stream a capped Pile sample into canonical JSONL.")
    pile_parser.add_argument("--output-dir", required=True)
    pile_parser.add_argument("--max-members", type=int, default=500)
    pile_parser.add_argument("--max-nonmembers", type=int, default=500)
    pile_parser.add_argument("--cache-dir")
    pile_parser.add_argument("--subset")

    finetune_parser = subparsers.add_parser("finetune-lora", help="Fine-tune a small causal LM with LoRA on local JSONL.")
    finetune_parser.add_argument("--base-model-id", required=True)
    finetune_parser.add_argument("--train-jsonl", required=True)
    finetune_parser.add_argument("--output-dir", required=True)
    finetune_parser.add_argument("--cache-dir")
    finetune_parser.add_argument("--max-train-samples", type=int, default=200)
    finetune_parser.add_argument("--epochs", type=int, default=1)
    finetune_parser.add_argument("--learning-rate", type=float, default=0.0002)
    finetune_parser.add_argument("--max-length", type=int, default=512)
    finetune_parser.add_argument("--seed", type=int, default=0)
    finetune_parser.add_argument("--lora-rank", type=int, default=16)
    finetune_parser.add_argument("--lora-alpha", type=int, default=32)
    finetune_parser.add_argument("--lora-dropout", type=float, default=0.05)
    finetune_parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    finetune_parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    finetune_parser.add_argument("--no-gradient-checkpointing", action="store_true")
    finetune_parser.add_argument("--load-in-4bit", action="store_true")
    finetune_parser.add_argument("--target-module", action="append", dest="target_modules")
    finetune_parser.add_argument("--resume-from-checkpoint")

    pan17_multi_parser = subparsers.add_parser(
        "prepare-pan17",
        help="Prepare raw multilingual PAN 2017 token windows without rebalancing.",
    )
    pan17_multi_parser.add_argument("--train-dir", required=True)
    pan17_multi_parser.add_argument("--test-dir", required=True)
    pan17_multi_parser.add_argument("--output-dir", required=True)
    pan17_multi_parser.add_argument("--languages", default="en,es,pt,ar")
    pan17_multi_parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3-4B-Base")
    pan17_multi_parser.add_argument("--cache-dir")
    pan17_multi_parser.add_argument("--window-tokens", type=int, default=256)

    pan18_parser = subparsers.add_parser("prepare-pan18", help="Prepare PAN 2018 multilingual XML data.")
    pan18_parser.add_argument("--train-dir", required=True)
    pan18_parser.add_argument("--test-dir", required=True)
    pan18_parser.add_argument("--output-dir", required=True)
    pan18_parser.add_argument("--languages", default="en,es,ar")
    pan18_parser.add_argument("--tokenizer-model-id", default="Qwen/Qwen3-4B-Base")
    pan18_parser.add_argument("--cache-dir")
    pan18_parser.add_argument("--window-tokens", type=int, default=256)

    variants_parser = subparsers.add_parser(
        "make-variants",
        help="Create raw, size-matched, and balanced LoRA/evaluation variants.",
    )
    variants_parser.add_argument("--members", required=True)
    variants_parser.add_argument("--nonmembers", required=True)
    variants_parser.add_argument("--output-dir", required=True)
    variants_parser.add_argument("--seed", type=int, default=29)
    variants_parser.add_argument("--calibration-fraction", type=float, default=0.2)
    variants_parser.add_argument("--audit-cap-per-cell", type=int, default=100)

    plan_parser = subparsers.add_parser("plan", help="Resolve and display a YAML LoRA study plan.")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--study")

    sweep_parser = subparsers.add_parser("run-sweep", help="Run or resume a YAML LoRA study.")
    sweep_parser.add_argument("--config", required=True)
    sweep_parser.add_argument("--study")
    sweep_parser.add_argument("--resume", action="store_true")
    sweep_parser.add_argument("--retry-failed", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="Regenerate aggregate artifacts for a sweep.")
    aggregate_parser.add_argument("--run-dir", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check VM readiness without downloading assets.")
    doctor_parser.add_argument("--config")

    internal_parser = subparsers.add_parser("_execute-job", help="Internal isolated sweep worker.")
    internal_parser.add_argument("--job-file", required=True)
    internal_parser.add_argument("--invocation-dir", required=True)
    internal_parser.add_argument("--gpu-id", type=int)
    return parser


def load_samples_from_config(config: BenchmarkConfig) -> list[TextSample]:
    return load_benchmark_samples(
        config.dataset.members_path,
        config.dataset.nonmembers_path,
        default_scenario=config.dataset.default_scenario,
    )


def load_calibration_samples_from_config(config: BenchmarkConfig) -> list[TextSample]:
    if config.dataset.calibration_members_path is None or config.dataset.calibration_nonmembers_path is None:
        return []
    return load_benchmark_samples(
        config.dataset.calibration_members_path,
        config.dataset.calibration_nonmembers_path,
        default_scenario=config.dataset.default_scenario,
    )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    progress_path: str | Path | None = None,
    progress_label: str | None = None,
) -> Path:
    samples = load_samples_from_config(config)
    calibration_samples = load_calibration_samples_from_config(config)
    if "rag" in config.scenarios:
        raise NotImplementedError(RagScenarioRunner.UNSUPPORTED_MESSAGE)

    sample_summary = summarize_samples(samples)
    run_dir = prepare_run_dir(config.outputs_dir, config.run_id)
    write_json(
        _build_run_manifest_payload(config=config, sample_summary=sample_summary, status="running"),
        run_dir / "run_manifest.json",
    )

    attacks = get_attacks(config.attacks)
    progress_enabled = progress_path is not None or sys.stdout.isatty()
    progress = ProgressReporter(
        label=progress_label or config.run_id,
        total=(len(samples) + len(calibration_samples)) * len(attacks) * len(config.scenarios),
        path=progress_path or (run_dir / "progress.json"),
        enabled=progress_enabled,
    )
    progress.phase(
        "model_loading",
        f"target={config.model.model_id} reference={config.model.reference_model_id or config.model.model_id}",
    )
    if config.model.backend == "hf":
        target_model = HuggingFaceCausalLMAdapter(
            model_id=config.model.model_id,
            revision=config.model.revision,
            cache_dir=config.model.cache_dir,
            dtype=config.model.dtype,
            device_map=config.model.device_map,
            max_length=config.model.max_length,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
            adapter_path=config.model.adapter_path,
            load_in_4bit=config.model.load_in_4bit,
            perturbation_model_id=config.model.perturbation_model_id,
        )
        reference_model = HuggingFaceCausalLMAdapter(
            model_id=config.model.reference_model_id or config.model.model_id,
            revision=config.model.revision,
            cache_dir=config.model.cache_dir,
            dtype=config.model.dtype,
            device_map=config.model.device_map,
            max_length=config.model.max_length,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
    else:
        target_model = FakeLanguageModelAdapter(
            name=config.model.name,
            seed=config.seed,
            member_bias=config.model.member_bias,
            group_bias=config.model.group_bias,
        )
        reference_model = FakeLanguageModelAdapter(
            name=f"{config.model.name}-reference",
            seed=config.seed + 101,
            member_bias=config.model.reference_member_bias,
            group_bias=config.model.reference_group_bias,
        )
    defense = NoOpDefense()
    scenario_registry = {
        "pretraining": PretrainingScenarioRunner(),
        "finetuning": FineTuningScenarioRunner(),
        "rag": RagScenarioRunner(top_k=config.rag.top_k),
    }

    unknown_scenarios = sorted(set(config.scenarios) - set(scenario_registry))
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown_scenarios)}")

    records: list[AttackRecord] = []
    calibration_records: list[AttackRecord] = []

    def progress_callback(split_name: str):
        def report(sample_index: int, sample_total: int, attack_name: str, sample_id: str) -> None:
            progress.update(
                detail=(
                    f"{split_name} sample {sample_index}/{sample_total} "
                    f"attack={attack_name} id={sample_id}"
                )
            )

        return report

    for scenario_name in config.scenarios:
        progress.phase(
            "evaluation_scoring",
            f"scenario={scenario_name} samples={len(samples)} attacks={len(attacks)}",
        )
        records.extend(
            scenario_registry[scenario_name].run(
                samples,
                attacks,
                target_model,
                reference_model,
                defense,
                progress_callback=progress_callback("evaluation"),
            )
        )
        if calibration_samples:
            progress.phase(
                "calibration_scoring",
                f"scenario={scenario_name} samples={len(calibration_samples)} attacks={len(attacks)}",
            )
            calibration_records.extend(
                scenario_registry[scenario_name].run(
                    calibration_samples,
                    attacks,
                    target_model,
                    reference_model,
                    defense,
                    progress_callback=progress_callback("calibration"),
                )
            )

    progress.phase("reporting", "writing records and bootstrap metrics")
    write_results_jsonl(records, run_dir / "results.jsonl")
    if calibration_records:
        write_results_jsonl(calibration_records, run_dir / "calibration_results.jsonl")
    evaluation_raw = config.raw.get("evaluation", {})
    write_summary_csv(
        records,
        run_dir / "summary.csv",
        calibration_records,
        min_cell_members=int(evaluation_raw.get("min_cell_members", 30)),
        bootstrap_replicates=int(evaluation_raw.get("bootstrap_replicates", 0)),
        seed=config.seed,
    )
    write_json(
        _build_run_manifest_payload(config=config, sample_summary=sample_summary, status="scored"),
        run_dir / "run_manifest.json",
    )

    overlap_status = "disabled" if not config.overlap.enabled else "complete"
    progress.phase("overlap", "computing member/nonmember n-gram overlap diagnostics")
    try:
        overlap_report = build_overlap_report(
            samples,
            enabled=config.overlap.enabled,
            max_nonmembers=config.overlap.max_nonmembers,
            max_members_per_nonmember=config.overlap.max_members_per_nonmember,
            n_values=config.overlap.n_values,
            threshold=config.overlap.threshold,
        )
        overlap_status = str(overlap_report.get("status", overlap_status))
    except Exception as exc:
        overlap_status = "failed"
        overlap_report = {
            "status": "failed",
            "error": str(exc),
            "member_count": sample_summary["membership"].get("member", 0),
            "nonmember_count": sample_summary["membership"].get("nonmember", 0),
            "sampled": False,
            "evaluated_nonmembers": 0,
            "evaluated_members_per_nonmember": 0,
            "total_pairs_evaluated": 0,
            "n_grams": {},
        }
    write_json(overlap_report, run_dir / "overlap_report.json")
    write_json(
        _build_run_manifest_payload(
            config=config,
            sample_summary=sample_summary,
            status="complete" if overlap_status != "failed" else "complete_with_overlap_failure",
            overlap_status=overlap_status,
        ),
        run_dir / "run_manifest.json",
    )
    progress.finish("benchmark artifacts complete")
    return run_dir


def _build_run_manifest_payload(
    *,
    config: BenchmarkConfig,
    sample_summary: dict[str, object],
    status: str,
    overlap_status: str | None = None,
) -> dict[str, object]:
    payload = {
        "run_id": config.run_id,
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "status": status,
        "attacks": config.attacks,
        "scenarios": config.scenarios,
        "sample_summary": sample_summary,
        "config": config.raw,
    }
    if overlap_status is not None:
        payload["overlap_status"] = overlap_status
    return payload


def cmd_list_attacks() -> int:
    for name in sorted(build_attack_registry()):
        print(name)
    return 0


def cmd_inspect_data(config_path: str) -> int:
    config = load_config(config_path)
    samples = load_samples_from_config(config)
    payload = {
        "sample_summary": summarize_samples(samples),
        "overlap_report": build_overlap_report(samples),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_run(config_path: str) -> int:
    config = load_config(config_path)
    run_dir = run_benchmark(config)
    print(f"Wrote benchmark outputs to {run_dir}")
    return 0


def cmd_cache_model(args: argparse.Namespace) -> int:
    cache_hf_model(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        task=args.task,
    )
    print(f"Cached {args.model_id} under {args.cache_dir}")
    return 0


def cmd_prepare_pan(args: argparse.Namespace) -> int:
    input_file = args.input_file or _find_pan_csv(args.input_dir)
    members_path, nonmembers_path = prepare_pan_from_csv(
        input_path=input_file,
        output_dir=args.output_dir,
        text_field=args.text_field,
        group_field=args.group_field,
        split_field=args.split_field,
        member_split=args.member_split,
        id_field=args.id_field,
        group0_value=args.group0_value,
        group1_value=args.group1_value,
    )
    print(f"Wrote {members_path}")
    print(f"Wrote {nonmembers_path}")
    return 0


def _find_pan_csv(input_dir: str | None) -> str:
    if not input_dir:
        raise ValueError("prepare-pan requires either --input-file or --input-dir.")
    candidates = sorted(Path(input_dir).glob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one CSV file in {input_dir}, found {len(candidates)}.")
    return str(candidates[0])


def cmd_prepare_pile_sample(args: argparse.Namespace) -> int:
    members_path, nonmembers_path = prepare_pile_sample(
        output_dir=args.output_dir,
        max_members=args.max_members,
        max_nonmembers=args.max_nonmembers,
        cache_dir=args.cache_dir,
        subset=args.subset,
    )
    print(f"Wrote {members_path}")
    print(f"Wrote {nonmembers_path}")
    return 0


def cmd_prepare_pan17_xml(args: argparse.Namespace) -> int:
    members_path, nonmembers_path = prepare_pan17_from_xml(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        tokenizer_model_id=args.tokenizer_model_id,
        cache_dir=args.cache_dir,
        window_tokens=args.window_tokens,
        max_windows_per_bucket=args.max_windows_per_bucket,
        seed=args.seed,
    )
    print(f"Wrote {members_path}")
    print(f"Wrote {nonmembers_path}")
    return 0


def _languages(value: str) -> tuple[str, ...]:
    languages = tuple(part.strip() for part in value.split(",") if part.strip())
    if not languages:
        raise ValueError("At least one language must be provided.")
    return languages


def cmd_prepare_pan17(args: argparse.Namespace) -> int:
    members, nonmembers = prepare_pan17_multilingual(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        languages=_languages(args.languages),
        tokenizer_model_id=args.tokenizer_model_id,
        cache_dir=args.cache_dir,
        window_tokens=args.window_tokens,
    )
    print(f"Wrote {members}")
    print(f"Wrote {nonmembers}")
    return 0


def cmd_prepare_pan18(args: argparse.Namespace) -> int:
    members, nonmembers = prepare_pan18_from_xml(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        languages=_languages(args.languages),
        tokenizer_model_id=args.tokenizer_model_id,
        cache_dir=args.cache_dir,
        window_tokens=args.window_tokens,
    )
    print(f"Wrote {members}")
    print(f"Wrote {nonmembers}")
    return 0


def cmd_make_variants(args: argparse.Namespace) -> int:
    from fair_mia.variants import materialize_variants

    manifest = materialize_variants(
        members_path=args.members,
        nonmembers_path=args.nonmembers,
        output_dir=args.output_dir,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        audit_cap_per_cell=args.audit_cap_per_cell,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_plan(config_path: str, study: str | None) -> int:
    from fair_mia.study_config import load_study_config, resolve_experiments
    from fair_mia.sweep import build_execution_plan, render_plan

    config = load_study_config(config_path)
    plan = build_execution_plan(config, resolve_experiments(config, study_name=study))
    print(render_plan(plan))
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def cmd_run_sweep(
    config_path: str,
    study: str | None,
    resume: bool,
    retry_failed: bool,
) -> int:
    from fair_mia.sweep import run_sweep

    run_dir = run_sweep(
        config_path,
        study_name=study,
        resume=resume,
        retry_failed=retry_failed,
    )
    print(f"Wrote sweep outputs to {run_dir}")
    return 0


def cmd_aggregate(run_dir: str) -> int:
    from fair_mia.sweep import aggregate_run

    aggregate_run(run_dir)
    print(f"Aggregated {run_dir}")
    return 0


def cmd_doctor(config_path: str | None) -> int:
    payload: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / (1024**3), 2),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    try:
        import torch
    except ImportError:
        payload["torch"] = "missing"
        payload["cuda_available"] = False
        payload["gpu_count"] = 0
    else:
        payload["torch"] = getattr(torch, "__version__", "unknown")
        payload["cuda_available"] = torch.cuda.is_available()
        payload["gpu_count"] = torch.cuda.device_count()
        payload["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if config_path:
        from fair_mia.study_config import load_study_config, resolve_experiments
        from fair_mia.sweep import build_execution_plan

        config = load_study_config(config_path)
        resolved_experiments = resolve_experiments(config)
        plan = build_execution_plan(config, resolved_experiments)
        payload["config_valid"] = True
        payload["planned_jobs"] = plan["job_count"]
        missing_sources = sorted(
            {
                path
                for job in plan["jobs"]
                if job["stage"] == "dataset_preparation"
                for path in (job["members_path"], job["nonmembers_path"])
                if not Path(path).exists()
            }
        )
        payload["missing_dataset_sources"] = missing_sources
        gated_models = [
            model.model_id for model in config.models.values() if model.gated
        ]
        payload["gated_models"] = gated_models
        payload["gated_model_access_ready"] = not gated_models or bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        missing_model_caches: list[str] = []
        active_model_aliases = {
            experiment.model for experiment in resolved_experiments if not experiment.historical
        }
        for alias in sorted(active_model_aliases):
            model = config.models[alias]
            cache_dir = Path(model.cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = Path.cwd() / cache_dir
            cache_key = f"models--{model.model_id.replace('/', '--')}"
            if not (cache_dir / cache_key).exists():
                missing_model_caches.append(model.model_id)
        perturbation_cache = Path.cwd() / "artifacts" / "models" / "models--FacebookAI--xlm-roberta-base"
        if not perturbation_cache.exists():
            missing_model_caches.append("FacebookAI/xlm-roberta-base")
        payload["missing_model_caches"] = sorted(set(missing_model_caches))
    readiness_issues: list[str] = []
    if float(payload["disk_free_gb"]) < 150:
        readiness_issues.append("Less than the recommended 150 GB disk space is free.")
    if int(payload.get("gpu_count", 0)) < 2:
        readiness_issues.append("Fewer than two CUDA GPUs are available.")
    if payload.get("gated_model_access_ready") is False:
        readiness_issues.append("HF_TOKEN is required for configured gated models.")
    if payload.get("missing_dataset_sources"):
        readiness_issues.append("Prepared canonical dataset sources are missing.")
    if payload.get("missing_model_caches"):
        readiness_issues.append("One or more configured model caches are missing.")
    payload["ready"] = not readiness_issues
    payload["readiness_issues"] = readiness_issues
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_finetune_lora(args: argparse.Namespace) -> int:
    from fair_mia.finetune import finetune_lora

    finetune_lora(
        base_model_id=args.base_model_id,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        load_in_4bit=args.load_in_4bit,
        target_modules=args.target_modules,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Wrote LoRA adapter to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-attacks":
        return cmd_list_attacks()
    if args.command == "inspect-data":
        return cmd_inspect_data(args.config)
    if args.command == "run":
        return cmd_run(args.config)
    if args.command == "cache-model":
        return cmd_cache_model(args)
    if args.command == "prepare-pan":
        return cmd_prepare_pan(args)
    if args.command == "prepare-pan17-xml":
        return cmd_prepare_pan17_xml(args)
    if args.command == "prepare-pile-sample":
        return cmd_prepare_pile_sample(args)
    if args.command == "finetune-lora":
        return cmd_finetune_lora(args)
    if args.command == "prepare-pan17":
        return cmd_prepare_pan17(args)
    if args.command == "prepare-pan18":
        return cmd_prepare_pan18(args)
    if args.command == "make-variants":
        return cmd_make_variants(args)
    if args.command == "plan":
        return cmd_plan(args.config, args.study)
    if args.command == "run-sweep":
        return cmd_run_sweep(args.config, args.study, args.resume, args.retry_failed)
    if args.command == "aggregate":
        return cmd_aggregate(args.run_dir)
    if args.command == "doctor":
        return cmd_doctor(args.config)
    if args.command == "_execute-job":
        from fair_mia.sweep import execute_job_file

        execute_job_file(args.job_file, args.invocation_dir, args.gpu_id)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
