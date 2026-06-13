"""Command-line entry point for the offline benchmark scaffold."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fair_mia import __version__
from fair_mia.config import BenchmarkConfig, load_config
from fair_mia.data import (
    load_benchmark_samples,
    prepare_pan_from_csv,
    prepare_pan17_from_xml,
    prepare_pile_sample,
    summarize_samples,
)
from fair_mia.defenses import NoOpDefense
from fair_mia.models import FakeLanguageModelAdapter, HuggingFaceCausalLMAdapter
from fair_mia.models.huggingface import cache_hf_model
from fair_mia.overlap import build_overlap_report
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
    return parser


def load_samples_from_config(config: BenchmarkConfig) -> list[TextSample]:
    return load_benchmark_samples(
        config.dataset.members_path,
        config.dataset.nonmembers_path,
        default_scenario=config.dataset.default_scenario,
    )


def run_benchmark(config: BenchmarkConfig) -> Path:
    samples = load_samples_from_config(config)
    if "rag" in config.scenarios:
        raise NotImplementedError(RagScenarioRunner.UNSUPPORTED_MESSAGE)

    sample_summary = summarize_samples(samples)
    run_dir = prepare_run_dir(config.outputs_dir, config.run_id)
    write_json(
        _build_run_manifest_payload(config=config, sample_summary=sample_summary, status="running"),
        run_dir / "run_manifest.json",
    )

    attacks = get_attacks(config.attacks)
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
    for scenario_name in config.scenarios:
        records.extend(
            scenario_registry[scenario_name].run(samples, attacks, target_model, reference_model, defense)
        )

    write_results_jsonl(records, run_dir / "results.jsonl")
    write_summary_csv(records, run_dir / "summary.csv")
    write_json(
        _build_run_manifest_payload(config=config, sample_summary=sample_summary, status="scored"),
        run_dir / "run_manifest.json",
    )

    overlap_status = "disabled" if not config.overlap.enabled else "complete"
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
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
