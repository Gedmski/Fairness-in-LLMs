"""Output artifact writers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fair_mia.metrics import summarize_records
from fair_mia.types import AttackRecord


def prepare_run_dir(outputs_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(outputs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_results_jsonl(records: list[AttackRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_summary_csv(
    records: list[AttackRecord],
    path: str | Path,
    calibration_records: list[AttackRecord] | None = None,
    *,
    min_cell_members: int = 30,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
) -> None:
    rows = summarize_records(
        records,
        calibration_records,
        min_cell_members=min_cell_members,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    fieldnames = list(rows[0]) if rows else [
        "scenario", "attack", "dimension", "scope", "samples", "members", "nonmembers"
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
