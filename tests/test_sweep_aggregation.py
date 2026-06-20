import csv
import json
import tempfile
import unittest
from pathlib import Path

from fair_mia.sweep import _select_pending_jobs, aggregate_run


class SweepAggregationTests(unittest.TestCase):
    def test_aggregate_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_hash = "abc123"
            job = {
                "job_hash": job_hash,
                "stage": "attack_evaluation",
                "study_name": "study",
                "dataset": "pan17",
                "model": "model",
                "training_variant": "balanced",
                "evaluation_variant": "balanced_audit",
                "evaluation_tier": "full",
                "seed": 29,
                "epochs": 4,
            }
            (root / "execution_plan.json").write_text(
                json.dumps({"jobs": [job], "resolved_experiments": [{}]}),
                encoding="utf-8",
            )
            job_dir = root / "jobs" / job_hash
            (job_dir / "score").mkdir(parents=True)
            (job_dir / "status.json").write_text(
                json.dumps({"status": "success", "job_hash": job_hash}),
                encoding="utf-8",
            )
            with (job_dir / "score" / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["scenario", "attack", "dimension", "scope", "auc_roc"],
                )
                writer.writeheader()
                writer.writerow({
                    "scenario": "finetuning",
                    "attack": "loss",
                    "dimension": "all",
                    "scope": "all",
                    "auc_roc": "0.6",
                })
            aggregate_run(root)
            for name in (
                "runs.jsonl",
                "failures.jsonl",
                "summary.csv",
                "metrics.csv",
                "comparisons.csv",
                "report.md",
            ):
                self.assertTrue((root / name).exists(), name)
            self.assertTrue((root / "studies" / "study" / "summary.csv").exists())

    def test_resume_skips_success_and_failed_until_retry_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = [{"job_hash": value} for value in ("success", "failed", "missing")]
            for name, status in (("success", "success"), ("failed", "failed")):
                status_path = root / "jobs" / name / "status.json"
                status_path.parent.mkdir(parents=True)
                status_path.write_text(json.dumps({"status": status}), encoding="utf-8")
            pending = _select_pending_jobs(jobs, root, resume=True, retry_failed=False)
            self.assertEqual([job["job_hash"] for job in pending], ["missing"])
            pending_retry = _select_pending_jobs(jobs, root, resume=True, retry_failed=True)
            self.assertEqual(
                [job["job_hash"] for job in pending_retry],
                ["failed", "missing"],
            )


if __name__ == "__main__":
    unittest.main()
