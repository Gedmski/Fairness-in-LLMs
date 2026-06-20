import contextlib
import io
import json
import csv
import tempfile
import unittest
from pathlib import Path

from fair_mia.cli import cmd_inspect_data, cmd_list_attacks, run_benchmark
from fair_mia.config import load_config


class OfflinePipelineTests(unittest.TestCase):
    def test_cli_list_and_inspect_complete(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_list_attacks(), 0)
            self.assertEqual(cmd_inspect_data("configs/offline_demo.json"), 0)

    def test_offline_run_writes_expected_outputs(self):
        config = load_config("configs/offline_demo.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            patched_raw = dict(config.raw)
            patched_raw["outputs_dir"] = temp_dir
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps(patched_raw), encoding="utf-8")
            patched_config = load_config(config_path)

            run_dir = run_benchmark(patched_config)

            self.assertTrue((run_dir / "results.jsonl").exists())
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "overlap_report.json").exists())
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertGreater((run_dir / "results.jsonl").stat().st_size, 0)
            with (run_dir / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("balanced_accuracy", reader.fieldnames)
                self.assertIn("majority_class_accuracy", reader.fieldnames)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["sample_summary"]["is_four_cell_balanced"], True)

    def test_run_writes_disabled_overlap_stub(self):
        config = load_config("configs/offline_demo.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            patched_raw = dict(config.raw)
            patched_raw["outputs_dir"] = temp_dir
            patched_raw["overlap"] = {"enabled": False}
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps(patched_raw), encoding="utf-8")
            patched_config = load_config(config_path)

            run_dir = run_benchmark(patched_config)

            overlap = json.loads((run_dir / "overlap_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(overlap["status"], "disabled")
            self.assertEqual(manifest["overlap_status"], "disabled")
            self.assertEqual(manifest["status"], "complete")

    def test_run_writes_completed_progress_snapshot(self):
        config = load_config("configs/offline_demo.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            patched_raw = dict(config.raw)
            patched_raw["outputs_dir"] = temp_dir
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps(patched_raw), encoding="utf-8")
            progress_path = Path(temp_dir) / "job-progress.json"

            with contextlib.redirect_stdout(io.StringIO()):
                run_benchmark(
                    load_config(config_path),
                    progress_path=progress_path,
                    progress_label="offline-test",
                )

            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["phase"], "complete")
            self.assertEqual(progress["completed"], progress["total"])
            self.assertEqual(progress["percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
