import contextlib
import io
import json
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


if __name__ == "__main__":
    unittest.main()
