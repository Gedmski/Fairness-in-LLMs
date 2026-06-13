import json
import tempfile
import unittest
from pathlib import Path

from fair_mia.config import load_config
from fair_mia.data import load_benchmark_samples, prepare_pan_from_csv


class DataPreparationTests(unittest.TestCase):
    def test_prepare_pan_from_csv_writes_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "pan.csv"
            input_path.write_text(
                "sample_id,text,gender,split\n"
                "a,alpha beta,male,train\n"
                "b,gamma delta,female,train\n"
                "c,epsilon zeta,male,test\n"
                "d,eta theta,female,test\n",
                encoding="utf-8",
            )
            members_path, nonmembers_path = prepare_pan_from_csv(
                input_path=input_path,
                output_dir=Path(temp_dir) / "out",
                text_field="text",
                group_field="gender",
                split_field="split",
                member_split="train",
            )
            samples = load_benchmark_samples(members_path, nonmembers_path, default_scenario="finetuning")
            self.assertEqual(len(samples), 4)
            self.assertEqual({sample.group for sample in samples}, {"G0", "G1"})

    def test_non_binary_group_validation_rejects_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            members = Path(temp_dir) / "members.jsonl"
            nonmembers = Path(temp_dir) / "nonmembers.jsonl"
            rows = [
                {"sample_id": "a", "text": "alpha", "is_member": True, "group": "G0"},
                {"sample_id": "b", "text": "beta", "is_member": False, "group": "G2"},
            ]
            members.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            nonmembers.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_benchmark_samples(members, nonmembers, default_scenario="x")

    def test_missing_dataset_path_config_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "bad",
                        "seed": 1,
                        "outputs_dir": temp_dir,
                        "dataset": {"members_path": "missing_members.jsonl", "nonmembers_path": "missing_nonmembers.jsonl"},
                        "attacks": ["loss"],
                        "scenarios": ["pretraining"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()

