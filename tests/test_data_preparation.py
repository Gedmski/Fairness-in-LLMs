import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fair_mia.config import load_config
from fair_mia.data import load_benchmark_samples, prepare_pan17_from_xml, prepare_pan_from_csv


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(token_ids)


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

    def test_prepare_pan17_from_xml_writes_balanced_token_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train" / "en"
            test_dir = Path(temp_dir) / "test" / "en"
            self._write_pan17_fixture(train_dir, test_dir)

            with patch("fair_mia.data._load_pan17_tokenizer", return_value=FakeTokenizer()):
                members_path, nonmembers_path = prepare_pan17_from_xml(
                    train_dir=train_dir.parent,
                    test_dir=test_dir.parent,
                    output_dir=Path(temp_dir) / "out",
                    lang="en",
                    tokenizer_model_id="fake-tokenizer",
                    window_tokens=3,
                    max_windows_per_bucket=2,
                    seed=11,
                )

            samples = load_benchmark_samples(members_path, nonmembers_path, default_scenario="finetuning")
            self.assertEqual(len(samples), 8)
            self.assertEqual(sum(1 for sample in samples if sample.is_member), 4)
            self.assertEqual(sum(1 for sample in samples if not sample.is_member), 4)
            self.assertEqual({sample.group for sample in samples}, {"G0", "G1"})
            self.assertEqual({sample.metadata["variety"] for sample in samples}, {"canada"})
            self.assertTrue(all(":w" in sample.sample_id for sample in samples))
            self.assertTrue(all(sample.metadata["token_count"] == 3 for sample in samples))
            self.assertTrue(all(len(sample.text.split()) == 3 for sample in samples))
            self.assertTrue(all(sample.metadata["tokenizer_model_id"] == "fake-tokenizer" for sample in samples))

    def test_prepare_pan17_from_xml_is_seed_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train" / "en"
            test_dir = Path(temp_dir) / "test" / "en"
            self._write_pan17_fixture(train_dir, test_dir)

            with patch("fair_mia.data._load_pan17_tokenizer", return_value=FakeTokenizer()):
                out_a = Path(temp_dir) / "out_a"
                out_b = Path(temp_dir) / "out_b"
                prepare_pan17_from_xml(
                    train_dir=train_dir.parent,
                    test_dir=test_dir.parent,
                    output_dir=out_a,
                    lang="en",
                    tokenizer_model_id="fake-tokenizer",
                    window_tokens=3,
                    max_windows_per_bucket=2,
                    seed=7,
                )
                prepare_pan17_from_xml(
                    train_dir=train_dir.parent,
                    test_dir=test_dir.parent,
                    output_dir=out_b,
                    lang="en",
                    tokenizer_model_id="fake-tokenizer",
                    window_tokens=3,
                    max_windows_per_bucket=2,
                    seed=7,
                )

            self.assertEqual((out_a / "members.jsonl").read_text(encoding="utf-8"), (out_b / "members.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((out_a / "nonmembers.jsonl").read_text(encoding="utf-8"), (out_b / "nonmembers.jsonl").read_text(encoding="utf-8"))

    def test_prepare_pan17_from_xml_rejects_missing_balanced_bucket(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train" / "en"
            test_dir = Path(temp_dir) / "test" / "en"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            (train_dir / "truth.txt").write_text(
                "g0m:::female:::canada\n"
                "g1m:::male:::canada\n",
                encoding="utf-8",
            )
            (test_dir / "truth.txt").write_text(
                "g0n:::female:::canada\n",
                encoding="utf-8",
            )
            self._write_author_xml(train_dir / "g0m.xml", "a b c d e f")
            self._write_author_xml(train_dir / "g1m.xml", "g h i j k l")
            self._write_author_xml(test_dir / "g0n.xml", "m n o p q r")

            with patch("fair_mia.data._load_pan17_tokenizer", return_value=FakeTokenizer()):
                with self.assertRaises(ValueError):
                    prepare_pan17_from_xml(
                        train_dir=train_dir.parent,
                        test_dir=test_dir.parent,
                        output_dir=Path(temp_dir) / "out",
                        lang="en",
                        tokenizer_model_id="fake-tokenizer",
                        window_tokens=3,
                        max_windows_per_bucket=2,
                        seed=0,
                    )

    def _write_pan17_fixture(self, train_dir: Path, test_dir: Path) -> None:
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        (train_dir / "truth.txt").write_text(
            "g0m_canada:::female:::canada\n"
            "g1m_canada:::male:::canada\n"
            "g0m_us:::female:::us\n"
            "g1m_us:::male:::us\n",
            encoding="utf-8",
        )
        (test_dir / "truth.txt").write_text(
            "g0n_canada:::female:::canada\n"
            "g1n_canada:::male:::canada\n"
            "g0n_us:::female:::us\n",
            encoding="utf-8",
        )
        self._write_author_xml(train_dir / "g0m_canada.xml", "c0 c1 c2 c3 c4 c5 c6 c7 c8")
        self._write_author_xml(train_dir / "g1m_canada.xml", "d0 d1 d2 d3 d4 d5 d6 d7 d8")
        self._write_author_xml(test_dir / "g0n_canada.xml", "e0 e1 e2 e3 e4 e5 e6 e7 e8")
        self._write_author_xml(test_dir / "g1n_canada.xml", "f0 f1 f2 f3 f4 f5 f6 f7 f8")
        self._write_author_xml(train_dir / "g0m_us.xml", "u0 u1 u2 u3 u4 u5")
        self._write_author_xml(train_dir / "g1m_us.xml", "v0 v1 v2 v3 v4 v5")
        self._write_author_xml(test_dir / "g0n_us.xml", "w0 w1 w2 w3 w4 w5")

    def _write_author_xml(self, path: Path, text: str) -> None:
        documents = "".join(f"<document><![CDATA[{token}]]></document>" for token in text.split())
        path.write_text(f"<author lang=\"en\"><documents>{documents}</documents></author>", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
