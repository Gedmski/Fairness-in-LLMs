import json
import tempfile
import unittest
from pathlib import Path

from fair_mia.data import load_jsonl_samples, write_jsonl_samples
from fair_mia.types import TextSample
from fair_mia.variants import materialize_variants


class VariantTests(unittest.TestCase):
    def _samples(self, is_member):
        samples = []
        split_name = "member" if is_member else "nonmember"
        for language in ("en", "es"):
            for group, gender in (("G0", "female"), ("G1", "male")):
                for author_index in range(5):
                    for window in range(author_index + 1):
                        samples.append(
                            TextSample(
                                sample_id=f"{language}:{group}:{split_name}:a{author_index}:w{window}",
                                text=f"{language} {group} author {author_index} window {window}",
                                is_member=is_member,
                                group=group,
                                scenario="finetuning",
                                metadata={"author_id": f"{language}:{group}:{split_name}:a{author_index}"},
                                attributes={
                                    "gender": gender,
                                    "language": language,
                                    "dataset": "fixture",
                                },
                            )
                        )
        return samples

    def test_materialized_variants_preserve_exact_membership_and_author_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            members_path = root / "members.jsonl"
            nonmembers_path = root / "nonmembers.jsonl"
            write_jsonl_samples(self._samples(True), members_path)
            write_jsonl_samples(self._samples(False), nonmembers_path)
            manifest = materialize_variants(
                members_path=members_path,
                nonmembers_path=nonmembers_path,
                output_dir=root / "variants",
                seed=29,
                calibration_fraction=0.2,
                audit_cap_per_cell=2,
            )
            variants = manifest["variants"]
            self.assertLessEqual(
                variants["balanced"]["training_samples"],
                variants["raw_full"]["training_samples"],
            )
            eval_dir = root / "variants" / "balanced" / "evaluation" / "balanced_audit"
            calibration = load_jsonl_samples(
                eval_dir / "calibration_members.jsonl",
                is_member=True,
                default_scenario="finetuning",
            )
            test = load_jsonl_samples(
                eval_dir / "test_members.jsonl",
                is_member=True,
                default_scenario="finetuning",
            )
            calibration_authors = {sample.attributes["author_id"] for sample in calibration}
            test_authors = {sample.attributes["author_id"] for sample in test}
            self.assertFalse(calibration_authors & test_authors)
            train_rows = [
                json.loads(line)
                for line in (root / "variants" / "size_matched_random" / "train_members.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(row["is_member"] for row in train_rows))
            self.assertTrue(all(row["metadata"]["exposure_count"] == 1 for row in train_rows))


if __name__ == "__main__":
    unittest.main()
