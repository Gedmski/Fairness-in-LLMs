import unittest

from fair_mia.overlap import build_disabled_overlap_report, build_overlap_report, overlap_ratio
from fair_mia.types import TextSample


class OverlapTests(unittest.TestCase):
    def test_exact_duplicate_has_full_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        self.assertEqual(overlap_ratio(text, text, 4), 1.0)

    def test_unrelated_text_has_low_overlap(self):
        left = "alpha beta gamma delta epsilon zeta eta theta"
        right = "train museum weather ticket market soup library archive"
        self.assertEqual(overlap_ratio(left, right, 4), 0.0)

    def test_disabled_overlap_report_is_structurally_complete(self):
        samples = [
            TextSample("m0", "alpha beta gamma delta", True, "G0", "pretraining"),
            TextSample("n0", "alpha beta gamma epsilon", False, "G1", "pretraining"),
        ]
        report = build_disabled_overlap_report(samples)
        self.assertEqual(report["status"], "disabled")
        self.assertEqual(report["evaluated_nonmembers"], 0)
        self.assertEqual(report["n_grams"], {})

    def test_bounded_overlap_report_uses_sampling_caps(self):
        samples = [
            TextSample(f"m{index}", f"alpha beta gamma member {index}", True, "G0", "pretraining")
            for index in range(3)
        ] + [
            TextSample(f"n{index}", f"alpha beta gamma nonmember {index}", False, "G1", "pretraining")
            for index in range(2)
        ]
        report = build_overlap_report(
            samples,
            max_nonmembers=1,
            max_members_per_nonmember=2,
            n_values=(4,),
            threshold=0.8,
        )
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["sampled"])
        self.assertEqual(report["evaluated_nonmembers"], 1)
        self.assertEqual(report["evaluated_members_per_nonmember"], 2)
        self.assertEqual(report["total_pairs_evaluated"], 2)
        self.assertIn("4", report["n_grams"])


if __name__ == "__main__":
    unittest.main()
