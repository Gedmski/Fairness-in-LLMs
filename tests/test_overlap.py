import unittest

from fair_mia.overlap import overlap_ratio


class OverlapTests(unittest.TestCase):
    def test_exact_duplicate_has_full_overlap(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        self.assertEqual(overlap_ratio(text, text, 4), 1.0)

    def test_unrelated_text_has_low_overlap(self):
        left = "alpha beta gamma delta epsilon zeta eta theta"
        right = "train museum weather ticket market soup library archive"
        self.assertEqual(overlap_ratio(left, right, 4), 0.0)


if __name__ == "__main__":
    unittest.main()

