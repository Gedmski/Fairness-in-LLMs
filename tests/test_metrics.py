import unittest

from fair_mia.metrics import (
    auc_roc,
    best_threshold_accuracy,
    best_threshold_balanced_accuracy,
    majority_class_accuracy,
    privacy_leakage_disparity,
    tpr_at_fpr,
)


class MetricTests(unittest.TestCase):
    def test_auc_roc_known_ordering(self):
        labels = [True, True, False, False]
        scores = [0.9, 0.8, 0.2, 0.1]
        self.assertEqual(auc_roc(labels, scores), 1.0)

    def test_tpr_at_fpr_known_ordering(self):
        labels = [True, True, False, False]
        scores = [0.9, 0.8, 0.2, 0.1]
        self.assertEqual(tpr_at_fpr(labels, scores, 0.0), 1.0)

    def test_best_threshold_accuracy(self):
        labels = [True, False, True, False]
        scores = [0.9, 0.7, 0.6, 0.1]
        self.assertEqual(best_threshold_accuracy(labels, scores), 0.75)

    def test_balanced_accuracy_stays_near_chance_for_majority_prediction(self):
        labels = [True, True, True, False]
        scores = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(best_threshold_accuracy(labels, scores), 0.75)
        self.assertEqual(best_threshold_balanced_accuracy(labels, scores), 0.5)

    def test_majority_class_accuracy(self):
        labels = [True, True, True, False, False]
        self.assertEqual(majority_class_accuracy(labels), 0.6)

    def test_privacy_leakage_disparity(self):
        self.assertAlmostEqual(privacy_leakage_disparity({"G0": 0.7, "G1": 0.4}), 0.3)


if __name__ == "__main__":
    unittest.main()
