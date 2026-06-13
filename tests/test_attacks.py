import math
import unittest

from fair_mia.models import FakeLanguageModelAdapter
from fair_mia.registry import build_attack_registry
from fair_mia.types import TextSample


class AttackTests(unittest.TestCase):
    def test_all_attacks_return_deterministic_finite_scores(self):
        sample = TextSample("s1", "alpha beta gamma delta epsilon", True, "G0", "pretraining")
        target = FakeLanguageModelAdapter(name="target", seed=3, member_bias=-0.4)
        reference = FakeLanguageModelAdapter(name="reference", seed=4, member_bias=-0.1)

        for attack in build_attack_registry().values():
            first = attack.score(sample, target, reference)
            second = attack.score(sample, target, reference)
            self.assertEqual(first.raw_score, second.raw_score)
            self.assertEqual(first.membership_score, second.membership_score)
            self.assertTrue(math.isfinite(first.raw_score))
            self.assertTrue(math.isfinite(first.membership_score))


if __name__ == "__main__":
    unittest.main()

