import math
import unittest

from fair_mia.models import FakeLanguageModelAdapter
from fair_mia.models.base import LanguageModelAdapter
from fair_mia.registry import build_attack_registry
from fair_mia.types import TextSample, TokenScores


class LossOnlyAdapter(LanguageModelAdapter):
    name = "loss-only"

    def score_tokens(self, text, sample=None):
        return TokenScores(tokens=text.split(), losses=[1.0] * len(text.split()))


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

    def test_registry_contains_exactly_ten_attacks(self):
        self.assertEqual(
            set(build_attack_registry()),
            {
                "loss",
                "reference",
                "zlib",
                "min_k",
                "neighborhood",
                "min_k_plus_plus",
                "wbc",
                "recall",
                "samia",
                "spv_mia",
            },
        )

    def test_capability_failure_is_explicit(self):
        sample = TextSample("s1", "alpha beta gamma", True, "G0", "finetuning")
        with self.assertRaisesRegex(ValueError, "token_distributions"):
            build_attack_registry()["min_k_plus_plus"].score(sample, LossOnlyAdapter())


if __name__ == "__main__":
    unittest.main()
