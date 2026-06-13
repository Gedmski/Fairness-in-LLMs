import unittest

from fair_mia.attacks.loss import LossAttack
from fair_mia.defenses import NoOpDefense
from fair_mia.models import FakeLanguageModelAdapter
from fair_mia.scenarios.rag import RagScenarioRunner
from fair_mia.types import TextSample


class RagAttributionTests(unittest.TestCase):
    def test_rag_scenario_is_explicitly_blocked_until_context_is_scored(self):
        samples = [
            TextSample("a", "alpha beta gamma", True, "G0", "rag"),
            TextSample("b", "alpha beta delta", False, "G1", "rag"),
        ]
        model = FakeLanguageModelAdapter(name="fake", seed=1)
        with self.assertRaises(NotImplementedError):
            RagScenarioRunner(top_k=1).run(samples, [LossAttack()], model, model, NoOpDefense())


if __name__ == "__main__":
    unittest.main()
