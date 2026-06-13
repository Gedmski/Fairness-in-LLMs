import unittest

from fair_mia.attacks.loss import LossAttack
from fair_mia.defenses import NoOpDefense
from fair_mia.models import FakeLanguageModelAdapter
from fair_mia.scenarios.rag import RagScenarioRunner
from fair_mia.types import TextSample


class RagAttributionTests(unittest.TestCase):
    def test_rag_records_retrieved_ids(self):
        samples = [
            TextSample("a", "alpha beta gamma", True, "G0", "rag"),
            TextSample("b", "alpha beta delta", False, "G1", "rag"),
        ]
        model = FakeLanguageModelAdapter(name="fake", seed=1)
        records = RagScenarioRunner(top_k=1).run(samples, [LossAttack()], model, model, NoOpDefense())
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertIn("retrieved_ids", record.diagnostics)
            self.assertEqual(len(record.diagnostics["retrieved_ids"]), 1)


if __name__ == "__main__":
    unittest.main()

