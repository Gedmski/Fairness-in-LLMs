import unittest
from unittest.mock import patch

from fair_mia.models.huggingface import HuggingFaceCausalLMAdapter, MissingResearchDependencyError
from fair_mia.types import TextSample


class MockTokenizer:
    pass


class MockModel:
    def score_token_losses(self, text, sample):
        return text.split(), [0.1, 0.2, 0.3]


class HuggingFaceAdapterTests(unittest.TestCase):
    def test_injected_mock_model_scores_without_download(self):
        adapter = HuggingFaceCausalLMAdapter(
            model_id="mock",
            model=MockModel(),
            tokenizer=MockTokenizer(),
            torch_module=object(),
        )
        scores = adapter.score_tokens("alpha beta gamma", TextSample("s", "alpha beta gamma", True, "G0", "pretraining"))
        self.assertEqual(scores.tokens, ["alpha", "beta", "gamma"])
        self.assertEqual(scores.losses, [0.1, 0.2, 0.3])

    def test_missing_dependencies_fail_clearly(self):
        adapter = HuggingFaceCausalLMAdapter(model_id="EleutherAI/pythia-160m", local_files_only=True)
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"torch", "transformers"}:
                raise ImportError(f"mocked missing dependency: {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(MissingResearchDependencyError):
                adapter.score_tokens("alpha beta")


if __name__ == "__main__":
    unittest.main()
