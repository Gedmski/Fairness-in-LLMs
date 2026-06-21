import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fair_mia.models.huggingface import HuggingFaceCausalLMAdapter, MissingResearchDependencyError
from fair_mia.types import TextSample


class MockTokenizer:
    pass


class MockModel:
    def score_token_losses(self, text, sample):
        return text.split(), [0.1, 0.2, 0.3]

    def eval(self):
        return self


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

    def test_4bit_loading_uses_quantization_config(self):
        captured = {}

        class MockAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return MockTokenizer()

        class MockAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                captured.update(kwargs)
                return MockModel()

        class MockBitsAndBytesConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=MockAutoModelForCausalLM,
            AutoTokenizer=MockAutoTokenizer,
            BitsAndBytesConfig=MockBitsAndBytesConfig,
        )
        fake_torch = SimpleNamespace(bfloat16="bf16", float32="fp32", cuda=SimpleNamespace(is_available=lambda: True))
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return fake_torch
            if name == "transformers":
                return fake_transformers
            return real_import(name, globals, locals, fromlist, level)

        adapter = HuggingFaceCausalLMAdapter(model_id="mock", load_in_4bit=True)
        with patch("builtins.__import__", side_effect=fake_import):
            adapter._ensure_loaded()

        self.assertNotIn("load_in_4bit", captured)
        self.assertIn("quantization_config", captured)
        self.assertTrue(captured["quantization_config"].kwargs["load_in_4bit"])


if __name__ == "__main__":
    unittest.main()
