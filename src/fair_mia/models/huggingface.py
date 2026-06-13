"""Lazy Hugging Face causal language-model adapter for VM runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fair_mia.models.base import LanguageModelAdapter
from fair_mia.types import TextSample, TokenScores


class MissingResearchDependencyError(RuntimeError):
    """Raised when optional research dependencies are not installed."""


class HuggingFaceCausalLMAdapter(LanguageModelAdapter):
    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        max_length: int = 512,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        model: Any | None = None,
        tokenizer: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.name = model_id
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.dtype = dtype
        self.device_map = device_map
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module

    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        if hasattr(self._model, "score_token_losses"):
            tokens, losses = self._model.score_token_losses(text, sample)
            return TokenScores(tokens=list(tokens), losses=[float(loss) for loss in losses])

        assert self._torch is not None

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        device = getattr(self._model, "device", None)
        if device is not None and hasattr(input_ids, "to"):
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        with self._torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        if input_ids.shape[-1] < 2:
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            return TokenScores(tokens=tokens, losses=[0.0 for _ in tokens])

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = self._torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        losses = (-token_log_probs).detach().cpu().reshape(-1).tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(shift_labels.detach().cpu().reshape(-1).tolist())
        clean_losses = [float(loss) if math.isfinite(float(loss)) else 0.0 for loss in losses]
        return TokenScores(tokens=tokens, losses=clean_losses)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MissingResearchDependencyError(
                "Install research dependencies before using Hugging Face models: pip install -e .[research]"
            ) from exc

        torch_dtype = self._resolve_dtype(torch)
        common_kwargs = {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **common_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            **common_kwargs,
        )
        self._model.eval()
        self._torch = torch

    def _resolve_dtype(self, torch_module: Any) -> Any:
        if self.dtype == "auto":
            return "auto"
        if self.dtype in {"float16", "fp16"}:
            return torch_module.float16
        if self.dtype in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if self.dtype in {"float32", "fp32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported dtype: {self.dtype}")


def cache_hf_model(
    *,
    model_id: str,
    cache_dir: str | Path,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MissingResearchDependencyError(
            "Install research dependencies before caching models: pip install -e .[research]"
        ) from exc

    kwargs = {
        "revision": revision,
        "cache_dir": str(cache_dir),
        "trust_remote_code": trust_remote_code,
    }
    AutoTokenizer.from_pretrained(model_id, **kwargs)
    AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
