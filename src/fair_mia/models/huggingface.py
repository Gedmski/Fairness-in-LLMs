"""Lazy Hugging Face causal language-model adapter for VM runs."""

from __future__ import annotations

import math
import hashlib
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
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = False,
        perturbation_model_id: str = "FacebookAI/xlm-roberta-base",
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
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.load_in_4bit = load_in_4bit
        self.perturbation_model_id = perturbation_model_id
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._score_cache: dict[tuple[str, str], TokenScores] = {}
        self._mask_model: Any | None = None
        self._mask_tokenizer: Any | None = None

    @property
    def capabilities(self) -> frozenset[LanguageModelAdapter.Capability]:
        return frozenset(LanguageModelAdapter.Capability)

    def score_tokens(self, text: str, sample: TextSample | None = None) -> TokenScores:
        cache_key = (text, sample.sample_id if sample is not None else "")
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        if hasattr(self._model, "score_token_losses"):
            tokens, losses = self._model.score_token_losses(text, sample)
            result = TokenScores(tokens=list(tokens), losses=[float(loss) for loss in losses])
            self._score_cache[cache_key] = result
            return result

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
            result = TokenScores(tokens=tokens, losses=[0.0 for _ in tokens])
            self._score_cache[cache_key] = result
            return result

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = self._torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        losses = (-token_log_probs).detach().cpu().reshape(-1).tolist()
        distribution_means = (-log_probs.mean(dim=-1)).detach().cpu().reshape(-1).tolist()
        distribution_stds = log_probs.std(dim=-1).detach().cpu().reshape(-1).tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(shift_labels.detach().cpu().reshape(-1).tolist())
        clean_losses = [float(loss) if math.isfinite(float(loss)) else 0.0 for loss in losses]
        clean_means = [float(value) if math.isfinite(float(value)) else 0.0 for value in distribution_means]
        clean_stds = [max(float(value), 1e-8) if math.isfinite(float(value)) else 1.0 for value in distribution_stds]
        result = TokenScores(
            tokens=tokens,
            losses=clean_losses,
            distribution_means=clean_means,
            distribution_stds=clean_stds,
        )
        self._score_cache[cache_key] = result
        return result

    def score_conditional(
        self,
        prefix: str,
        text: str,
        sample: TextSample | None = None,
    ) -> TokenScores:
        self._ensure_loaded()
        assert self._tokenizer is not None
        prefix_ids = self._tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
        combined = self.score_tokens(f"{prefix}\n{text}", sample)
        trim = min(len(prefix_ids), len(combined.losses))
        return TokenScores(
            tokens=combined.tokens[trim:],
            losses=combined.losses[trim:],
            distribution_means=combined.distribution_means[trim:],
            distribution_stds=combined.distribution_stds[trim:],
        )

    def generate(self, prompt: str, *, max_new_tokens: int, seed: int = 0) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        device = getattr(self._model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        return self._tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)

    def multilingual_neighbors(self, text: str, count: int) -> list[str]:
        self._ensure_mask_model_loaded()
        assert self._mask_tokenizer is not None
        assert self._mask_model is not None
        assert self._torch is not None
        words = text.split()
        if not words:
            return [text] * count
        neighbors: list[str] = []
        for index in range(count):
            digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).hexdigest()
            position = int(digest[:8], 16) % len(words)
            masked_words = list(words)
            masked_words[position] = self._mask_tokenizer.mask_token
            encoded = self._mask_tokenizer(
                " ".join(masked_words),
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            mask_positions = (encoded["input_ids"] == self._mask_tokenizer.mask_token_id).nonzero(as_tuple=False)
            if len(mask_positions) == 0:
                neighbors.append(text)
                continue
            with self._torch.no_grad():
                logits = self._mask_model(**encoded).logits
            batch_index, token_index = mask_positions[0].tolist()
            candidates = logits[batch_index, token_index].topk(5).indices.tolist()
            replacement = next(
                (
                    self._mask_tokenizer.decode([token_id]).strip()
                    for token_id in candidates
                    if self._mask_tokenizer.decode([token_id]).strip()
                    and self._mask_tokenizer.decode([token_id]).strip().lower() != words[position].lower()
                ),
                words[position],
            )
            masked_words[position] = replacement
            neighbors.append(" ".join(masked_words))
        return neighbors

    def _ensure_mask_model_loaded(self) -> None:
        if self._mask_model is not None and self._mask_tokenizer is not None:
            return
        self._ensure_loaded()
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise MissingResearchDependencyError(
                "Install transformers before using multilingual neighborhood attacks."
            ) from exc
        kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        self._mask_tokenizer = AutoTokenizer.from_pretrained(self.perturbation_model_id, **kwargs)
        self._mask_model = AutoModelForMaskedLM.from_pretrained(self.perturbation_model_id, **kwargs)
        self._mask_model.eval()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        model_kwargs = dict(common_kwargs)
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            **model_kwargs,
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise MissingResearchDependencyError(
                    "Install PEFT before loading a LoRA adapter: pip install -e .[research]"
                ) from exc
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path, is_trainable=False)
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
    task: str = "causal",
) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
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
    if task == "causal":
        AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    elif task == "masked":
        AutoModelForMaskedLM.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError("task must be either 'causal' or 'masked'.")
