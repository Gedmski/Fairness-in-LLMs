"""VM-only fine-tuning helpers loaded behind optional research dependencies."""

from __future__ import annotations

import random
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fair_mia.types import TextSample


def finetune_lora(
    *,
    base_model_id: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    max_train_samples: int = 200,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    seed: int = 0,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    gradient_checkpointing: bool = True,
    load_in_4bit: bool = False,
    target_modules: list[str] | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> None:
    try:
        import torch
        import peft
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install research dependencies before LoRA fine-tuning: pip install -e .[research]") from exc

    from fair_mia.data import load_jsonl_samples

    samples = _select_stratified_training_samples(
        load_jsonl_samples(train_jsonl, is_member=True, default_scenario="finetuning"),
        max_train_samples=max_train_samples,
        seed=seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=str(cache_dir) if cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_kwargs,
    )
    resolved_target_modules = target_modules or _detect_lora_target_modules(model)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolved_target_modules,
        ),
    )

    class TextDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            encoded = tokenizer(
                samples[index].text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=seed,
        data_seed=seed,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=gradient_checkpointing,
    )
    trainer = Trainer(model=model, args=args, train_dataset=TextDataset())
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    training_config = {
        "base_model_id": base_model_id,
        "train_jsonl": str(train_jsonl),
        "max_train_samples": max_train_samples,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "load_in_4bit": load_in_4bit,
        "target_modules": resolved_target_modules,
        "selected_sample_ids": [sample.sample_id for sample in samples],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(
            {
                "metrics": dict(train_result.metrics),
                "log_history": list(trainer.state.log_history),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": getattr(torch, "__version__", "unknown"),
                "transformers": getattr(transformers, "__version__", "unknown"),
                "peft": getattr(peft, "__version__", "unknown"),
                "base_model_revision": getattr(
                    getattr(model, "config", None),
                    "_commit_hash",
                    None,
                ),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _detect_lora_target_modules(model: object) -> list[str]:
    preferred = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query_key_value",
        "dense",
    )
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    resolved = [name for name in preferred if name in module_names]
    if not resolved:
        raise ValueError(
            "Could not auto-detect LoRA target modules. Pass explicit target_modules for this architecture."
        )
    return resolved


def _select_stratified_training_samples(
    samples: list[TextSample],
    *,
    max_train_samples: int,
    seed: int,
) -> list[TextSample]:
    if max_train_samples <= 0:
        return []

    buckets: dict[tuple[str, str, str], list[TextSample]] = defaultdict(list)
    for sample in samples:
        variety = str(sample.metadata.get("variety", "unknown")).strip().lower() or "unknown"
        language = str(
            sample.attributes.get("language") or sample.metadata.get("lang", "unknown")
        ).strip().lower() or "unknown"
        buckets[(sample.group, language, variety)].append(sample)

    rng = random.Random(seed)
    active_keys = sorted(buckets)
    for key in active_keys:
        rng.shuffle(buckets[key])

    selected: list[TextSample] = []
    remaining = {key: list(values) for key, values in buckets.items()}
    while active_keys and len(selected) < min(max_train_samples, len(samples)):
        next_active: list[tuple[str, str, str]] = []
        for key in active_keys:
            bucket = remaining[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_active.append(key)
            if len(selected) >= min(max_train_samples, len(samples)):
                break
        active_keys = next_active

    return selected
