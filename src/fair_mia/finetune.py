"""VM-only fine-tuning helpers loaded behind optional research dependencies."""

from __future__ import annotations

from pathlib import Path


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
) -> None:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install research dependencies before LoRA fine-tuning: pip install -e .[research]") from exc

    from fair_mia.data import load_jsonl_samples

    samples = load_jsonl_samples(train_jsonl, is_member=True, default_scenario="finetuning")[:max_train_samples]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=str(cache_dir) if cache_dir else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=TextDataset())
    trainer.train()
    merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
