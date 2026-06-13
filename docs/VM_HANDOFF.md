# VM Handoff

## Prerequisites

- Python 3.10+
- Two NVIDIA L4 GPUs for the intended profile
- CUDA-compatible PyTorch installation through `pip install -e ".[research]"`
- Disk space for model caches under `artifacts/models`
- Dataset space under `data/`

## Setup

Windows PowerShell:

```powershell
.\scripts\vm_setup.ps1
```

Linux shell:

```bash
bash scripts/vm_setup.sh
```

These scripts install dependencies and run offline tests. They do not download models.

## Data Preparation

For PAN-style local data:

```powershell
fair-mia prepare-pan --input-file data/raw/pan.csv --output-dir data/pan_demo --group-field gender --member-split train
```

If the directory contains exactly one CSV, this equivalent form also works:

```powershell
fair-mia prepare-pan --input-dir data/raw/pan --output-dir data/pan_demo --group-field gender --member-split train
```

For a capped Pile mechanics sample:

```powershell
fair-mia prepare-pile-sample --output-dir data/pile_sample --max-members 500 --max-nonmembers 500 --cache-dir artifacts/datasets
```

PAN data may require manual access approval. Do not infer sensitive attributes from free text; use explicit metadata.

## Model Caching

Download models only with explicit cache commands:

```powershell
fair-mia cache-model --model-id EleutherAI/pythia-160m --cache-dir artifacts/models
fair-mia cache-model --model-id EleutherAI/pythia-410m --cache-dir artifacts/models
```

Run configs use `local_files_only=true`, so benchmark execution will fail clearly if a model has not been cached.

## Runs

Offline validation:

```powershell
fair-mia run --config configs/offline_demo.json
```

Real-model smoke run:

```powershell
fair-mia run --config configs/vm_smoke_pythia_160m.json
```

RAG run:

```powershell
fair-mia run --config configs/vm_rag_local.json
```

LoRA fine-tuning:

```powershell
fair-mia finetune-lora --base-model-id EleutherAI/pythia-160m --train-jsonl data/pan_demo/members.jsonl --output-dir artifacts/adapters/pythia_160m_lora --cache-dir artifacts/models --max-train-samples 200 --epochs 1
fair-mia run --config configs/vm_finetune_lora_pythia_160m.json
```

Main Pythia run:

```powershell
fair-mia run --config configs/vm_main_pythia_410m.json
```

## Outputs

Each run writes:

- `outputs/<run_id>/results.jsonl`
- `outputs/<run_id>/summary.csv`
- `outputs/<run_id>/overlap_report.json`
- `outputs/<run_id>/run_manifest.json`

Review `summary.csv` for `all`, `G0`, `G1`, and `gap:G0-G1` rows.

## Troubleshooting

- If model loading fails, run `cache-model` first and confirm `cache_dir` matches the config.
- If CUDA is unavailable, start with `EleutherAI/pythia-160m`, lower sample counts, or use CPU only for smoke tests.
- If PAN conversion fails, check that the CSV has text, group, and split columns and exactly two group values.
- Full Pile download is not required for this project; keep sample caps small until the pipeline is validated.
