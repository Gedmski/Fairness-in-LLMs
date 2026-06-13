# Fair MIA

Fair MIA is a research benchmark for auditing membership inference attack (MIA) risk across binary demographic subgroups in LLM workflows. It is built to compare the three scenarios defined in the PRD/TRD:

- **Pre-training style evaluation** on open causal LMs such as Pythia
- **Fine-tuning / instruction-tuning** on smaller demographic or domain datasets
- **Retrieval-augmented generation (RAG)** where leakage can come from retrieved context, once context-conditioned scoring is implemented

The project keeps the offline scaffold runnable without model downloads, while providing VM-ready commands for real Hugging Face models and user-provided datasets.

## Current Status

Implemented:

- Canonical JSONL dataset pipeline with strict `G0`/`G1` binary group validation
- Five baseline attacks: `loss`, `reference`, `zlib`, `min_k`, and `neighborhood`
- Scenario runners for `pretraining` and `finetuning`
- Explicit RAG safety error until context-conditioned scoring is implemented
- Offline fake model adapter for local validation
- Lazy Hugging Face causal LM adapter for VM runs
- Explicit model caching command so benchmark runs do not download models unexpectedly
- PAN-style CSV conversion into canonical member/nonmember JSONL
- Capped Pile streaming helper for pre-training mechanics experiments
- LoRA fine-tuning helper for small VM runs
- Per-group reporting with `all`, `G0`, `G1`, and `gap:G0-G1` rows
- Output artifacts: `results.jsonl`, `summary.csv`, `overlap_report.json`, and `run_manifest.json`

Not implemented as defaults:

- Full Pile download
- Automatic PAN download
- Large Pythia `12B` runs
- Strong claims from one attack family

## Repository Layout

```text
configs/                 Offline and VM run configs
docs/                    PRD, TRD, VM handoff, dataset/model notes
examples/                Tiny offline JSONL examples
scripts/                 VM setup scripts
src/fair_mia/            Benchmark package
tests/                   Stdlib unittest suite
```

## Local Offline Validation

The offline path does not require PyTorch, Transformers, Datasets, GPUs, or internet access.

```powershell
python -m fair_mia.cli list-attacks
python -m fair_mia.cli inspect-data --config configs/offline_demo.json
python -m fair_mia.cli run --config configs/offline_demo.json
python -m unittest discover -s tests
```

If `python` is not on PATH in the Codex desktop environment, use the bundled Python path shown by Codex.

## VM Setup

Target VM profile: **2 x NVIDIA L4 GPUs**.

Windows PowerShell:

```powershell
.\scripts\vm_setup.ps1
```

Linux shell:

```bash
bash scripts/vm_setup.sh
```

These scripts create a virtual environment, install `.[research]`, print CUDA/GPU information, and run the offline tests. They do **not** download models.

## Dataset Preparation

### PAN-Style Demographic Dataset

PAN data is treated as user-provided because access may require manual approval. The converter expects a CSV with text, binary group metadata, and a split column.

```powershell
fair-mia prepare-pan --input-file data/raw/pan.csv --output-dir data/pan_demo --group-field gender --member-split train
```

If the input directory contains exactly one CSV:

```powershell
fair-mia prepare-pan --input-dir data/raw/pan --output-dir data/pan_demo --group-field gender --member-split train
```

The converter maps the two observed group values to `G0` and `G1`. Use `--group0-value` and `--group1-value` when the CSV has more than two possible values or when you need explicit mapping.

PAN 2017 author-profiling releases are often distributed as per-author XML files plus `truth.txt` instead of a flat CSV. Use the dedicated converter for that layout:

```powershell
fair-mia prepare-pan17-xml --train-dir C:\path\to\pan17-author-profiling-training-dataset-2017-03-10 --test-dir C:\path\to\pan17-author-profiling-test-dataset-2017-03-16 --output-dir data/pan_demo --lang en --tokenizer-model-id EleutherAI/pythia-160m --window-tokens 256 --max-windows-per-bucket 250 --seed 0
```

This converter treats the training authors as members, the test authors as non-members, creates exact model-token windows, balances by `group x variety x membership`, and maps `female -> G0` and `male -> G1`.

### Capped Pile Sample

The Pile helper is for pre-training-style MIA mechanics. It streams a capped sample and does not download the full dataset.

```powershell
fair-mia prepare-pile-sample --output-dir data/pile_sample --max-members 500 --max-nonmembers 500 --cache-dir artifacts/datasets
```

The Pile does not provide demographic labels for fairness claims. The helper assigns synthetic `G0`/`G1` labels only so the pipeline can run; use PAN-style or another metadata-rich dataset for demographic conclusions.

## Model Caching

Model downloads happen only through explicit cache commands.

```powershell
fair-mia cache-model --model-id EleutherAI/pythia-160m --cache-dir artifacts/models
fair-mia cache-model --model-id EleutherAI/pythia-410m --cache-dir artifacts/models
```

VM configs set `local_files_only=true`, so `run` fails clearly if a model has not been cached.

Recommended order:

1. `EleutherAI/pythia-160m` for smoke tests
2. `EleutherAI/pythia-410m` for the first main run
3. `EleutherAI/pythia-1b` if throughput is acceptable
4. `EleutherAI/pythia-2.8b` or `6.9b` only after smaller runs are stable

## VM Runs

Offline baseline:

```powershell
fair-mia run --config configs/offline_demo.json
```

Real-model smoke run after preparing `data/pan_demo` and caching `pythia-160m`:

```powershell
fair-mia run --config configs/vm_smoke_pythia_160m.json
```

Main Pythia mechanics run after preparing `data/pile_sample` and caching `pythia-410m` plus `pythia-160m`:

```powershell
fair-mia run --config configs/vm_main_pythia_410m.json
```

Recommended PAN 2017 LoRA fine-tuning run:

```powershell
fair-mia finetune-lora --base-model-id EleutherAI/pythia-160m --train-jsonl data/pan_demo/members.jsonl --output-dir artifacts/adapters/pythia_160m_lora --cache-dir artifacts/models --max-train-samples 1000 --epochs 2 --max-length 256 --seed 0
fair-mia run --config configs/vm_finetune_lora_pythia_160m.json
```

## Outputs

Each run writes to `outputs/<run_id>/`:

- `results.jsonl`: per-sample attack scores and diagnostics
- `summary.csv`: aggregate, per-group, PLD/gap metrics, plus balanced and majority-class accuracy columns
- `overlap_report.json`: 4-gram, 7-gram, and 13-gram overlap diagnostics
- `run_manifest.json`: config, seed, sample summary, attacks, scenarios, and version

Read `summary.csv` first. Every attack/scenario should have `all`, `G0`, `G1`, and `gap:G0-G1` rows.

## Alignment With PRD/TRD

- LLM target: real runs use Hugging Face causal LMs, with Pythia configs for the VM.
- Multiple MIA families: all five baseline attacks are implemented and run through the same registry.
- Binary attributes: canonical JSONL validation accepts only `G0` and `G1`.
- Scenario comparison: configs cover pre-training, fine-tuning, and RAG.
- Fuzzy boundary controls: overlap diagnostics are produced for every run.
- General-result framing: outputs support cross-attack comparison instead of relying on one attack.
- VM constraints: model downloads are explicit, sample caps are configurable, and full Pile download is avoided.

## Troubleshooting

- `ModuleNotFoundError` for `torch`, `transformers`, or `datasets`: run the VM setup script or `pip install -e ".[research]"`.
- `prepare-pile-sample` fails with `Dataset scripts are no longer supported`: install a compatible Hugging Face Datasets release with `pip install "datasets<5"` and retry.
- Model not found during `run`: execute `fair-mia cache-model` for the model ID and cache dir used by the config.
- PAN conversion fails: confirm the CSV has text, group, and split columns, and exactly two group values or explicit `--group0-value` / `--group1-value`.
- RAG benchmark execution fails with an unsupported error: this is intentional until retrieved context is injected into model scoring.
- CUDA memory pressure: reduce sample count, lower `max_length`, use `pythia-160m`, or keep only one scenario per run.
- Unexpected fairness signal on Pile sample: treat it as a pipeline/mechanics run unless the dataset has real sensitive attributes.

For the full VM sequence, see [docs/VM_HANDOFF.md](docs/VM_HANDOFF.md). For dataset/model details, see [docs/DATASETS_AND_MODELS.md](docs/DATASETS_AND_MODELS.md).
