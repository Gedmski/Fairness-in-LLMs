# Fair MIA

Fair MIA is a study-driven benchmark for membership-inference risk and privacy disparity in fine-tuned causal language models. The current implementation focuses on LoRA/QLoRA experiments over PAN author-profiling data. RAG remains a deferred scenario.

## Implemented

- Dependency-free JSON/offline benchmark retained for local validation.
- YAML study runner with shared defaults, named studies, controlled sweeps, stable job hashes, two-GPU scheduling, durable status files, resume, aggregation, and failure logs.
- PAN 2017 multilingual preparation for English, Spanish, Portuguese, and Arabic.
- PAN 2018 preparation for English, Spanish, and Arabic.
- Raw, size-matched-random, and balanced training variants.
- Native-raw and balanced-audit evaluation variants with author-disjoint calibration/test splits.
- Binary gender, language, variety, and gender-by-language reporting.
- PEFT adapter-only LoRA, optional 4-bit QLoRA, gradient checkpointing, target-module detection, and training manifests.
- Ten attacks: `loss`, `reference`, `zlib`, `min_k`, `neighborhood`, `min_k_plus_plus`, `wbc`, `recall`, `samia`, and `spv_mia`.
- Calibrated privacy, classification, low-FPR, calibration, utility, disparity, and author-clustered bootstrap metrics.

## Local Validation

The existing offline path still requires no model download, GPU, or third-party runtime package:

```powershell
python -m fair_mia.cli list-attacks
python -m fair_mia.cli run --config configs/offline_demo.json
python -m unittest discover -s tests
```

`list-attacks` now returns exactly ten attacks. The fake adapter implements every declared capability so all attack contracts can be tested offline.

## VM Setup

Target profile: two NVIDIA L4 GPUs with 24 GB VRAM each and at least 150 GB free disk.

```bash
bash scripts/vm_setup.sh
```

or:

```powershell
.\scripts\vm_setup.ps1
```

The setup scripts install `.[research]`, run the complete offline suite, and execute:

```bash
fair-mia doctor --config configs/lora_studies.yaml
```

They do not download models or datasets.

## Data Preparation

PAN archives are user-provided. PAN data may require PAN/TIRA access, manual download, or archive credentials. Do not infer demographic attributes from text.

PAN 2017:

```bash
fair-mia prepare-pan17 \
  --train-dir /data/pan17/train \
  --test-dir /data/pan17/test \
  --output-dir data/pan17_multilingual \
  --languages en,es,pt,ar \
  --tokenizer-model-id Qwen/Qwen3-4B-Base \
  --cache-dir artifacts/models
```

PAN 2018:

```bash
fair-mia prepare-pan18 \
  --train-dir /data/pan18/train \
  --test-dir /data/pan18/test \
  --output-dir data/pan18_multilingual \
  --languages en,es,ar \
  --tokenizer-model-id Qwen/Qwen3-4B-Base \
  --cache-dir artifacts/models
```

Preparation creates exact 256-token windows. Author identity, gender, language, variety, dataset, and source split remain explicit in JSONL.

Variants can also be materialized manually:

```bash
fair-mia make-variants \
  --members data/pan17_multilingual/members.jsonl \
  --nonmembers data/pan17_multilingual/nonmembers.jsonl \
  --output-dir data/variants/pan17_multilingual/seed_29 \
  --seed 29
```

Only selected training windows are marked as members. Unselected windows are never relabeled as exposed training examples.

## Model Caching

Review and accept the Gemma license before caching it, then set `HF_TOKEN`.

```bash
bash scripts/cache_lora_assets.sh
```

Equivalent explicit commands:

```bash
fair-mia cache-model --model-id Qwen/Qwen3-4B-Base --cache-dir artifacts/models
fair-mia cache-model --model-id allenai/OLMo-2-0425-1B --cache-dir artifacts/models
fair-mia cache-model --model-id google/gemma-3-4b-pt --cache-dir artifacts/models
fair-mia cache-model --model-id FacebookAI/xlm-roberta-base --task masked --cache-dir artifacts/models
```

Sweep scoring uses `local_files_only=true`; no experiment job downloads assets unexpectedly.

## Study Workflow

The primary experiment contract is [configs/lora_studies.yaml](configs/lora_studies.yaml).

Resolve without running:

```bash
fair-mia plan --config configs/lora_studies.yaml
```

Run the fully open smoke study first:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml --study smoke_olmo_lora
```

If you need a faster mechanics-only smoke before the full ten-attack validation, use:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml --study smoke_olmo_lora_fast
```

If the full PAN 2017 distribution ablation is too expensive for an exploratory run, use the overnight pilot:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml --study pilot_training_distribution_pan17
```

Run all enabled studies:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml
```

Resume the latest invocation with the same configuration hash:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml --resume
```

Failed hashes are left unchanged by default. Retry them explicitly:

```bash
fair-mia run-sweep --config configs/lora_studies.yaml --resume --retry-failed
```

Regenerate reports without rerunning models:

```bash
fair-mia aggregate --run-dir outputs/YYYY-MM-DD/HHMMSS-configHash
```

The configured studies are:

- `legacy_pythia_reference`
- `smoke_olmo_lora`
- `ablation_training_distribution_pan17`
- `robustness_pan18`
- `ablation_model_family`
- `ablation_training_exposure`

Gemma PAN 2018 and the full model-by-distribution matrix are present but disabled.

## Attack Tiers

The full evaluation set runs seven reusable-score attacks:

- `loss`
- `reference`
- `zlib`
- `min_k`
- `neighborhood`
- `min_k_plus_plus`
- `wbc`

The common intersectionally stratified audit subset runs all ten, adding:

- `recall`
- `samia`
- `spv_mia`

The audit cap is 100 samples per language × gender × membership cell: at most 1,600 PAN 2017 samples and 1,200 PAN 2018 samples.

## Outputs

Each invocation writes:

```text
outputs/YYYY-MM-DD/<invocation_id>/
  resolved_config.yaml
  execution_plan.json
  runs.jsonl
  failures.jsonl
  summary.csv
  metrics.csv
  comparisons.csv
  report.md
  plots/
  studies/<study_name>/
  jobs/<stable_job_hash>/
```

Every job writes `status.json`, `stdout.log`, and `stderr.log` immediately. Failed runs include the resolved job, stage, GPU, timestamps, reason, and traceback.

Active scoring jobs also write `progress.json` and stream progress bars to the sweep terminal. The progress snapshot includes the current phase, completed and total sample-attack units, percentage, elapsed time, ETA, processing rate, and current sample/attack detail. Watch all active progress snapshots with:

```bash
watch -n 10 'find outputs -name progress.json -print -exec cat {} \;'
```

`summary.csv` contains aggregate rows. `metrics.csv` contains aggregate, gender, language, variety, and intersectional rows. Statistical cells with fewer than 30 members or 30 nonmembers are marked suppressed. `comparisons.csv` records matched raw-versus-balanced AUC differences.

## Interpretation

- Treat Pythia runs as historical references, not part of the modern-model aggregate.
- Report cross-attack and cross-seed patterns; do not promote one attack-specific result into a general conclusion.
- PAN 2018 does not supply Portuguese in this experiment design. Portuguese analysis comes from PAN 2017.
- The Pile helper remains a mechanics-only pretraining path because its subgroup labels are synthetic.
- RAG execution remains intentionally blocked until context-conditioned retrieval scoring is implemented.

See [docs/VM_HANDOFF.md](docs/VM_HANDOFF.md) and [docs/DATASETS_AND_MODELS.md](docs/DATASETS_AND_MODELS.md) for operational details.
