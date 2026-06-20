# VM Handoff: LoRA Studies

## Required VM

- Linux recommended
- Python 3.10+
- Two NVIDIA L4 GPUs
- At least 150 GB free disk
- PAN 2017 and PAN 2018 archives supplied separately
- Hugging Face token for gated Gemma runs

## 1. Setup

```bash
git clone <repository-url>
cd Fairness-in-LLMs
bash scripts/vm_setup.sh
```

Do not continue until `doctor` reports the expected GPUs and no unexplained readiness failures:

```bash
./.venv/bin/python -m fair_mia.cli doctor --config configs/lora_studies.yaml
```

Missing PAN canonical files and model caches are expected before the preparation and caching steps.

## 2. Cache Models

After accepting the Gemma terms and exporting `HF_TOKEN`:

```bash
bash scripts/cache_lora_assets.sh
```

The script caches Qwen3-4B-Base, OLMo-2-1B, Gemma-3-4B-PT when authorized, and the XLM-R masked model used by the multilingual neighborhood attack.

## 3. Prepare PAN Data

```bash
./.venv/bin/python -m fair_mia.cli prepare-pan17 \
  --train-dir /data/pan17/train \
  --test-dir /data/pan17/test \
  --output-dir data/pan17_multilingual \
  --languages en,es,pt,ar \
  --cache-dir artifacts/models

./.venv/bin/python -m fair_mia.cli prepare-pan18 \
  --train-dir /data/pan18/train \
  --test-dir /data/pan18/test \
  --output-dir data/pan18_multilingual \
  --languages en,es,ar \
  --cache-dir artifacts/models
```

Verify:

```bash
./.venv/bin/python -m fair_mia.cli doctor --config configs/lora_studies.yaml
```

## 4. Resolve the Experiment Graph

```bash
./.venv/bin/python -m fair_mia.cli plan --config configs/lora_studies.yaml
```

Review the experiment count, training-job count, scoring-job count, stable adapter paths, and GPU list before starting.

## 5. Smoke Test

```bash
./.venv/bin/python -m fair_mia.cli run-sweep \
  --config configs/lora_studies.yaml \
  --study smoke_olmo_lora
```

This is the required live VM validation. It cannot be completed on the offline development machine because it requires cached model weights and CUDA.

Check:

- Adapter directory contains `adapter_config.json`, adapter weights, tokenizer, `training_config.json`, `training_history.json`, and `environment.json`.
- Both full and audit scoring jobs succeeded.
- Ten attacks appear in the audit output.
- `failures.jsonl` is empty.

## 6. Main Studies

Run one study:

```bash
./.venv/bin/python -m fair_mia.cli run-sweep \
  --config configs/lora_studies.yaml \
  --study ablation_training_distribution_pan17
```

Run all enabled studies:

```bash
bash scripts/run_lora_studies.sh
```

The runner launches at most one isolated subprocess per configured GPU. Each subprocess receives one physical GPU through `CUDA_VISIBLE_DEVICES`.

## 7. Resume and Recovery

```bash
./.venv/bin/python -m fair_mia.cli run-sweep \
  --config configs/lora_studies.yaml \
  --resume
```

Resume uses the newest output directory with the same resolved configuration hash. Successful job hashes are skipped and missing or interrupted jobs are retried.

By default, failed hashes remain recorded and only missing/interrupted work is resumed. Retry failed jobs explicitly:

```bash
./.venv/bin/python -m fair_mia.cli run-sweep \
  --config configs/lora_studies.yaml \
  --resume \
  --retry-failed
```

Inspect a failure:

```text
outputs/<date>/<invocation>/jobs/<job_hash>/
  status.json
  stdout.log
  stderr.log
```

Set `runtime.continue_on_error: false` when a study should stop after the first failing stage.

## 8. Re-Aggregate

```bash
./.venv/bin/python -m fair_mia.cli aggregate \
  --run-dir outputs/<date>/<invocation>
```

Aggregation reads completed job artifacts only and does not load models.

## Operational Notes

- Qwen and Gemma default to 4-bit QLoRA in the supplied study config.
- OLMo uses BF16 LoRA.
- Model execution is offline after caching.
- Do not delete adapters until every evaluation variant and attack tier using that training hash has completed.
- Historical Pythia runs are indexed separately and must not be pooled with the new model results.
- RAG is outside this VM campaign.
