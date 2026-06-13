# Datasets and Models

## Canonical JSONL Schema

Every benchmark run consumes two JSONL files: one for members and one for non-members.

Required fields:

- `sample_id`: unique string identifier
- `text`: text to score
- `is_member`: boolean
- `group`: binary group label, exactly `G0` or `G1`
- `scenario`: `pretraining`, `finetuning`, `rag`, or another explicit scenario name
- `metadata`: optional object

The loader rejects empty text, duplicate IDs, non-binary groups, and datasets that do not contain exactly two groups.

## Dataset Paths

PAN-style data is treated as user-provided. Convert a CSV with explicit text, group, and split columns:

```powershell
fair-mia prepare-pan --input-file data/raw/pan.csv --output-dir data/pan_demo --group-field gender --member-split train
```

Pile sampling is optional and capped. The helper streams a small sample through Hugging Face `datasets`; it does not download the full Pile:

```powershell
fair-mia prepare-pile-sample --output-dir data/pile_sample --max-members 500 --max-nonmembers 500 --cache-dir artifacts/datasets
```

The current Pile helper assigns synthetic `G0`/`G1` labels because The Pile does not provide demographic labels suitable for fairness claims. Use it for pretraining-style MIA mechanics, not demographic conclusions.

## Supported Model IDs

Recommended order for the 2 x NVIDIA L4 VM:

- `EleutherAI/pythia-160m`: smoke tests and first real scoring run
- `EleutherAI/pythia-410m`: main small run
- `EleutherAI/pythia-1b`: larger run if throughput is acceptable
- `EleutherAI/pythia-2.8b`: optional with `device_map="auto"`
- `EleutherAI/pythia-6.9b`: optional, expect slower runs and tighter memory

Do not make `12B` a default. Use capped sample sizes before expanding.

