# Datasets and Models

## Canonical JSONL

Required fields:

- `sample_id`
- `text`
- `is_member`
- `group`: `G0` or `G1`
- `scenario`
- `metadata`
- `attributes`

Fine-tuning records carry:

- `gender`
- `language`
- `variety`
- `dataset`
- `author_id`
- `training_variant`
- `exposure_count`

`group` remains the compatibility field for binary gender analysis. Multi-class analysis uses explicit attributes and never infers them from text.

## Membership Definition

A record is a member only if its exact token window was supplied to the trainer. The variant builder records selected sample IDs and sets `exposure_count=1` for those windows. Held-out and unselected records remain nonmembers.

Authors are split before metric calibration:

- 20% of authors per membership × gender × language cell: threshold calibration
- 80%: final test

No author may occur in both partitions.

## Training Variants

- `raw_full`: every eligible training window.
- `size_matched_random`: seeded random sample equal to the balanced training-set size.
- `balanced`: equal gender × language counts among selected training members.

## Evaluation Variants

- `native_raw`: all selected member windows plus a size-matched natural-distribution nonmember sample.
- `balanced_audit`: matched gender × language × membership cells.

Each variant also contains an expensive-attack audit subset capped at 100 samples per gender × language × membership cell.

## PAN 2017

Configured languages:

- English
- Spanish
- Portuguese
- Arabic

PAN 2017 supplies binary gender and language-variety metadata. It is the source for four-language and regional-variety analyses.

Official task/download information: <https://pan.webis.de/clef17/pan17-web/author-profiling.html>

## PAN 2018

Configured languages:

- English
- Spanish
- Arabic

PAN 2018 is a robustness dataset. It does not supply the requested Portuguese condition; Portuguese results come from PAN 2017.

Official task/download information: <https://pan.webis.de/clef18/pan18-web/author-profiling.html>

PAN corpora are not downloaded automatically. Use official PAN/TIRA distribution channels and comply with the corpus terms.

## Models

### Primary

`Qwen/Qwen3-4B-Base`

- Primary multilingual fine-tuning model
- 4-bit QLoRA default
- Runs on one L4, allowing two independent jobs concurrently
- Model card: <https://huggingface.co/Qwen/Qwen3-4B-Base>

### Replication

`google/gemma-3-4b-pt`

- Multilingual model-family replication
- Gated; requires accepted terms and `HF_TOKEN`
- 4-bit QLoRA default
- Model card: <https://huggingface.co/google/gemma-3-4b-pt>

### Fully Open Smoke/Scale

`allenai/OLMo-2-0425-1B`

- Fully open model path
- Required first VM smoke study
- BF16 LoRA default
- Model card: <https://huggingface.co/allenai/OLMo-2-0425-1B>

### Historical

`EleutherAI/pythia-160m`

- Existing three-seed results remain indexed
- Not pooled with modern-model studies because their member construction predates exact exposure tracking

### Perturbation

`FacebookAI/xlm-roberta-base`

- Masked language model used to create multilingual neighborhood substitutions
- Cached explicitly with `cache-model --task masked`

## WikiMIA and The Pile

The Pile helper remains available for pretraining mechanics, but its synthetic groups cannot support demographic conclusions. A future WikiMIA importer may be used for attack conformance only; it must not be treated as proof of membership for unrelated modern checkpoints.
