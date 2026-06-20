# Supervisor Update: PAN 2017 Fine-Tuning MIA Benchmark

> Historical-result note: this report describes the June 2026 Pythia-160M
> experiments, which ran the original five attacks. The current repository now
> registers ten attacks and uses the YAML LoRA study framework documented in
> `README.md`. These historical results remain references and are not pooled
> with the modern Qwen, Gemma, or OLMo studies.

## Executive Summary

This document summarizes the current state of the `Fairness-in-LLMs` benchmark, the methodology implemented in the repository, and the results from the latest three-seed PAN 2017 fine-tuning experiments:

- `outputs/vm_finetune_lora_pythia_160m_seed29_ep4/`
- `outputs/vm_finetune_lora_pythia_160m_seed41_ep4/`
- `outputs/vm_finetune_lora_pythia_160m_seed53_ep4/`

The main finding is that the benchmark now produces a scientifically defensible fine-tuning result on a balanced PAN 2017 English dataset. Across three random seeds, membership inference signal is weak but consistent for the strongest attacks, especially `loss` and `min_k`. The observed subgroup gaps are small, but they are not purely artifacts of class imbalance because the evaluation set is exactly balanced and the report now includes `balanced_accuracy` and `majority_class_accuracy`.

This is meaningful progress. It is not yet a final headline result.

## Existing Project Documents

The repository already contains a coherent research and implementation stack:

- [README.md](../README.md): main runbook, current benchmark status, commands, and output interpretation
- [docs/prd_fair_privacy_llm.md](./prd_fair_privacy_llm.md): project scope, research questions, and scenario framing
- [docs/technical_requirements.md](./technical_requirements.md): benchmark architecture, metrics, scenarios, and output requirements
- [docs/DATASETS_AND_MODELS.md](./DATASETS_AND_MODELS.md): canonical JSONL schema, dataset policy, and recommended model ladder
- [docs/VM_HANDOFF.md](./VM_HANDOFF.md): VM setup and execution path

Taken together, these documents define the benchmark as:

- LLM-specific rather than generic tabular MIA work
- Binary subgroup reporting with explicit `G0` and `G1`
- Comparison across `pretraining`, `finetuning`, and `rag`
- Multiple MIA families rather than a claim built on one attack
- Overlap-aware evaluation because fuzzy membership boundaries matter for text

One documentation note is worth surfacing clearly: the runnable fine-tuning workflow is currently centered on PAN 2017 English XML, while the older PRD/TRD text still mentions PAN 2018 in a few places. For presentation purposes, the README plus VM handoff reflect the current executable path more accurately than the older dataset wording in the PRD/TRD.

## Paper Support for the Current Direction

The current benchmark design is well aligned with the three papers already stored in `papers/`:

### 1. LLM pretraining MIA is often weak

Paper:
- `papers/Do Membership Inference AttacksWork on Large Language Models_.pdf`

Relevance:
- This paper reports that MIAs on pre-trained Pythia models over The Pile often perform near random.
- It attributes weak performance to limited effective memorization, large-scale training data, and fuzzy member versus non-member boundaries.
- It also warns that apparent MIA success can come from distribution shift or temporal mismatch rather than clean membership leakage.

Why this matters here:
- It supports treating pretraining as a baseline where weak aggregate signal is expected.
- It also justifies the benchmark's overlap diagnostics and the decision not to oversell near-random results.
- It is consistent with the current repository position that fine-tuning is the most promising next scenario for clearer fairness analysis.

### 2. Fairness can redistribute privacy risk

Paper:
- `papers/21-On the Privacy Risks of Algorithmic Fairness.pdf`

Relevance:
- This paper argues that fairness interventions and privacy risk interact.
- It shows that privacy cost can be distributed unevenly across subgroups.
- It also motivates subgroup-specific analysis instead of relying only on a single global threshold or a single aggregate score.

Why this matters here:
- It supports the benchmark requirement to report `all`, `G0`, `G1`, and `gap:G0-G1`.
- It justifies evaluating privacy and fairness jointly rather than only asking whether an attack works on average.

### 3. Privacy leakage disparity should be measured explicitly

Paper:
- `papers/22-Understanding Disparate Effects.pdf`

Relevance:
- This paper formalizes privacy-leakage disparity, or PLD.
- It shows that subgroup-level privacy risk can differ even when aggregate privacy risk looks modest.
- It also motivates future defense studies, because defenses can redistribute protection unevenly across groups.

Why this matters here:
- It directly supports the benchmark's PLD-style group gap reporting.
- It motivates the next phase after attack benchmarking: defense benchmarking.

## Methodology Implemented in the Current PAN 2017 Run

### Dataset construction

The latest meaningful experiments use PAN 2017 English author profiling data, prepared through the repository's PAN 2017 XML pipeline instead of the older "one author equals one sample" approach.

Current preparation policy:

- Training authors are treated as `member`
- Test authors are treated as `nonmember`
- `female -> G0`
- `male -> G1`
- Author documents are concatenated, tokenized once with the target model tokenizer, then split into exact non-overlapping 256-token windows
- Short tails are dropped
- Windows are matched and downsampled by `group x variety x membership`
- Sampling is deterministic under a fixed seed

This produces a far better benchmark dataset than the earlier PAN demo setup because it:

- matches the TRD requirement for fixed token windows
- avoids sequence-length confounding
- avoids arbitrary class imbalance
- controls for English variety across member and nonmember buckets

### Evaluation set used in the three runs

All three runs use the same balanced evaluation structure:

- Total windows: `6000`
- Members: `3000`
- Nonmembers: `3000`
- `G0`: `3000`
- `G1`: `3000`
- Six English varieties:
  - `australia`
  - `canada`
  - `great britain`
  - `ireland`
  - `new zealand`
  - `united states`
- Each `group x variety x membership` bucket contains `250` windows

This is important because it means the reported `accuracy` is no longer inflated by class imbalance. The `majority_class_accuracy` baseline is `0.5` throughout these runs.

### Fine-tuning setup

The current fine-tuning experiments use:

- Base model: `EleutherAI/pythia-160m`
- Fine-tuning method: LoRA
- Training cap: `1000` member windows
- Epochs: `4`
- Max length: `256`
- Seeds: `29`, `41`, `53`

The training subset is seed-shuffled and stratified by `group x variety`, which is a stronger policy than simply taking the first `N` examples in file order.

### Overlap diagnostics

Each run includes bounded overlap diagnostics:

- `250` sampled nonmembers
- up to `250` members compared per sampled nonmember
- `62500` total pairs evaluated
- no flagged pairs above the configured threshold
- max overlap values were low:
  - 4-gram max overlap: `0.05814`
  - 7-gram max overlap: `0.010989`
  - 13-gram max overlap: `0.0`

This does not prove the dataset is perfectly clean, but it does reduce the risk that the current result is driven by obvious near-duplicate leakage.

## MIA Types Currently Present in the Benchmark

The historical Pythia experiment reported here used five baseline MIA families:

| Attack | Type | High-level idea |
|---|---|---|
| `loss` | likelihood-based | Members tend to have lower model loss than nonmembers |
| `reference` | calibrated/reference-based | Compares target-model loss to a reference-model loss on the same text |
| `zlib` | compression-normalized | Normalizes model loss by text compressibility as a rough complexity control |
| `min_k` | token-tail likelihood | Focuses on the hardest or lowest-confidence token region rather than whole-sequence average loss |
| `neighborhood` | perturbation-based | Compares target loss to losses on deterministic perturbation neighbors |

This matches the PRD goal of claiming general patterns across multiple MIA families instead of building the argument around a single attack.

## Current Results

### Aggregate AUC across the three 4-epoch seeds

| Attack | Seed 29 | Seed 41 | Seed 53 | Mean AUC |
|---|---:|---:|---:|---:|
| `loss` | 0.514727 | 0.513627 | 0.513489 | 0.513948 |
| `min_k` | 0.515619 | 0.514894 | 0.516205 | 0.515573 |
| `neighborhood` | 0.504673 | 0.509081 | 0.497913 | 0.503889 |
| `reference` | 0.507986 | 0.504099 | 0.498314 | 0.503466 |
| `zlib` | 0.496966 | 0.498172 | 0.495825 | 0.496988 |

Interpretation:

- `loss` and `min_k` are the strongest attacks in the current benchmark.
- Their signal is weak in absolute terms, but notably stable across seeds.
- `reference` and `neighborhood` are much closer to chance.
- `zlib` is effectively at chance in this setting.

### Subgroup differences

Mean AUC over the three seeds:

| Attack | Mean `G0` AUC | Mean `G1` AUC | Mean gap |
|---|---:|---:|---:|
| `loss` | 0.521232 | 0.507227 | 0.014004 |
| `min_k` | 0.525058 | 0.506680 | 0.018378 |
| `neighborhood` | 0.500682 | 0.507586 | 0.016582 |
| `reference` | 0.497137 | 0.509835 | 0.012697 |
| `zlib` | 0.500411 | 0.493244 | 0.007167 |

Interpretation:

- For the two strongest attacks, `G0` has consistently higher AUC than `G1`.
- The subgroup gaps are small, but they are directionally stable for `loss` and `min_k`.
- The weaker attacks do not show as clean or as interpretable a pattern.

### Balanced accuracy and majority baseline

For these balanced runs:

- `majority_class_accuracy = 0.5`
- reported `accuracy` and `balanced_accuracy` are nearly identical

This is a useful sanity check:

- the earlier `accuracy ~0.6` effect from an imbalanced split is gone
- the current result is not being driven by a majority-class shortcut

## What we can claim now

- The benchmark now produces a valid, balanced, multi-seed fine-tuning experiment on PAN 2017 English.
- There is weak but reproducible membership inference signal in the fine-tuning scenario.
- The strongest attacks in this setting are `loss` and `min_k`.
- There is preliminary evidence of subgroup disparity, with `G0` more exposed than `G1` for the strongest attacks.

## What we should not claim yet

- We should not claim strong leakage.
- We should not claim a universal fairness pattern across all attacks.
- We should not claim that the observed subgroup difference is final or statistically conclusive.
- We should not present RAG results yet, because true context-conditioned RAG scoring is still intentionally blocked in the codebase.

## Relation to the Existing Pretraining and RAG Paths

### Pretraining

The repository already supports a pretraining-style benchmark path, but the currently checked-in Pile sample uses synthetic `G0` and `G1` labels for pipeline mechanics. That means:

- it is useful for validating pretraining MIA mechanics
- it is not yet suitable for fairness claims about real demographic groups

This is still useful scientifically because the LLM paper suggests pretraining MIA may be near-random. A clean pretraining baseline would let us test whether weak aggregate signal remains weak after stronger decontamination and whether any subgroup effect survives under a more realistic demographic dataset.

Historical note:

- older checked-in output directories may contain `rag` rows from before the RAG safety block was enforced
- those rows should not be used for benchmark claims

### RAG

RAG is not yet benchmark-ready in the repository. The code intentionally raises an error because retrieved context is not yet injected into the model scoring path. This is the correct current behavior, because it prevents misleading benchmark-like numbers from a diagnostic-only path.

## Recommended Next Steps

### 1. Complete a real pretraining fairness baseline

Goal:
- build a pretraining-style experiment with real subgroup metadata rather than synthetic labels

Why:
- the Duan et al. paper suggests weak aggregate MIA is expected in large-scale LLM pretraining
- a fairness-aware replication would show whether weak overall leakage still hides subgroup effects

Concrete next steps:

- identify a text corpus with explicit binary subgroup metadata and a defensible member/nonmember split
- keep overlap diagnostics mandatory
- run the existing five attacks first on a small Pythia model, then scale upward only if signal remains stable
- treat pretraining as a baseline, not as the most likely source of strong leakage

### 2. Implement true RAG scoring before any RAG claims

Goal:
- support context-conditioned scoring where retrieved passages are actually inserted into the prompt seen by the model

Why:
- current RAG placeholder is intentionally blocked
- the PRD and TRD treat RAG as a distinct leakage channel, not just a copy of pretraining or fine-tuning scoring

Concrete next steps:

- build a retrieval index over a controlled corpus
- log retrieved passages per query
- score each sample with retrieval disabled and enabled
- attribute any leakage increase to retrieved context rather than only to parametric memory
- evaluate subgroup exposure and retrieval exposure together

### 3. Add statistical reliability to the current fine-tuning result

Goal:
- move from "stable-looking" to "statistically supported"

Concrete next steps:

- aggregate the three-seed results into summary plots and confidence intervals
- add bootstrap intervals for AUC and subgroup gaps
- optionally run additional seeds if runtime permits

### 4. Expand the model ladder

Goal:
- determine whether the weak signal persists, strengthens, or disappears with model scale

Concrete next steps:

- repeat the PAN fine-tuning workflow on `pythia-410m`
- if throughput is acceptable, test `pythia-1b`
- keep the same balanced windowed PAN setup so scale is the only major change

### 5. Move into defense benchmarking

Goal:
- align the benchmark more directly with the fairness papers on unequal protection

Concrete next steps:

- add at least one trainable defense condition
- reasonable first candidates are `dropout`, `weight decay`, or a lightweight differential privacy setting
- compare not only aggregate attack reduction, but also whether protection is distributed evenly across `G0` and `G1`

## Supervisor-Facing Takeaway

The project has moved past a toy pipeline stage. The current PAN 2017 fine-tuning benchmark is balanced, reproducible across seeds, aligned with the repo's PRD and TRD, and consistent with the literature's expectation that LLM membership inference is often weak unless the scenario makes memorization more visible.

The present evidence supports a careful claim:

> In a balanced PAN 2017 fine-tuning setting on `pythia-160m`, membership inference signal is weak but reproducible across seeds, with `loss` and `min_k` emerging as the most informative attacks and showing small but consistent subgroup differences.

That is enough to present as real progress. The next milestones should be:

1. finish a real pretraining fairness baseline
2. implement benchmark-valid RAG scoring
3. add statistical intervals and possibly more seeds
4. expand to larger Pythia models
5. start defense fairness experiments

## Pointers to the Exact Artifacts

- Seed 29 summary: [outputs/vm_finetune_lora_pythia_160m_seed29_ep4/summary.csv](../outputs/vm_finetune_lora_pythia_160m_seed29_ep4/summary.csv)
- Seed 41 summary: [outputs/vm_finetune_lora_pythia_160m_seed41_ep4/summary.csv](../outputs/vm_finetune_lora_pythia_160m_seed41_ep4/summary.csv)
- Seed 53 summary: [outputs/vm_finetune_lora_pythia_160m_seed53_ep4/summary.csv](../outputs/vm_finetune_lora_pythia_160m_seed53_ep4/summary.csv)
- Seed 29 manifest: [outputs/vm_finetune_lora_pythia_160m_seed29_ep4/run_manifest.json](../outputs/vm_finetune_lora_pythia_160m_seed29_ep4/run_manifest.json)
- Seed 41 manifest: [outputs/vm_finetune_lora_pythia_160m_seed41_ep4/run_manifest.json](../outputs/vm_finetune_lora_pythia_160m_seed41_ep4/run_manifest.json)
- Seed 53 manifest: [outputs/vm_finetune_lora_pythia_160m_seed53_ep4/run_manifest.json](../outputs/vm_finetune_lora_pythia_160m_seed53_ep4/run_manifest.json)
