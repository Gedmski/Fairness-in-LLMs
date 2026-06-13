# Technical Requirements Document (TRD)
## Project: Benchmarking Infrastructure for Fairness-Aware MIA Auditing in LLMs

### 1. System Overview
The system will be a modular Python benchmarking package for auditing membership inference fairness in LLM workflows. The architecture must support:

- Dataset adapters
- Model adapters
- Scenario runners
- Attack implementations
- Fairness and utility metrics
- Defense modules
- Reproducible reporting

The benchmark should support three scenario classes from the start:
- Pre-training style evaluation
- Fine-tuning or instruction-tuning evaluation
- Retrieval-augmented generation evaluation

### 2. Runtime and Environment
#### 2.1 Core Stack
- Python as the primary language
- PyTorch plus Hugging Face `transformers` and `datasets`
- Standard experiment tooling for metrics, logging, and configuration management

#### 2.2 Hardware Assumptions
- The project should run on the available 2 x NVIDIA L4 GPUs with 24 GB VRAM each
- `bfloat16` is preferred when supported
- The code must allow fallback to `float16`, quantization, or parameter-efficient fine-tuning for larger models

#### 2.3 Model Family Assumptions
- Initial target models should be open causal LMs suitable for text MIA experiments
- The Pythia family is a practical default because it aligns with the paper set and spans multiple model sizes
- A scale ladder should be used instead of forcing all experiments onto the largest model

Recommended first-pass target sizes:
- Small: 160M to 1.4B
- Medium: 2.8B to 6.9B
- Large or optional: 12B, only if runtime and memory allow

### 3. Data Pipeline Requirements
#### 3.1 Supported Datasets
- The Pile for pre-training-oriented member/non-member evaluation
- PAN 2018 for demographic text experiments
- Optional additional text datasets with explicit binary sensitive attributes

#### 3.2 Sample Construction
- Samples must be built from fixed token windows instead of variable free-form lengths
- Window size must be configurable so experiments can test sensitivity to sequence length
- Train, validation, and test splits must be reproducible from configuration

#### 3.3 Binary Attribute Handling
- The primary benchmark uses binary protected attributes only
- Sensitive attributes must come from dataset metadata, not ad hoc inference from text
- Results must always include separate values for `G0` and `G1`, not only a single disparity score

#### 3.4 Overlap and Decontamination Controls
The data pipeline must explicitly handle fuzzy membership boundaries highlighted in the LLM paper.

Required controls:
- Match member and non-member sets by domain where possible
- Match time period when possible to reduce temporal distribution shift
- Compute overlap statistics such as 4-gram, 7-gram, and 13-gram overlap
- Flag or remove near-duplicate pairs according to configurable thresholds
- Track a separate "fuzzy member" bucket for highly similar records when useful

### 4. Scenario Runner Requirements
#### 4.1 Pre-training Baseline Runner
- Evaluate existing checkpoints trained on large corpora such as The Pile
- Measure baseline MIA performance and subgroup disparity without additional task adaptation
- Treat weak aggregate MIA performance as an expected outcome, not a failure

#### 4.2 Fine-Tuning Runner
- Support continued pre-training, supervised fine-tuning, or instruction tuning
- Allow experiments on smaller text datasets where repeated exposure may increase memorization
- Record training configuration, number of epochs, and subgroup sampling policy

#### 4.3 RAG Runner
- Build a retrieval index from a text corpus that may contain sensitive or member records
- Evaluate leakage with retrieval enabled and disabled
- Log which passages were retrieved so leakage can be attributed to retrieval behavior rather than only model parameters

### 5. Attack Framework Requirements
Each attack must implement a common interface:
- Input: text sample, target model, and scenario metadata
- Output: continuous membership score and optional auxiliary diagnostics

#### 5.1 Required Baseline Attacks
- LOSS attack
- Reference-based attack
- Zlib entropy attack
- Min-k% probability attack
- Neighborhood or mask-based attack

#### 5.2 Reference-Based Attack Requirements
- The reference model must be configurable
- The code should prefer a reference model with compatible tokenization or clearly normalized scoring
- The system must not hardcode one reference checkpoint as a universal requirement

#### 5.3 Neighborhood or Mask-Based Attack Requirements
- The masking or perturbation model must be configurable
- Perturbation ratio, number of neighbors, and sampling policy must be configurable
- The default implementation should be treated as one baseline attack, not the main project claim

#### 5.4 Extension Support
The attack registry must allow adding:
- Shadow-model attacks
- White-box attacks
- Embedding-based variants
- Scenario-specific attacks for RAG or fine-tuned models

### 6. Thresholding and Fairness Evaluation
The fairness papers indicate that a single global attack threshold can hide subgroup-specific leakage patterns.

Therefore the system must support:
- Global threshold evaluation
- Group-conditional threshold evaluation
- Per-group score distribution analysis

Primary reporting format:
- `metric(G0)`
- `metric(G1)`
- `gap = |metric(G0) - metric(G1)|`

### 7. Metrics and Reporting
#### 7.1 Privacy Metrics
- AUC-ROC
- TPR@1%FPR
- TPR@0.1%FPR
- Attack accuracy when a thresholded decision is used

#### 7.2 Fairness Metrics
- Privacy-leakage disparity on each primary privacy metric
- Per-group attack calibration and score distribution summaries
- Utility disparity across groups

#### 7.3 Utility Metrics by Scenario
- Pre-training: loss, perplexity, and token-level likelihood summaries
- Fine-tuning: task accuracy, F1, or other task-specific utility metrics
- RAG: answer quality, retrieval attribution, and retrieval exposure statistics

#### 7.4 Statistical Reliability
- Main tables should include confidence intervals or bootstrap estimates
- Repeated runs should be supported for stochastic attacks or training settings

### 8. Defense Modules
The benchmark must support defense studies because defense fairness is part of the research question.

Initial defense modules:
- Differential privacy for trainable scenarios
- L2 regularization or weight decay
- Dropout
- RAG-specific controls such as retrieval deduplication, access filtering, or chunk suppression

For each defense setting, the system must report:
- Aggregate attack reduction
- Per-group attack reduction
- Utility loss
- Change in subgroup disparity

### 9. Experiment Management
The experiment layer must support:
- Configuration-driven runs
- Scenario, dataset, model, and attack sweeps
- Reproducible random seeds
- Structured outputs for tables and plots

Suggested top-level experiment axes:
- Scenario: pre-training, fine-tuning, RAG
- Dataset: The Pile, PAN 2018, optional demographic text datasets
- Model scale: small, medium, large
- Attack family
- Defense setting
- Protected attribute pair

### 10. Minimum Viable Benchmark
The first complete benchmark release should include:

1. One pre-training baseline on open LMs trained on The Pile
2. One fine-tuning setting on PAN 2018 or another demographic text dataset
3. One RAG setting built from a controlled text corpus
4. The five baseline attacks
5. Per-group and gap-based reporting for one binary protected attribute
6. At least one defense comparison in a trainable scenario

### 11. Output Artifacts
Required outputs for each experiment:
- Machine-readable result files
- Per-group metric tables
- Gap summary tables
- Overlap and decontamination diagnostics
- Scenario comparison plots
- A final report that separates attack-specific observations from general conclusions
