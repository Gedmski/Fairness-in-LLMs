# Product Requirements Document (PRD)
## Project: Fairness-Aware Membership Inference Evaluation for Large Language Models

### 1. Executive Summary
This project will build a benchmarking framework to study how membership inference attack (MIA) risk is distributed across demographic subgroups in text-based large language model (LLM) settings. The framework will compare three operational scenarios:

1. Pre-training style evaluation on models trained on large corpora such as The Pile
2. Fine-tuning or instruction-tuning on smaller demographic or domain datasets
3. Retrieval-augmented generation (RAG), where leakage may come from retrieved context instead of only parametric memory

The goal is not to claim a strong result for one specific attack such as mask-based MIA. The goal is to identify general fairness patterns across multiple MIA families, datasets, and LLM deployment scenarios, then determine which scenario is the most informative for fairness assessment.

### 2. Research Motivation
The paper "On the Privacy Risks of Algorithmic Fairness" shows that fairness interventions can change privacy leakage unevenly across groups, often increasing leakage for unprivileged or minority subgroups. The paper "Understanding Disparate Effects of Membership Inference Attacks and Their Countermeasures" formalizes privacy-leakage disparity (PLD) and shows that subgroup size and subgroup distribution can affect both privacy risk and the effectiveness of defenses.

At the same time, "Do Membership Inference Attacks Work on Large Language Models?" shows that MIAs on pre-trained LLMs are often near random on large-scale corpora such as The Pile because of few training iterations and fuzzy member/non-member boundaries. That result does not remove the fairness question. It changes the research focus: the project should compare settings where memorization and subgroup disparity may become more visible, especially fine-tuning and RAG.

### 3. Core Research Questions
1. Do MIAs affect binary demographic subgroups differently in LLM-related text settings?
2. Which scenario exposes fairness disparities most reliably: pre-training, fine-tuning, or RAG?
3. Which MIA families lead to stable, scenario-level conclusions, and which findings are attack-specific?
4. Which model scales and deployment settings are most affected by privacy unfairness?
5. Do defenses reduce overall leakage while redistributing protection unevenly across groups?

### 4. Scope
#### In Scope
- Text-based LLMs and language modeling workflows only
- Multiple MIA families, including but not limited to mask-based attacks
- Binary sensitive attributes for the primary benchmark
- Disaggregated reporting for each subgroup, not only a single aggregate gap value
- Comparison across pre-training, fine-tuning, and RAG settings
- Fairness analysis of both attacks and defenses

#### Out of Scope
- Claims that one attack is universally best
- Unsupported sensitive labels inferred from text without dataset metadata
- Purely classifier-only conclusions that do not transfer to LLM settings
- Strong claims about RAG or fine-tuning before empirical comparison is complete

### 5. Target Scenarios
#### Scenario A: Pre-training Baseline
- Evaluate open LLMs trained on large corpora such as The Pile
- Use this as the baseline because prior work suggests overall MIA performance may be weak in this setting
- Measure whether subgroup disparities still appear even when absolute attack performance is low

#### Scenario B: Fine-Tuning / Instruction-Tuning
- Evaluate continued pre-training, supervised fine-tuning, or instruction-tuning on smaller text corpora
- Hypothesis: repeated exposure to smaller datasets may increase memorization and make subgroup disparities more visible
- This is a primary candidate scenario for fairness assessment

#### Scenario C: Retrieval-Augmented Generation (RAG)
- Evaluate systems where sensitive or member records can be retrieved into the prompt context
- Treat RAG as a separate leakage channel that may differ from parametric memorization
- Compare leakage with and without retrieval, and inspect whether subgroup disparities are amplified by retrieval behavior

### 6. Datasets and Subgroup Policy
#### Primary Data Sources
1. The Pile
   - Used for pre-training-oriented LLM evaluation and member/non-member construction
   - Important for reproducing the LLM setting studied in the paper set
2. PAN 2018
   - Used as the primary demographics-based text dataset for downstream subgroup analysis
   - Default protected attribute should come from explicit binary metadata available in the chosen PAN 2018 subset, such as gender if that subset is used

#### Additional Dataset Rule
- Additional text datasets may be added if they contain explicit sensitive or demographic metadata relevant to fairness auditing
- Any added dataset must support clear member/non-member construction and binary subgroup definition for the main benchmark

#### Binary Attribute Rule
- The main benchmark keeps binary attributes at all times
- Every result must be reported separately for the two groups, for example `male` and `female`, in addition to the disparity gap
- If a dataset contains multi-class attributes, those attributes must either be converted into a justified binary split for the primary benchmark or moved to secondary analysis

#### Data Integrity Rule
- Do not infer race or other sensitive labels from free text when the dataset does not provide them
- Member and non-member sets should be matched by domain and time period when possible
- Overlap analysis is required to control fuzzy boundaries between members and non-members

### 7. Attack Coverage
#### Baseline Attack Families
The first implementation round must cover at least the following attacks:
- LOSS attack
- Reference-based attack
- Zlib entropy attack
- Min-k% probability attack
- Neighborhood or mask-based attack

#### Extension Requirement
The framework must remain extensible to additional MIA families, such as:
- Shadow-model or classifier-based attacks when applicable
- White-box or gradient-based attacks when model access allows
- Embedding-based or semantic similarity variants

#### Reporting Rule
- The project should claim general results across attack families whenever possible
- If an observed pattern appears only in one attack, it must be labeled as attack-specific

### 8. Metrics and Evaluation Policy
#### Privacy Metrics
- AUC-ROC
- TPR@1%FPR
- TPR@0.1%FPR
- Attack accuracy when relevant

#### Fairness Metrics
- Per-group attack performance reported separately for `G0` and `G1`
- Privacy-leakage disparity (PLD), defined as an absolute subgroup gap on the selected attack metric
- Utility gaps across groups, such as perplexity, loss, or downstream task accuracy depending on the scenario

#### Methodology Rules
- Use subgroup-disaggregated evaluation, not only a single global threshold
- Track both aggregate risk and group-specific risk
- Use confidence intervals or bootstrap estimates for main comparisons
- Audit whether observed attack success could be explained by distribution shift or overlap artifacts

### 9. Defense Analysis
The framework should evaluate whether defenses reduce leakage fairly across groups. Initial defenses may include:
- Differential privacy for trainable scenarios
- L2 regularization or weight decay
- Dropout
- Scenario-specific retrieval controls for RAG, such as deduplication or retrieval filtering

The project should report both:
- Overall privacy reduction
- Whether the protection is distributed evenly or unevenly across groups

### 10. Success Criteria
The project is successful if it:
- Identifies the scenario that best exposes fairness-related MIA behavior
- Produces subgroup-disaggregated results for binary attributes across multiple attacks
- Distinguishes attack-specific findings from robust, cross-attack findings
- Determines whether pre-training, fine-tuning, or RAG leads to the clearest fairness signal
- Produces conclusions that are specific to LLMs, not only inherited from tabular or classifier settings

### 11. Deliverables
- A modular benchmarking codebase for datasets, attacks, scenarios, metrics, and defenses
- A reproducible experiment matrix spanning pre-training, fine-tuning, and RAG
- Tables and figures that report subgroup-specific results and PLD side by side
- A research summary that states which models and scenarios are most affected by privacy unfairness
