# Neural Tokenizer

A research project exploring whether **neural networks can learn meaningful tokenization boundaries directly from data**, rather than relying on manually designed or rule-based tokenization schemes.

The project studies tokenization as a **learned boundary-detection problem** and investigates whether a neural tokenizer can discover stable, meaningful, and generalizable structural units in sequential data.

> **Status:** Active research / experimental
> **Current version:** Neural Tokenizer V2
> **Primary focus:** Learned token boundaries, boundary alignment, tolerance analysis, and statistical null-model evaluation

---

## Overview

Tokenization is one of the fundamental preprocessing steps in modern machine learning systems.

Traditional tokenizers generally rely on manually designed rules, frequency statistics, subword algorithms, or fixed heuristics. While these approaches work extremely well in many settings, they impose an external structure on the input before the learning system processes it.

This project asks a different question:

> **Can a neural network learn where tokens should begin and end directly from data?**

Instead of treating tokenization purely as a preprocessing algorithm, the Neural Tokenizer treats it as a **learnable representation problem**.

The central idea is to train a neural model to identify meaningful boundaries in a sequence and use those boundaries to construct learned tokens.

The project therefore has two interconnected goals:

1. **Build a working neural tokenizer.**
2. **Determine rigorously whether the boundaries it learns are meaningful rather than artifacts of the training process or chance.**

The second question is particularly important. A model producing boundaries is not, by itself, evidence that it has discovered useful structure. This motivates the boundary-alignment, tolerance, null-model, and statistical analyses developed in the later stages of the project.

---

# Research Question

The broad research question is:

> **Can tokenization itself be learned as a neural representation, and can the resulting token boundaries be shown to capture meaningful structure in the underlying data?**

This leads to several more specific questions:

* Can a neural model reliably learn token boundaries?
* Are the learned boundaries reproducible?
* Do learned boundaries align with reference boundaries?
* How sensitive is boundary alignment to small positional errors?
* How much alignment would occur by chance?
* Are learned tokens stable across datasets, random seeds, and model configurations?
* Do learned tokens improve downstream performance?
* Do the learned tokens correspond to meaningful structural units?
* Can the approach generalize beyond the data on which the tokenizer was trained?

---

# Conceptual Framework

The project views tokenization as a sequence segmentation problem.

Given an input sequence

```text
x₁ x₂ x₃ ... xₙ
```

the tokenizer attempts to learn a sequence of boundary decisions

```text
b₁ b₂ b₃ ... bₙ
```

where each boundary indicates a potential transition between learned tokens.

For example:

```text
Input:

x₁ x₂ x₃ | x₄ x₅ | x₆ x₇ x₈ | x₉

             ↓

Learned tokens:

[x₁ x₂ x₃] [x₄ x₅] [x₆ x₇ x₈] [x₉]
```

The model therefore does not need to be given the complete tokenization procedure explicitly. Instead, it learns a representation from which token boundaries can be inferred.

---

# Why Boundary Evaluation Matters

A central challenge in this project is distinguishing between:

**"The model produces boundaries."**

and

**"The model has learned meaningful boundaries."**

A neural model can produce apparently reasonable segmentation even when the boundaries contain little useful information.

For this reason, the project evaluates boundaries at several levels.

### 1. Exact alignment

Does the predicted boundary occur at exactly the same position as the reference boundary?

### 2. Tolerance-based alignment

If the prediction is shifted by a small number of positions, should it still be considered a successful match?

For example:

```text
Reference:  --------|--------
Prediction: ---------|-------
                     ↑
                  +1 offset
```

The current analysis evaluates multiple tolerances:

```text
Tolerance = 0
Tolerance = 1
Tolerance = 2
Tolerance = 3
Tolerance = 4
Tolerance = 5
```

### 3. Null-model comparison

Even random boundaries can occasionally align with reference boundaries.

Therefore, observed alignment should be compared against a suitable random/null baseline.

The project includes randomized boundary trials to estimate this chance-level alignment.

This provides a more meaningful question:

> Is the observed boundary alignment substantially better than what would be expected from random boundary placement?

---

# Project Evolution

The project is being developed incrementally.

```text
Neural Tokenizer
│
├── V1 — Initial neural tokenizer
│   ├── Baseline architecture
│   ├── Initial training pipeline
│   ├── Boundary prediction
│   └── Baseline evaluation
│
├── V2 — Boundary-aware evaluation
│   ├── Improved training/evaluation pipeline
│   ├── Boundary alignment
│   ├── Tolerance analysis
│   ├── Boundary-null analysis
│   └── Randomized null trials
│
└── Future
    ├── Statistical validation
    ├── Token quality analysis
    ├── Ablation studies
    ├── Generalization
    ├── Downstream evaluation
    ├── Scaling
    └── V3 architecture
```

---

# V1 — Initial Neural Tokenizer

V1 establishes the initial proof-of-concept implementation.

The objective of V1 is deliberately simple:

> Build a neural system capable of learning a tokenization/boundary prediction task end-to-end.

V1 establishes:

* The initial model architecture
* Dataset handling
* Training pipeline
* Loss computation
* Model optimization
* Checkpointing
* Boundary/token prediction
* Initial evaluation
* Experiment organization

V1 serves as the **baseline implementation** against which later architectural and methodological changes can be evaluated.

The purpose of V1 is not to provide a final tokenizer, but to establish that the learning problem can be formulated and trained successfully.

---

# V2 — Boundary-Aware Neural Tokenizer

V2 extends the baseline with a stronger emphasis on evaluating the quality of learned boundaries.

The main development areas are:

* Improved training/evaluation workflow
* Reproducible experiments
* Boundary alignment analysis
* Tolerance-based boundary evaluation
* Boundary-null experiments
* Randomized boundary trials
* More systematic experiment configuration

The V2 stage therefore shifts the project from:

> **"Can we train a neural tokenizer?"**

toward:

> **"Can we demonstrate that the tokenizer is learning meaningful boundary structure?"**

This distinction is important for the research direction of the project.

---

# Current V2 Evaluation

## Boundary Alignment

Predicted boundaries are compared against reference boundaries.

A prediction is considered aligned when it falls within a specified positional threshold of a reference boundary.

The current experiments use:

```text
Alignment threshold: 0.5
```

and evaluate multiple positional tolerances.

---

## Tolerance Analysis

Boundary prediction is inherently sensitive to small positional shifts.

Consider:

```text
Reference:

--------|--------

Prediction:

---------|------
         ↑
       +1
```

An exact-match metric would classify this as incorrect.

However, a tolerance-based metric can recognize that the prediction is very close to the true boundary.

The current experiments evaluate:

| Tolerance | Interpretation       |
| --------: | -------------------- |
|         0 | Exact boundary match |
|         1 | ±1 position          |
|         2 | ±2 positions         |
|         3 | ±3 positions         |
|         4 | ±4 positions         |
|         5 | ±5 positions         |

The resulting curve can provide information about how precisely the model localizes boundaries.

A useful tokenizer should ideally show:

* Strong exact alignment
* Gradual improvement under small tolerances
* Robust performance without requiring excessively large tolerances

If meaningful alignment only appears at very large tolerances, the interpretation becomes substantially weaker.

---

# Boundary Null Analysis

The null analysis is intended to answer an important statistical question:

> **How much boundary alignment would we observe if boundaries were generated without knowledge of the underlying structure?**

Randomized boundary trials provide an empirical estimate of chance-level alignment.

Conceptually:

```text
Reference boundaries
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Predicted boundaries  Random boundaries
        │               │
        ▼               ▼
   Alignment score   Null alignment score
        │               │
        └───────┬───────┘
                ▼
       Compare observed
       against null
```

This allows future experiments to move beyond raw accuracy and toward statistically grounded claims.

---

# Reproducibility

Experiments are configured to make results reproducible.

Current experiments record parameters such as:

```text
Device
Random seed
Batch size
Checkpoint
Alignment threshold
Tolerance range
Number of random/null trials
```

The current analysis uses:

```text
Seed: 42
Device: CUDA
Batch size: 64
Alignment threshold: 0.5

Tolerances:
0, 1, 2, 3, 4, 5
```

The exact configuration should always be recorded with experimental results.

---

# Current Status

The project currently has a functioning V1/V2 pipeline and a first generation of boundary-quality analysis.

### Completed

* [x] Initial neural tokenizer formulation
* [x] V1 baseline implementation
* [x] V1 training pipeline
* [x] V1 evaluation
* [x] V2 implementation
* [x] Model checkpointing
* [x] Reproducible experiment configuration
* [x] Boundary prediction
* [x] Boundary alignment evaluation
* [x] Exact-match evaluation
* [x] Tolerance analysis
* [x] Boundary-null analysis
* [x] Randomized null trials
* [x] Initial research documentation

### In progress

* [ ] More rigorous statistical validation
* [ ] Boundary precision/recall/F1
* [ ] Better null models
* [ ] Token stability analysis
* [ ] Ablation studies
* [ ] Cross-dataset evaluation
* [ ] Downstream task evaluation

---

# Important Current Limitation

The current results should be considered **exploratory**.

Successful boundary prediction or alignment does not automatically establish that the tokenizer has discovered semantically meaningful or generally useful tokens.

Several additional questions remain open:

1. Could the observed alignment occur through chance?
2. Are the boundaries stable across random seeds?
3. Are they stable across datasets?
4. Do the learned tokens correspond to meaningful structures?
5. Does learned tokenization improve a downstream model?
6. How does the neural tokenizer compare against established tokenization methods?
7. Does the tokenizer generalize to unseen distributions?

The null-model and tolerance analyses are therefore not the final evaluation. They are the foundation for a more rigorous evaluation framework.

---

# Roadmap

## Phase 1 — Baseline

**Goal:** Establish a reliable neural tokenizer baseline.

* V1 architecture
* V1 training
* V1 evaluation
* Reproducible experiments

**Status:** Complete

---

## Phase 2 — Boundary Analysis

**Goal:** Understand whether the model learns meaningful boundaries.

* Boundary alignment
* Exact-match analysis
* Tolerance analysis
* Random/null boundaries
* Chance-level comparison

**Status:** In progress / V2

---

## Phase 3 — Rigorous Statistical Validation

The next stage is to make the boundary analysis statistically rigorous.

### Planned work

* Boundary precision
* Boundary recall
* Boundary F1
* Distance-based metrics
* Per-sequence statistics
* Confidence intervals
* Multiple random seeds
* Larger null ensembles
* Empirical null distributions
* Statistical significance testing
* Effect-size reporting

The goal is to move from:

```text
Observed score = X
```

to something closer to:

```text
Observed score = X
Null expectation = Y
Effect size = Z
Confidence interval = ...
p-value / empirical significance = ...
```

where appropriate.

---

# Phase 4 — Token Quality

A boundary can be statistically significant without being useful.

Therefore, the next question is whether the resulting tokens contain meaningful information.

Potential analyses include:

* Token length distributions
* Token frequency distributions
* Token entropy
* Token stability
* Token reuse
* Token similarity
* Token information content
* Representation quality
* Correlation with known structures

A particularly important experiment will be to determine whether the same structures are recovered repeatedly under different:

* Random seeds
* Training subsets
* Model initializations
* Dataset sizes

---

# Phase 5 — Ablation Studies

A systematic ablation study will determine which components of the tokenizer are actually responsible for its behavior.

Potential ablations:

### Architecture

* Smaller model
* Larger model
* Alternative encoder
* Alternative boundary head

### Training objective

* Boundary loss
* Reconstruction loss
* Auxiliary objectives
* Combined objectives

### Data

* Dataset size
* Noise level
* Sequence length
* Distribution changes

### Optimization

* Learning rate
* Batch size
* Initialization
* Regularization

The objective is to identify which design choices genuinely matter.

---

# Phase 6 — Generalization

A learned tokenizer should ideally generalize beyond the exact data distribution used during training.

Future experiments will evaluate:

```text
Train distribution
       │
       ▼
Tokenizer
       │
       ├── Same-distribution test
       │
       ├── Unseen samples
       │
       ├── Distribution shift
       │
       └── Different datasets
```

Questions include:

* Do boundaries remain meaningful on unseen sequences?
* Does the tokenizer transfer between related datasets?
* How sensitive is it to noise?
* Does performance degrade gracefully under distribution shift?

---

# Phase 7 — Downstream Evaluation

Ultimately, the strongest evidence for useful tokenization may come from downstream performance.

Potential experiments include comparing:

```text
Raw sequence
     │
     ▼
Baseline model
```

against:

```text
Raw sequence
     │
     ▼
Neural Tokenizer
     │
     ▼
Token representation
     │
     ▼
Downstream model
```

Possible downstream metrics include:

* Classification accuracy
* Regression error
* Sequence modeling loss
* Sample efficiency
* Computational efficiency
* Context compression
* Representation quality

The key question becomes:

> **Does learned tokenization provide a measurable advantage over operating directly on the original sequence?**

---

# Phase 8 — Comparison With Existing Methods

The Neural Tokenizer should eventually be compared against appropriate baselines.

Depending on the target domain, these may include:

* Fixed-length segmentation
* Rule-based segmentation
* Frequency-based tokenization
* Standard subword tokenizers
* Existing learned tokenizers
* Random segmentation
* Oracle/reference segmentation

Comparison should use identical datasets and downstream tasks wherever possible.

---

# Phase 9 — Scaling

Once the methodology is validated on manageable datasets, the project can be scaled.

Potential directions:

* Larger datasets
* Longer sequences
* Larger neural architectures
* GPU-optimized training
* Automated hyperparameter sweeps
* Distributed experiments
* Experiment tracking
* Efficient inference

Scaling should come **after** the evaluation methodology is sufficiently mature. Larger models alone do not solve an unclear evaluation problem.

---

# Future V3

V3 is intentionally not fully specified yet.

The architecture should be driven by what the V2 experiments reveal.

Potential V3 directions include:

### Better boundary representation

Explore alternative ways of representing token boundaries and uncertainty.

### Differentiable token formation

Investigate methods that allow token formation itself to participate directly in end-to-end optimization.

### Joint boundary and representation learning

Rather than learning boundaries independently, jointly optimize:

```text
Input
  │
  ▼
Encoder
  │
  ├───────────────┐
  ▼               ▼
Boundary head   Representation head
  │               │
  └───────┬───────┘
          ▼
    Learned tokens
```

### Hierarchical tokenization

Investigate whether meaningful structures exist at multiple scales:

```text
fine units
    ↓
tokens
    ↓
higher-level tokens
    ↓
larger structures
```

### Adaptive token lengths

Allow the model to determine token lengths dynamically rather than imposing a fixed segmentation scale.

---

# Research Philosophy

The project follows an important principle:

> **A neural model producing an interesting result is not sufficient evidence that the learned structure is meaningful.**

Every major architectural improvement should therefore eventually be accompanied by:

1. Controlled experiments
2. Baseline comparisons
3. Ablation studies
4. Statistical analysis
5. Reproducibility checks
6. Generalization tests
7. Downstream evaluation

This is intended to keep the project focused on **scientific validation**, rather than simply increasing model complexity.

---

# Repository Structure

The repository is organized around experiments, models, evaluation, and analysis.

A representative structure is:

```text
neural-tokenizer/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── src/
│   └── ...
│
├── experiments/
│   ├── neural_v1/
│   │   └── ...
│   │
│   └── neural_v2/
│       ├── ...
│       └── best.pt
│
├── tests/
│   └── ...
│
├── configs/
│   └── ...
│
├── scripts/
│   └── ...
│
└── notebooks/
    └── ...
```

The exact structure may evolve as the project grows.

Large datasets, checkpoints, generated artifacts, and temporary experiment files should generally not be committed to Git unless there is a specific reason to version them.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd neural-tokenizer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project is packaged through `pyproject.toml`, installation can also be performed with:

```bash
pip install -e .
```

---

# Running Experiments

The exact commands may evolve with the project structure.

Typical workflows follow:

```text
Prepare data
    ↓
Configure experiment
    ↓
Train tokenizer
    ↓
Save checkpoint
    ↓
Evaluate boundaries
    ↓
Run tolerance analysis
    ↓
Run null-model analysis
    ↓
Analyze results
```

Example:

```bash
python <training-script>.py
```

Evaluation:

```bash
python <evaluation-script>.py
```

Boundary/null analysis:

```bash
python <analysis-script>.py
```

Refer to the relevant experiment configuration and source files for the current command-line interface.

---

# Reproducibility

When reporting an experiment, record at minimum:

```text
Model version
Dataset/version
Random seed
Device
Batch size
Learning rate
Number of epochs
Checkpoint
Evaluation threshold
Tolerance range
Number of null trials
```

A result should ideally be reproducible from the repository and its associated configuration.

---

# Experiments and Results

Experimental results are intentionally kept separate from the core implementation.

Each experiment should answer a specific question rather than simply generating another model checkpoint.

For example:

```text
Experiment 1
    Does the model learn boundaries?

Experiment 2
    How accurately are boundaries localized?

Experiment 3
    How does alignment change under tolerance?

Experiment 4
    Is the alignment above chance?

Experiment 5
    Are boundaries stable across random seeds?

Experiment 6
    Do learned tokens improve a downstream task?
```

This structure is intended to keep the project hypothesis-driven.

---

# What Would Constitute Strong Evidence?

A successful future version of the project should ideally demonstrate several independent properties.

### 1. Boundary accuracy

Predicted boundaries align strongly with reference boundaries.

### 2. Statistical significance

Observed alignment is substantially above an appropriate null distribution.

### 3. Stability

The tokenizer learns similar structures across random seeds and datasets.

### 4. Generalization

The learned structure persists on unseen data.

### 5. Interpretability

The resulting tokens correspond to identifiable structural properties.

### 6. Downstream utility

Using the learned tokens improves or meaningfully changes downstream performance.

A combination of these would provide much stronger evidence than any individual metric.

---

# Open Research Questions

Several questions remain intentionally open.

### Representation

What information should a learned token preserve?

### Segmentation

Is there a unique optimal segmentation, or are multiple tokenizations equally valid?

### Scale

Do meaningful structures exist at multiple tokenization scales?

### Stability

Should a useful tokenizer produce identical boundaries across random seeds?

### Supervision

How much supervision is actually necessary?

### Generalization

Can tokenization learned on one distribution transfer to another?

### Utility

Does discovering "natural" boundaries actually improve machine-learning performance?

### Compression

Can neural tokenization provide useful sequence compression without losing important information?

---

# Long-Term Vision

The long-term goal is not simply to build another tokenizer.

The broader objective is to investigate **tokenization as a learned scientific object**.

Rather than assuming that a particular segmentation is correct, the project aims to develop methods for asking:

```text
What structures does the data contain?
        ↓
Can a neural model discover them?
        ↓
Are the discovered structures statistically real?
        ↓
Are they stable?
        ↓
Do they generalize?
        ↓
Are they useful?
```

This turns tokenization from a preprocessing choice into an experimentally testable representation-learning problem.

---

# Development Principles

The project follows several principles:

### Reproducibility over convenience

Experiments should be reproducible rather than dependent on undocumented settings.

### Evaluation before scaling

Improve the evaluation methodology before simply increasing model size.

### Baselines matter

Every claimed improvement should have an appropriate baseline.

### Null models matter

A pattern is more convincing when it is shown to exceed an appropriate chance-level expectation.

### Architecture should follow evidence

Future architectures should be motivated by observed failure modes rather than added complexity for its own sake.

### Negative results are useful

Failure to find meaningful boundaries is itself a valuable scientific result if the experiment is properly controlled.

---

# Roadmap Summary

```text
                    Neural Tokenizer
                           │
                           ▼
                  ┌─────────────────┐
                  │       V1        │
                  │   Baseline      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │       V2        │
                  │ Boundary-aware  │
                  │    analysis     │
                  └────────┬────────┘
                           │
                           ▼
             ┌──────────────────────────┐
             │ Statistical validation   │
             │ Null models + metrics    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │     Token quality        │
             │ Stability + structure   │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │     Ablation studies     │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │      Generalization      │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │   Downstream evaluation  │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │       V3 architecture    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Scaling + broader study │
             └──────────────────────────┘
```

---

# Current Milestone

The current repository represents the transition from a **proof-of-concept neural tokenizer** toward a more rigorous research framework.

V1 established the basic learning pipeline.

V2 extends this with boundary-aware evaluation, including:

* Boundary alignment
* Positional tolerance analysis
* Boundary-null experiments
* Randomized trials
* Reproducible evaluation settings

The immediate priority is to determine whether the observed boundary structure is robust, statistically meaningful, and useful before committing to a major V3 architectural redesign.

---

# License

License information will be added as the project matures.

---

# Citation

If this project develops into a publication, the citation information will be added here.

```bibtex
@software{neural_tokenizer,
  title  = {Neural Tokenizer},
  author = {Nachiket Tanksale},
  year   = {2026},
  url    = {https://github.com/nachiket273/neural-tokenizer-lab}
}
```

---

# Author

**Nachiket Tanksale**

Research interests spanning:

* Machine Learning
* Neural Representation Learning
* Quantum Technologies
* Scientific Machine Learning
* Computational Physics

---

## Project Status

**Active research project.**

The implementation, experiments, evaluation methodology, and research questions are expected to evolve as new results are obtained.
