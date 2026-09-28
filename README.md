# ANLY 735 Replication Laboratory #2

## Can the Model Keep Learning?

This repository contains my work for **ANLY 735 – Research Seminar in Predictive AI, Replication Laboratory #2**.

The laboratory is a **proxy replication** of a broader phenomenon discussed in:

> Klein, T., Luther, C., McAuliffe, M., Miklautz, L., Plant, C., & Tschiatschek, S. (2026). *Plasticity Loss in Deep Reinforcement Learning: A Survey*.

The survey describes plasticity as a model's ability to continue adapting when learning conditions change. Because the anchor paper is a survey rather than a primary empirical study with one original dataset and codebase, this laboratory uses a controlled from-scratch experiment rather than attempting to reproduce an exact numerical result.

## Research Question

**What happens to learning behavior when conditions change?**

The experiment examines whether a previously trained neural network can continue learning after an input-distribution shift and compares its post-change learning behavior with a freshly initialized model.

The analysis follows the Stability–Plasticity Diagnostic:

**CHANGE → RETENTION → NEW LEARNING → RATE → TRADEOFF**

An immediate drop in performance after the condition changes is not treated as evidence of plasticity loss by itself. The main evidence comes from what happens during subsequent learning.

---

## Experiment Overview

The experiment uses the scikit-learn **Digits dataset**.

Each observation is an 8 × 8 grayscale handwritten digit image represented by 64 pixel features. The prediction target is the digit class from 0 through 9.

Two learning conditions are created:

### Condition A

The neural network is trained on the original digit images.

### Condition B

A fixed permutation is applied to all 64 pixel positions. The labels remain unchanged, but the input distribution is altered.

Two models are then compared:

- **Warm model** — trained first on Condition A and then continues learning under Condition B.
- **Fresh model** — newly initialized when Condition B begins and trained only on Condition B.

The warm and fresh models use the same architecture, hyperparameters, Condition B training data, and example ordering.

---

## Repository Structure

anly735-lab02-RikinBasu/
│
├── README.md
├── replication-lab.qmd
├── references.bib
│
├── python/
│   └── lab02_analysis.py
│
├── analysis/
│   ├── lab02_phase_a_trajectory.csv
│   ├── lab02_phase_b_trajectory.csv
│   ├── lab02_summary_by_seed.csv
│   ├── lab02_summary_overall.csv
│   ├── lab02_pixel_permutation.csv
│   ├── lab02_config.json
│   └── lab02_environment.txt
│
└── figures/
    ├── lab02_condition_b_learning.png
    └── lab02_retention_vs_adaptation.png