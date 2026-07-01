# MTech Research Project — Learning to Rank with Context Effects

**Student:** Ajaz Hussain (DA25M006)
**Institution:** IIT Madras
**Supervisor:** Prof. Suryanarayana Sankagiri

---

## Overview

This repository contains the complete research work for my MTech project on **Learning to Rank (LTR) with Context Effects**.

The project is structured in two phases:

**Phase 1 — Classical LTR Foundations**
Implementation and analysis of three foundational LTR algorithms on the LETOR 4.0 benchmark dataset, with a focus on understanding *why* each algorithm improves on the previous one.

**Phase 2 — Context-Aware Reranking (PASAR)**
Design and implementation of **PASAR (Position-Aware Self-Attention Reranker)** — a novel reranker that merges Pobrotyn et al.'s self-attention reranker with RewardRank-style learned position bias embeddings on the Expedia Hotel Search dataset.

---

## Phase 1 — Classical LTR

**Algorithms implemented:**
- RankNet (Burges et al., ICML 2005)
- LambdaRank (Burges et al., NeurIPS 2006)
- LambdaMART (Burges, MSR 2010)

**Dataset:** LETOR 4.0 (MQ2007) — 1,692 queries, ~69,000 query-document pairs, 46 features

---

## Phase 2 — PASAR: Position-Aware Self-Attention Reranker

### Research Hypothesis

> A reranking model that jointly captures **inter-item context effects** (via self-attention over the candidate slate) and **display position bias** (via learned rank-slot embeddings) will outperform models that capture only one of these two effects in isolation.

This hypothesis is motivated by the observation that classical LTR models like LambdaMART score each document independently — making them structurally blind to two types of IIA violations present in real e-commerce data:
1. **Inter-item context effects** — the presence of other items in the slate influences perceived relevance
2. **Position bias** — users attend to items differently depending on their display position

### Dataset

**Expedia Hotel Search (ICDM 2013)** — available at [Kaggle](https://www.kaggle.com/c/expedia-personalized-sort/data)

| Property | Value |
|----------|-------|
| Sessions used | 80,000 (stratified) |
| Booking sessions | 15,845 (all retained) |
| Features per hotel | 23 |
| Max position slots | 38 |
| Relevance labels | `click_bool + 4 × booking_bool` (values: 0 / 1 / 5) |

**Key preprocessing decision:** Only `random_bool=1` sessions used for training to avoid confounding display position with hotel quality — ensuring position bias is learnable from genuine random exposure.

### Model Architecture

```
Input: [batch, slate_size, 23 features]
         │
         ▼
Feature Encoder
Linear → LayerNorm → ReLU → 128-dim
         │
         ▼
Transformer Encoder (2 layers, 4 heads)
Self-attention over full candidate slate
(captures inter-item context effects)
         │
    ┌────┴────┐
    │         │
    │  Learned Position Embedding Table
    │  (38 slots × 128-dim)
    │  (captures display position bias)
    │         │
    └────┬────┘
         │ Additive Fusion
         ▼
Linear Scorer → scalar relevance score per item
```

**Total parameters:** 413,057

### Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam (lr = 3e-3) |
| Loss | Weighted pairwise (relevance-gap + position-discount) |
| Slate sampling | Booked = 3.0×, Clicked = 1.5×, Neutral = 0.5× |
| Epochs | 20 |
| Compute | Kaggle Tesla T4 GPU |

### Ablation Results

| Model | NDCG@10 |
|-------|---------|
| SelfAttentionOnly (Pobrotyn baseline) | 0.2993 |
| **PASAR (full model)** | **0.3848** |
| PositionBiasOnly | 0.3922 |

**Key finding:** PositionBiasOnly outperforms full PASAR, suggesting that inter-item self-attention can interfere with position signal utilization rather than complementing it — a finding that itself warrants further investigation and is a publishable observation.

---

## Repository Structure

```
├── src/
│   ├── data/          # Data loading and fold splitting
│   ├── models/        # RankNet, LambdaRank, LambdaMART, PASAR implementations
│   ├── metrics/       # NDCG@k, MAP evaluation
│   ├── training/      # Training loops, early stopping, logging
│   └── utils/         # Reproducibility, device setup, result saving
│
├── notebooks/         # Kaggle experiment notebooks
├── plots/             # Generated figures and visualisations
├── results/           # Saved metrics and experiment outputs
├── datasets/          # LETOR 4.0 data (not committed — see below)
├── papers/
│   ├── originals/     # PDFs of papers read
│   └── summaries/     # Paper summaries (Markdown)
└── report/            # Exported research report (Overleaf sync)
```

---

## Setup

```bash
git clone https://github.com/ajazhsn/Ajaz_Hussain_MTP-.git
cd Ajaz_Hussain_MTP-
pip install -r requirements.txt
```

**LETOR 4.0:** Download MQ2007 from the official source and place in `datasets/LETOR4.0/`.

**Expedia:** Download from [Kaggle](https://www.kaggle.com/c/expedia-personalized-sort/data) and place in `datasets/Expedia/`.

---

## Phase 1 Results

| Model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|-------|--------|--------|--------|---------|
| RankNet | — | — | — | — |
| LambdaRank | — | — | — | — |
| LambdaMART | — | — | — | — |

---

## Papers Read

| # | Paper | Summary |
|---|-------|---------|
| 1 | Burges (2010) — RankNet to LambdaRank to LambdaMART | [Summary](papers/summaries/ranknet_lambdarank_lambdamart.md) |
| 2 | Seshadri et al., ICML 2019 — CDM: Context Effects from Raw Choice Data | [Summary](papers/summaries/cdm_context_effects.md) |
| 3 | Seshadri et al., NeurIPS 2020 — Learning Rich Rankings | [Summary](papers/summaries/learning_rich_rankings.md) |
| 4 | Pei et al., RecSys 2019 — RewardRank | [Summary](papers/summaries/rewardrank.md) |
| 5 | Pobrotyn et al., 2020 — Context-Aware LTR with Self-Attention | [Summary](papers/summaries/self_attention_reranker.md) |
| 6 | Tomlinson & Benson, 2020 — Learning Interpretable Feature Context Effects | [Summary](papers/summaries/lcl_dlcl.md) |

---

## Progress Log

| Date | Milestone |
|------|-----------|
| 2025-06 | Repository setup, folder structure created |
| 2025-06 | LETOR 4.0 data loader implementation |
| 2025-06 | RankNet implementation and training |
| 2025-06 | LambdaRank implementation and training |
| 2025-06 | LambdaMART implementation and training |
| 2025-07 | Expedia dataset preprocessing and EDA |
| 2025-07 | PASAR architecture design and implementation |
| 2025-07 | PASAR training with weighted slate sampling |
| 2025-07 | Ablation study: SelfAttentionOnly vs PASAR vs PositionBiasOnly |
| | Comparative analysis and plots |
| | Report writing |

---

## Broader Research Direction

This project sits at the intersection of **Learning to Rank**, **discrete choice modeling**, and **context-aware recommendation**. The professor's broader vision is towards **Conversational Recommender Systems for E-Commerce** — an interactive system where users refine preferences through dialogue and receive context-aware ranked recommendations.

The PASAR work is a foundational step: establishing that context effects (both inter-item and positional) matter in e-commerce ranking, and that they can be explicitly modeled rather than ignored.
