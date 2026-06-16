# MTech Research Project — Learning to Rank

**Student:** Ajaz Hussain (DA25M006)
**Institution:** [IIT Madras]
**Supervisor:** [Professor Suryanarayna Sankagiri]

---

## Overview

This repository contains the complete research work for my MTech project on **Learning to Rank (LTR)**.
The project implements and analyses three foundational LTR algorithms on the LETOR 4.0 benchmark dataset,
with a focus on understanding *why* each algorithm improves on the previous one.

**Algorithms:**
- RankNet (Burges et al., ICML 2005)
- LambdaRank (Burges et al., NeurIPS 2006)
- LambdaMART (Burges, MSR 2010)

**Dataset:** LETOR 4.0 (MQ2007) — 1,692 queries, ~69,000 query-document pairs, 46 features

---

## Repository Structure

```
├── src/
│   ├── data/          # Data loading and fold splitting
│   ├── models/        # RankNet, LambdaRank, LambdaMART implementations
│   ├── metrics/       # NDCG@k, MAP evaluation
│   ├── training/      # Training loops, early stopping, logging
│   └── utils/         # Reproducibility, device setup, result saving
│
├── notebooks/         # Colab experiment notebooks
├── plots/             # Generated figures and visualisations
├── results/           # Saved metrics and experiment outputs
├── datasets/          # LETOR 4.0 data (not committed — see below)
├── papers/
│   ├── originals/     # PDFs of papers read
│   └── summaries/     # My understanding of each paper (Markdown)
└── report/            # Exported research report (Overleaf sync)
```

---

## Setup

```bash
git clone https://github.com/ajazhsn/Ajaz_Hussain_MTP-.git
cd Ajaz_Hussain_MTP-
pip install -r requirements.txt
```

**Dataset:** Download LETOR 4.0 (MQ2007) from the official source and place it in `datasets/LETOR4.0/`.

---

## Results

*(To be updated as experiments are completed)*

| Model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|-------|--------|--------|--------|---------|
| RankNet | — | — | — | — |
| LambdaRank | — | — | — | — |
| LambdaMART | — | — | — | — |

---

## Papers Read

| Paper | Summary |
|-------|---------|
| Burges et al. (2010) — RankNet to LambdaMART | [Summary](papers/summaries/ranknet_lambdarank_lambdamart.md) |
| CDM — Context Effects from Raw Choice Data | [Summary](papers/summaries/cdm_context_effects.md) |

---

## Progress Log

| Date | Milestone |
|------|-----------|
| 2025-06 | Repository setup, folder structure created |
| | Data loader implementation |
| | RankNet implementation and training |
| | LambdaRank implementation and training |
| | LambdaMART implementation and training |
| | Comparative analysis and plots |
| | Report writing |
