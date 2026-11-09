# Can Self-Attention Models Capture Context Effects?

M.Tech Project, Data Science and AI, IIT Madras
**Ajaz Hussain (DA25M006)**, guided by **Prof. Suryanarayana Sankagiri**

When people choose, they rarely look at one option alone. Whether a hotel or a phone looks good depends on what else is shown next to it. This is called a **context effect**, and it breaks the IIA assumption on which the multinomial logit and most ranking models are built. Models like CDM and LCL capture only the first-order part of context, where each other item (or the slate average) pushes an item's utility up or down.

This repository holds the whole MTP, in the order it was done:

1. **Learning to rank (early work).** RankNet, LambdaRank and LambdaMART on LETOR (MQ2007), and PASAR, a self-attention reranker on Expedia hotel search.
2. **First-order context: LCL and CDM (thesis).** Does a self-attention model that predicts choices well also learn the true context matrix A and the base weights θ?
3. **Higher-order context: self-attention vs DeepHalo (final report).** Does self-attention learn context effects of order 1, 2 and 3, and does it do this better than **DeepHalo** (Zhang et al., NeurIPS 2025)? We also run the models on real data (LPMC and Expedia).

**Final report:** [`report/Ajaz_Hussain_MTP_report.pdf`](report/Ajaz_Hussain_MTP_report.pdf)
**LCL thesis:** [`report/LCL_thesis_Ajaz_Hussain.pdf`](report/LCL_thesis_Ajaz_Hussain.pdf)

---

## Main results in short

| Question | Answer |
|---|---|
| First-order (LCL) context | A self-attention model that predicts well has also learned the context rule: its recovered A has cosine 0.990 with the truth, and it does not invent context on data without any. Recovery depends mostly on the amount of data. |
| Second- and third-order context (30 seeds) | Self-attention has a lower test loss than DeepHalo on **every seed**. It captures 16.8–32.3% of the part that the slate mean and sd cannot explain, against 5.7–9.8% for DeepHalo. |
| Why not more? | Mainly data. With 1,000,000 slates, self-attention reaches 38.2–53.1% (DeepHalo 15.8–21.8%). Below about 100,000 slates, neither model beats the simple mean+sd summary. |
| Depth | Two self-attention layers are enough for effects of order 2 to 5. Attention without its feed-forward network learns very little context. |
| Real data | No usable context in LPMC (London travel mode). There is a small but reliable gain on Expedia, and most of it comes from comparing each hotel with the others' standardised features. |

---

## Repository layout

```
Ajaz_Hussain_MTP-/
├── README.md
├── requirements.txt / pyproject.toml
│
├── src/                              all Python code
│   ├── data/, lcl/, metrics/,        earlier work: data loading and splits, the LCL study,
│   │   models/, training/, utils/    ranking models (RankNet, LambdaRank, LambdaMART), training loop
│   └── higher_order/                 final study: self-attention vs DeepHalo   (see below)
│
├── scripts/                          earlier work: run.py, merge.py, curves.py (LCL sweeps on the cluster)
├── slurm/                            SLURM job scripts for the WSAI cluster
│   ├── submit*.sbatch                earlier work: LCL sweeps
│   └── higher_order/                 final study: one script per experiment
│
├── notebooks/                        Jupyter notebooks
│   ├── LCL_Main_Notebook.ipynb, LCL_sweeps.ipynb, LCL_thesis_figures_.ipynb     LCL study
│   ├── Cdm-Main-Notebook.ipynb, CDM_sweeps.ipynb, Structural_parameter_recovery(CDM).ipynb   CDM study
│   ├── E0_probe_and_null_test.ipynb  behavioural probe and the no-context (null) test
│   ├── 01_preflight.ipynb, FETA_FATE_vs_Attention_comparison.ipynb, Structure_Aware_Loss_.ipynb
│   ├── Practice Notebooks/           early experiments (PASAR on Expedia, LETOR MQ2007, first LCL/CDM tries)
│   ├── higher_order_calibration_kaggle.ipynb        final study: first timing + strength calibration
│   └── comparison_with_deephalo_lpmc_expedia.ipynb  final study: real data (LPMC and Expedia)
│
├── results/
│   ├── thesis_inputs/                LCL thesis: sweep tables and training curves behind its figures
│   └── higher_order/                 final study: printed outputs of every run used in the report
├── plots/                            LCL thesis figures (ch3_*, ch4_*) and early LTR plots
├── report/                           final report (PDF + LaTeX source + figures) and the LCL thesis PDF
├── Presentation/                     slides from the project meetings
├── papers/originals/                 the main papers we read
└── dataset/                          LETOR 4.0 (MQ2007) folder for the early LTR work
```

Everything from the earlier phases is kept as it was. The final study only adds `src/higher_order/`, `slurm/higher_order/`, `results/higher_order/`, two notebooks and the report.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate      # or a conda environment
pip install -r requirements.txt
```

The final study needs `numpy`, `pandas`, `scipy`, `torch` and `matplotlib`. Everything runs on CPU, but a GPU makes the sweeps much faster. On the WSAI cluster we used the conda environment `pytorch-env` and one V100 per array task. The `source /storage/opt/miniforge3/...` and `conda activate` lines in the job scripts are specific to that cluster, so change them for your own machine.

---

## Final study: higher-order context (`src/higher_order/`)

### The files

| File | What it does |
|---|---|
| `synth.py` | data generation, all models, training loop and the command line |
| `calib_box.py` | calibrates the box-kernel datasets and writes `box_calib.json` |
| `calib_order.py` | calibrates the order-4/5 datasets and writes `order_calib.json` |
| `tune_select.py` | picks one configuration per model family from the tuning grid (validation loss only) |
| `summarize.py` | main tables: gap to twin closed, hard part captured |
| `paired.py` | seed-by-seed paired comparisons (wins out of 30) |
| `size_summary.py` | tables for the training-size sweep |
| `depth_summary.py` | tables for depth against order |
| `interp_train.py` | retrains the chosen models and saves their weights |
| `interpret.py` | the look inside the models: LCL fit, attention ablation, attention closeness, IIA test |

### The data

No data files are stored. Every dataset is regenerated from `(dataset, seed)`, so the same seed gives exactly the same data on every machine. Each slate has 10 items with 6 standard-normal features. The utility is `u_i = θ·x_i + s·λ·c_i(S)`, and the chosen item is drawn from the softmax.

| Dataset | Context term | Order |
|---|---|---|
| `D0_lcl` | LCL: `x_iᵀ A x̄_S`, with A of rank 2 | 1 (linear) |
| `D1_sim` / `D1b_sim` | closeness of i to one other item (similarity, s = −1) | 1 |
| `D2_comp` / `D2b_comp` | closeness to the midpoint of two others (compromise) | 2 |
| `D3_centre` / `D3b_centre` | closeness to the centre of three others | 3 |
| `D4_quad`, `D5_quint` | closeness to the centre of four / five others (depth test only) | 4, 5 |

Closeness uses features 0 and 1 only. The `b` versions use a box kernel (radius 0.25) instead of a Gaussian (σ = 0.2). The strength λ is set so that the part that the slate mean and sd cannot explain (the **hard part**) is about 0.03 nats for every dataset. The values for D0–D3 are fixed in `synth.py`. The values for the box and order-4/5 datasets are read from `box_calib.json` and `order_calib.json` in the same folder.

### Models (names used on the command line)

| Name | Model |
|---|---|
| `sa_L3_w32_h2` | tuned self-attention: 3 post-norm encoder layers, width 32, 2 heads, no positional encoding (25,953 parameters) |
| `dhq_L4_e16_H8` | tuned DeepHalo, quadratic version: 4 layers, embedding 16, 8 heads (11,713 parameters) |
| `sa1_self` | **blindfolded twin**: 1-layer self-attention where each item can only attend to itself, so it sees no context |
| `pool_meansd` | each item plus the mean and sd of the slate (the reference for the hard part) |
| `sa_L{L}_w{W}_h{H}`, `dhq_L{L}_e{E}_H{H}` | any other size, used in the tuning grid |
| `aosoft_…`, `aolin_…` | attention-only models (no feed-forward network), used in the depth test |
| `sa1`, `sa2`, `sa1_wide`, `dh1`–`dh4`, … | models of the first sweep and the size controls (Section 8.7 of the report) |

### A quick test run

```bash
cd src/higher_order
python synth.py --datasets D2_comp --models sa_L3_w32_h2,dhq_L4_e16_H8,pool_meansd,sa1_self \
                --seeds 0 --out quick_test --patience 10,6
python summarize.py quick_test
```

Each run writes one JSON file to `quick_test/results/`. Runs that already exist are skipped, so every command can be resumed after a crash or a time-out.

### Reproducing the report, step by step

Submit every job script from the **repo root**, for example `sbatch slurm/higher_order/run_final.sbatch`. The scripts move into `src/higher_order/` themselves, and their result folders are created there. Run the summary commands from inside `src/higher_order/`.

| Step | Job script(s) in `slurm/higher_order/` | Then run | Printed output in `results/higher_order/` | Report section |
|---|---|---|---|---|
| 1. First sweep (fixed-size models) | `run_gpu`, `run_dhq` | `python summarize.py synth_out` | `01_…`, `02_…` | 8.5 |
| 2. Size controls (depth vs width) | `run_ctrl` | `python paired.py synth_out` | `03_…` | 8.7 |
| 3. Box kernel: calibrate, then run | `calib_box`, then `python calib_box.py --pick 0.25`, then `run_box` | `python summarize.py synth_out` | `04_…` | 8.6 |
| 4. Tuning on selection seeds 100–104 | `run_tune` | `python tune_select.py synth_tune` | — | 7.2 |
| 5. **Final comparison**, reporting seeds 0–29 | `run_final` | `python summarize.py synth_final`, `python paired.py synth_final` | `05_…` | 8.4–8.6 |
| 6. Training-size sweep | `run_size` | `python size_summary.py` | `06_…` | 8.9 |
| 7. Look inside the models (5 seeds) | `run_interp`, then `run_interp_analysis` | read `interp_report.txt` | `07_…` | 8.12 |
| 8. Same with box data and 10 seeds | `run_interp2`, then `run_interp2_analysis` | read `interp_report_10seeds.txt` | `08_…` | 8.13 |
| 9. Depth against order | `calib_order`, then `run_depth` | `python depth_summary.py` | `08_…` | 8.14 |

Tuning (step 4) uses seeds 100–104 only, and every number in the report comes from seeds 0–29, so no configuration is chosen on the seeds we report.

---

## Real data (`notebooks/comparison_with_deephalo_lpmc_expedia.ipynb`)

- **LPMC** (London Passenger Mode Choice): 4 travel modes, all always available. The notebook downloads it from `http://transp-or.epfl.ch/data/lpmc.dat`, and we split by survey year.
- **Expedia** (ICDM 2013 hotel search, Kaggle): we keep only the searches with `random_bool = 1`, where the hotel order on the page is random, so position cannot fake a context effect. The preprocessing follows the PASAR notebook in `notebooks/Practice Notebooks/pasar.ipynb`.

The notebook trains MNL, MLPs, pooling models, DeepHalo and self-attention (with its blindfolded twin) on both datasets over 15 seeds. It was written for Kaggle, so change the input paths at the top (`/kaggle/input/...`) to your own.

---

## Earlier work

- **LCL and CDM study (thesis).** The notebooks `LCL_Main_Notebook`, `LCL_sweeps`, `Cdm-Main-Notebook`, `CDM_sweeps`, `Structural_parameter_recovery(CDM)` and `E0_probe_and_null_test` hold this study. The cluster sweeps were run with `scripts/run.py` and `slurm/submit*.sbatch` (merged with `scripts/merge.py`, training curves from `scripts/curves.py`), and their tables are in `results/thesis_inputs/` (see `results/README.md`). The figures are in `plots/`, and the write-up is `report/LCL_thesis_Ajaz_Hussain.pdf`. Its main results are also summarised in Section 4.4 of the final report.
- **Learning to rank.** The ranking models are in `src/models/`, with notebooks in `notebooks/Practice Notebooks/` (LETOR MQ2007 analysis, PASAR on Expedia).

---

## Rebuilding the report

```bash
cd report
python make_figs.py          # optional: remakes figs/*.pdf
pdflatex report.tex && pdflatex report.tex
```

---

## Main references

- Zhang, Wang, Gao and Li (2025). *Deep Halo: Learning context effects of any order.* NeurIPS 2025. arXiv:2601.04616.
- Tomlinson and Benson (2021). *Learning interpretable feature context effects in discrete choice.* KDD 2021.
- Seshadri, Peysakhovich and Ugander (2019). *Discovering context effects from raw choice data.* ICML 2019.
- Pobrotyn et al. (2020). *Context-aware learning to rank with self-attention.* SIGIR eCom 2020.
- Vaswani et al. (2017). *Attention is all you need.* NeurIPS 2017.

The full list is in the report, and PDFs of the main papers are in `papers/originals/`.

Contact: Ajaz Hussain, da25m006@smail.iitm.ac.in