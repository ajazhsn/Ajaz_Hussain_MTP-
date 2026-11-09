# Can Self-Attention Models Capture Context Effects?

M.Tech Project, Data Science and AI, IIT Madras
**Ajaz Hussain (DA25M006)**, guided by **Prof. Suryanarayana Sankagiri**

When people choose, they rarely look at one option alone. Whether a hotel or a phone looks good depends on what else is shown next to it. This is called a **context effect**, and it breaks the IIA assumption on which the multinomial logit and most ranking models are built. Models like CDM and LCL capture only the first-order part of context, where each other item (or the slate average) pushes an item's utility up or down.

In this project we check whether a **self-attention (Transformer) choice model** actually *learns* context effects of a known order from choice data, and whether it does this better than **DeepHalo** (Zhang et al., NeurIPS 2025), a recent model designed to control the order of interactions.

**Full report:** [`report/Ajaz_Hussain_MTP_report.pdf`](report/Ajaz_Hussain_MTP_report.pdf)

---

## Main results in short

| Question | Answer |
|---|---|
| First-order (LCL) context | Both models capture about 97% of it, and the true A can be read back from every trained model (correlation 0.999). |
| Second- and third-order context (30 seeds) | Self-attention has a lower test loss than DeepHalo on **every seed**. It captures 16.8–32.3% of the part that slate mean and sd cannot explain, against 5.7–9.8% for DeepHalo. |
| Why not more? | Mainly data. With 1,000,000 slates, self-attention reaches 38.2–53.1% (DeepHalo 15.8–21.8%); below about 100,000 slates neither model beats the simple mean+sd summary. |
| Depth | Two self-attention layers are enough for effects of order 2 to 5. Attention without its feed-forward network learns very little context. |
| Real data | No usable context in LPMC (London travel mode). A small but reliable gain on Expedia, most of it explained by comparing each hotel with the others' standardised features. |

---

## Repository layout

```
Ajaz_Hussain_MTP-/
├── README.md
├── requirements.txt
├── report/                          final report (LaTeX source + PDF)
│   ├── Ajaz_Hussain_MTP_report.pdf
│   ├── report.tex
│   ├── make_figs.py                 makes the figures in figs/ from the printed results
│   └── figs/
├── 1_lcl_first_order/               earlier work: self-attention on LCL data (thesis)
│   ├── README.md
│   └── LCL_thesis_Ajaz_Hussain.pdf
├── 2_synthetic_higher_order/        main study: self-attention vs DeepHalo (Sections 5-8 of the report)
│   ├── synth.py                     data generation, all models, training loop, command line
│   ├── calib_box.py                 calibrates the box-kernel datasets -> box_calib.json
│   ├── calib_order.py               calibrates the order-4/5 datasets -> order_calib.json
│   ├── tune_select.py               picks one configuration per family from the tuning grid
│   ├── summarize.py                 main tables (gap to twin closed, hard part captured)
│   ├── paired.py                    seed-by-seed paired comparisons (wins out of 30)
│   ├── size_summary.py              training-size sweep tables
│   ├── depth_summary.py             depth vs order tables
│   ├── interp_train.py              retrains the chosen models and saves their weights
│   ├── interpret.py                 the look inside: LCL fit, attention ablation, attention closeness, IIA test
│   ├── slurm/                       the SLURM job scripts used on the WSAI cluster
│   └── notebooks/
│       └── higher_order_calibration_kaggle.ipynb   first timing + strength calibration (Kaggle)
└── 3_real_data/                     LPMC and Expedia (Section 8.11)
    ├── comparison_with_deephalo_lpmc_expedia.ipynb
    └── pasar_expedia_preprocessing.ipynb
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate      # or a conda environment
pip install -r requirements.txt
```

Everything runs on CPU, but a GPU makes the sweeps much faster. On the WSAI cluster we used the conda environment `pytorch-env` and one V100 per array task. The `source /storage/opt/miniforge3/...` and `conda activate` lines in the job scripts are specific to that cluster, so change them for your own machine.

---

## Synthetic study (`2_synthetic_higher_order/`)

### The data

No data files are stored. Every dataset is regenerated from `(dataset, seed)`, so the same seed gives exactly the same data on every machine. Each slate has 10 items with 6 standard-normal features. The utility is `u_i = θ·x_i + s·λ·c_i(S)`, and the chosen item is drawn from the softmax.

| Dataset | Context term | Order |
|---|---|---|
| `D0_lcl` | LCL: `x_iᵀ A x̄_S`, A of rank 2 | 1 (linear) |
| `D1_sim` / `D1b_sim` | closeness of i to one other item (similarity, s = −1) | 1 |
| `D2_comp` / `D2b_comp` | closeness to the midpoint of two others (compromise) | 2 |
| `D3_centre` / `D3b_centre` | closeness to the centre of three others | 3 |
| `D4_quad`, `D5_quint` | closeness to the centre of four / five others (depth test only) | 4, 5 |

Closeness uses features 0 and 1 only. The `b` versions use a box kernel (radius 0.25) instead of a Gaussian (σ = 0.2). The strength λ is set so that the part that the slate mean and sd cannot explain (the **hard part**) is about 0.03 nats for every dataset. The values for D0–D3 are fixed in `synth.py`. The box and order-4/5 values are read from `box_calib.json` and `order_calib.json`, which must sit next to `synth.py`. Run the two calibration scripts once to create them.

### Models (names used on the command line)

| Name | Model |
|---|---|
| `sa_L3_w32_h2` | tuned self-attention: 3 post-norm encoder layers, width 32, 2 heads, no positional encoding (25,953 params) |
| `dhq_L4_e16_H8` | tuned DeepHalo, quadratic version: 4 layers, embedding 16, 8 heads (11,713 params) |
| `sa1_self` | **blindfolded twin**: 1-layer self-attention where each item can only attend to itself, so it has no context |
| `pool_meansd` | each item + the mean and sd of the slate (the reference for the hard part) |
| `sa_L{L}_w{W}_h{H}`, `dhq_L{L}_e{E}_H{H}` | any other size, used in the tuning grid |
| `aosoft_…`, `aolin_…` | attention-only models (no feed-forward network), used in the depth test |
| `sa1`, `sa2`, `sa1_wide`, `dh4`, … | models of the first sweep and the size controls (Section 8.7) |

### A quick test run

```bash
cd 2_synthetic_higher_order
python synth.py --datasets D2_comp --models sa_L3_w32_h2,dhq_L4_e16_H8,pool_meansd,sa1_self \
                --seeds 0 --out quick_test --patience 10,6
python summarize.py quick_test
```

Each run writes one JSON file to `quick_test/results/`. Runs that already exist are skipped, so every command can be resumed after a crash or time-out.

### Reproducing the report, step by step

Submit every job script from inside `2_synthetic_higher_order/`, for example `sbatch slurm/run_final.sbatch`.

| Step | Job script | Then run | Report section |
|---|---|---|---|
| 1. First sweep (fixed-size models) | `slurm/run_gpu.sbatch`, `slurm/run_dhq.sbatch` | `python summarize.py synth_out` | 8.5 (first look) |
| 2. Size controls (depth vs width) | `slurm/run_ctrl.sbatch` | `python paired.py synth_out` | 8.7 |
| 3. Box kernel: calibrate, then run | `slurm/calib_box.sbatch`, then `python calib_box.py --pick 0.25`, then `slurm/run_box.sbatch` | `python summarize.py synth_out` | 8.6 |
| 4. Tuning on selection seeds 100–104 | `slurm/run_tune.sbatch` | `python tune_select.py synth_tune` | 7.2 |
| 5. **Final comparison**, 30 reporting seeds (0–29) | `slurm/run_final.sbatch` | `python summarize.py synth_final`, `python paired.py synth_final` | 8.4–8.6 |
| 6. Training-size sweep | `slurm/run_size.sbatch` | `python size_summary.py` | 8.9 |
| 7. Look inside the models (5 seeds) | `slurm/run_interp.sbatch`, then `slurm/run_interp_analysis.sbatch` | read `interp_report.txt` | 8.12 |
| 8. Same with box data and 10 seeds | `slurm/run_interp2.sbatch`, then `slurm/run_interp2_analysis.sbatch` | read `interp_report_10seeds.txt` | 8.13 |
| 9. Depth vs order | `slurm/calib_order.sbatch`, then `slurm/run_depth.sbatch` | `python depth_summary.py` | 8.14 |

Tuning (step 4) uses seeds 100–104 only, and every number in the report comes from seeds 0–29, so the configuration is never chosen on the test seeds.

---

## Real data (`3_real_data/`)

- **LPMC** (London Passenger Mode Choice): 4 travel modes, all always available. Downloaded inside the notebook from `http://transp-or.epfl.ch/data/lpmc.dat`. We split by survey year.
- **Expedia** (ICDM 2013 hotel search, Kaggle): we keep only searches with `random_bool = 1`, where the hotel order on the page is random, so position cannot fake a context effect. `pasar_expedia_preprocessing.ipynb` holds the earlier PASAR work and the preprocessing; `comparison_with_deephalo_lpmc_expedia.ipynb` trains MNL, MLPs, pools, DeepHalo and self-attention (with its blindfolded twin) on both datasets over 15 seeds.

Both notebooks were written for Kaggle, so change the input paths at the top (`/kaggle/input/...`) to your own.

---

## Earlier work: LCL (`1_lcl_first_order/`)

This is our earlier study on first-order (LCL) data. It asks whether a self-attention model that predicts well has also learned the true θ and A, using a behavioural probe that reads them out of any trained model. The main results are in Section 4.4 of the report and in the thesis PDF.

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
- Vaswani et al. (2017). *Attention is all you need.* NeurIPS 2017.

The full list is in the report.

Contact: Ajaz Hussain, da25m006@smail.iitm.ac.in
