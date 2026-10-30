# Reproducibility notes

Everything reported in the paper is produced by the scripts in `src/` and is shipped as plain
CSV/JSON under `runs/`. This file records the exact protocol, the environment, and the per-fold
numbers so that a reviewer can audit the tables without a GPU.

## 1. Protocol

| Item | Value |
|---|---|
| Dataset | AneRBC-I, 1,000 images (500 anemic / 500 healthy), 1224×960 |
| Primary protocol | Pooled 5-fold stratified CV (`StratifiedKFold`, shuffle, seed 42) |
| Per-fold sizes | train 680 / val 120 / test 200 |
| Pooling | Each image is a test sample exactly once → 1,000 pooled predictions |
| Threshold | Tuned per fold on **validation** macro-F1, applied to that fold's test predictions |
| CIs | Bootstrap, 10,000 resamples (AUROC, accuracy); seeds are fixed in the code |
| Contrast CIs | Paired bootstrap on pooled test scores (Δ AUROC) |
| Extra test | DeLong test per seed for the multi-seed study |
| Diagnostic split | 70/15/15 (seed 42, stratified), test n = 150 — backbones and the architecture probe only |

Metric implementations live in `src/run_cv_study.py` (`bootstrap_ci`, `bootstrap_auroc_ci`,
`bootstrap_acc_ci_from_pred`, `bootstrap_gap_ci`) and `src/run_gp_study.py`
(`delong_roc_test`, `bootstrap_auroc_gap`).

## 2. Training recipe (HemaFormer)

| Item | Value |
|---|---|
| Backbone | ImageNet-pretrained ConvNeXt-Tiny (`timm`, `convnext_tiny.in12k_ft_in1k`) |
| Aggregation | Global average pooling (primary); patch-token transformer is the probe |
| Input | long side 896, padded to a multiple of 32, ImageNet mean/std |
| Augmentation (train) | horizontal flip p=0.5, rotation ±10°, mild ColourJitter |
| Optimiser | AdamW, backbone LR 1e-5, new-head LR 5e-5, weight decay 0.05 |
| Schedule | 5-epoch linear warmup → cosine decay |
| Early stopping | min 15 / patience 12 / max 60 epochs on validation macro-F1 |
| Batch | micro-batch 4 × grad-accum 6 = effective 24 |
| Sampling | class-balanced `WeightedRandomSampler` on the anemia label |
| Precision / layout | bf16 autocast + `channels_last` |
| Loss | unweighted sum of anemia CE + background CE + poikilocyte BCE + masked L1 on CBC |
| Grounding (full only) | InfoNCE (τ = 0.07) vs. frozen BiomedBERT-base-abstract; training only |
| Hardware | 1 × NVIDIA GeForce RTX 3060 (12 GB), ≈30 s/epoch at long-side 896 |

## 3. Environment

The environment recorded for the reported runs (`anerbc`) pins: Python 3.11.15, `torch 2.6.0+cu124`,
`torchvision 0.21.0+cu124`, `timm 1.0.28`, `transformers 5.14.1`, `tokenizers 0.22.2`, `numpy 2.4.4`,
`pandas 3.0.5`, `pillow 12.2.0`, `scikit-learn 1.9.0`, `matplotlib 3.11.1`.

`requirements.txt` states these as minimum bounds instead of hard pins; `environment.yml` builds a
conda env on top of them. Nothing in the code depends on a specific CUDA build — any recent
`torch`+`torchvision` pair works (this release was additionally re-verified on Python 3.14,
`torch 2.14.0+cu130`, `timm 1.0.29`, `transformers 4.57.6`). `transformers` is optional: without it
(or without the BiomedBERT weights) grounding switches off automatically and the global-pool model
still trains.

## 4. Shipped results and how to regenerate them

| File | Produced by |
|---|---|
| `artifacts/*.csv` | `python -m src.build_anerbc_i_labels` |
| `artifacts/folds.csv` | `python -m src.run_cv_study --build-folds-only` |
| `runs/method/cv_results.csv` | `python -m src.run_cv_study --configs ... --folds 0,1,2,3,4` |
| `runs/method/ablation.csv` | `python -m src.train_method --ablation` |
| `runs/method/gp_ablation.csv` | global-pool ablation grid (same entry point, `use_transformer=False`) |
| `runs/method/robustness*.csv/json` | `python -m src.run_gp_study --part all --seeds 42,1,2,3,4` |
| `runs/baselines.csv` | `python -m src.run_baselines --mode full` |
| `runs/cbc_stats.json` | written on first load by `src/data.py` (`ensure_cbc_stats`) |
| `runs/method/cv/preds_cbc.csv` | pooled CBC predictions from the CV study |
| `figures/fig_*.png` | `python scripts/make_figures.py` (CPU only) |

Three figures are static assets rather than script output: `fig_architecture.png` (the model
diagram) and `fig_qualitative.png` (prediction panel, rendered from the camera-ready figure at 3×)
are committed because they are drawn/rendered rather than computed; `fig_dataset_samples.png` is
regenerated when `AneRBC_dataset/` is present and skipped otherwise.

`runs/method/cv/<config>/fold<k>/` holds `config.json` (full hyper-parameters + package versions +
prediction SHA-1), `history.json` (per-epoch curves), `results.json` (fold metrics) and
`test_scores.npz` (raw probabilities, so every metric can be recomputed).

### Verification performed for this release

- Re-running the miner on the original AneRBC-I download reproduces `manifest.csv` label columns,
  `split.csv`, `labels_morphology.csv`, `labels_cbc.csv`, `morph_freq.csv` and
  `cbc_parse_dropped.csv` byte-identically (Windows: CRLF and `\` separators aside).
- Rebuilding `folds.csv` from `manifest.csv` reproduces the shipped file exactly (CRLF aside) with
  the documented 680/120/200 per-fold sizes.
- `python -m src.<each script> --help` runs, and `python -m compileall src` is clean.
- Figures are regenerated from the shipped CSVs only (no checkpoints, no dataset).

## 5. Per-fold results (pooled CV)

### HemaFormer (full) — `gp_full`

| fold | AUROC | Acc@thr | thr | bg macro-F1 | feat mAP | HGB MAE | MCV MAE | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.961 | 0.895 | 0.47 | 0.517 | 0.425 | 2.35 | 8.79 | 24 |
| 1 | 0.922 | 0.865 | 0.50 | 0.555 | 0.479 | 2.14 | 8.43 | 22 |
| 2 | 0.966 | 0.875 | 0.37 | 0.663 | 0.466 | 2.36 | 8.64 | 44 |
| 3 | 0.979 | 0.900 | 0.56 | 0.685 | 0.508 | 2.34 | 7.22 | 20 |
| 4 | 0.970 | 0.885 | 0.36 | 0.584 | 0.465 | 2.31 | 7.38 | 10 |

### HemaFormer (no grounding) — `gp_no_grounding`

| fold | AUROC | Acc@thr | bg macro-F1 | feat mAP | HGB MAE | MCV MAE | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.969 | 0.905 | 0.534 | 0.422 | 2.31 | 8.47 | 24 |
| 1 | 0.921 | 0.825 | 0.560 | 0.503 | 2.33 | 9.04 | 19 |
| 2 | 0.959 | 0.860 | 0.590 | 0.527 | 2.34 | 8.90 | 29 |
| 3 | 0.975 | 0.910 | 0.644 | 0.532 | 2.11 | 7.63 | 36 |
| 4 | 0.983 | 0.920 | 0.631 | 0.473 | 2.29 | 7.47 | 42 |

### HemaFormer (anemia-only) — `gp_anemia_only`

| fold | AUROC | Acc@thr | thr | best epoch |
|---:|---:|---:|---:|---:|
| 0 | 0.967 | 0.905 | 0.49 | 49 |
| 1 | 0.915 | 0.830 | 0.50 | 28 |
| 2 | 0.961 | 0.860 | 0.47 | 28 |
| 3 | 0.974 | 0.875 | 0.37 | 29 |
| 4 | 0.978 | 0.900 | 0.67 | 42 |

### ConvNeXt-Tiny @896 baseline — `convnext@896`

| fold | AUROC | Acc@thr | thr | best epoch |
|---:|---:|---:|---:|---:|
| 0 | 0.952 | 0.835 | 0.42 | 9 |
| 1 | 0.881 | 0.795 | 0.50 | 5 |
| 2 | 0.950 | 0.830 | 0.50 | 32 |
| 3 | 0.980 | 0.915 | 0.50 | 18 |
| 4 | 0.973 | 0.890 | 0.73 | 38 |

Mean ± std over folds (from `cv_results.csv`):

| Config | AUROC | Acc@thr | bg macro-F1 | feat mAP | MCV MAE |
|---|---|---|---|---|---|
| `gp_full` | 0.9596 ± 0.0221 | 0.8840 ± 0.0143 | 0.6005 ± 0.0715 | 0.4684 ± 0.0299 | 8.089 ± 0.736 |
| `gp_no_grounding` | 0.9615 ± 0.0243 | 0.8840 ± 0.0402 | 0.5919 ± 0.0465 | 0.4914 ± 0.0454 | 8.302 ± 0.720 |
| `gp_anemia_only` | 0.9588 ± 0.0255 | 0.8740 ± 0.0307 | — | — | — |
| `convnext@896` | 0.9471 ± 0.0393 | 0.8530 ± 0.0486 | — | — | — |

## 6. Multi-seed robustness (Part B)

Seeds 42, 1, 2, 3, 4; method = HemaFormer without the CBC head (the Part-A winner), baseline =
ConvNeXt-Tiny @896.

| seed | method AUROC | baseline AUROC | Δ | DeLong p |
|---:|---:|---:|---:|---:|
| 42 | 0.9633 | 0.9395 | +0.0238 | 0.0096 |
| 1 | 0.9409 | 0.9357 | +0.0052 | 0.768 |
| 2 | 0.9599 | 0.9574 | +0.0025 | 0.815 |
| 3 | 0.9597 | 0.9395 | +0.0203 | 0.079 |
| 4 | 0.9513 | 0.9581 | −0.0068 | 0.474 |

Mean gap +0.0090, pooled-bootstrap 95% CI [−0.0047, 0.0226] → includes zero. Mean AUROC:
0.9550 ± 0.0091 (method) vs. 0.9460 ± 0.0108 (baseline).

## 7. Known limitations (as stated in the paper)

- Single public dataset; no cross-site or cross-stain generalisation claim.
- Each CV test fold has 200 images, so CIs stay wide even when pooled.
- Schistocytes (n = 31) and nRBCs (n = 33) are support-limited; their AP is not reliable.
- The transformer probe has no confidence interval and is reported as a preliminary comparison.
- Negative results are statements about anemia AUROC at AneRBC-I scale — not a claim that
  grounding or multi-task heads are useless for auxiliary quality or larger cohorts.
