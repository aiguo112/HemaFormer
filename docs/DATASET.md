# Dataset and label mining

This document specifies exactly how AneRBC-I is turned into the benchmark shipped under
`artifacts/`. It is the reference for [README → Dataset](../README.md#dataset) and for the paper's
"Dataset and Benchmark" section.

## 1. Source data

**AneRBC-I** (subset of the AneRBC dataset, Shahzad et al., *Database* 2024,
[doi:10.1093/database/baae120](https://doi.org/10.1093/database/baae120)):

| Folder | Files | Content |
|---|---:|---|
| `Anemic_individuals/Original_images` | 500 | full-field smears, 1224×960 RGB |
| `Healthy_individuals/Original_images` | 500 | full-field smears, 1224×960 RGB |
| `Binary_segmented` | 1,000 | dataset segmentation visualisation (unused here) |
| `RGB_segmented` | 1,000 | dataset segmentation visualisation (unused here) |
| `CBC_reports` | 1,000 | `.txt` CBC panels (one per image) |
| `Morphology_reports` | 1,000 | `.txt` free-text smear narratives (one per image) |

Naming: `NNN_a.png` (anemic) / `NNN_h.png` (healthy). Image, mask and report files share the same
stem, so they align 1:1. Reports are **per image** in AneRBC-I (unlike AneRBC-II, whose reports are
per parent case). AneRBC-II is not used in this work.

Place the data as described in [README → Dataset](../README.md#dataset); nothing else is needed.

## 2. Morphology background (3 classes)

The smear clause of the morphology report — the sentence after `RBC MORPHOLOGY / PERIPHERAL SMEAR:`
up to the WBC/platelet sections — is normalised and matched against a keyword lexicon:

| Class | Trigger |
|---|---|
| `microcytic_hypochromic` | "microcytic hypochromic" and spelling variants |
| `normocytic_normochromic` | "normocytic normochromic" and spelling variants |
| `other` | residual bin: dimorphic, macrocytic, mixed or unparseable wording |

Counts over the 1,000 images: **455 / 420 / 125**.

## 3. Poikilocyte features (7 binary multi-labels)

A feature is present (1) when its term appears anywhere in the morphology report. Whole-phrase
matching, case-insensitive, with misspelling variants:

| Feature | Patterns (examples) | Positives |
|---|---|---:|
| elliptocytes | `elliptocytes?`, `ovalocytes?` | 432 |
| anisocytosis | `anisocytosis` | 316 |
| polychromasia | `polychromasia`, `polycromasia`, `polychromatophilia` | 262 |
| target_cells | `target\s*cells?` | 218 |
| tear_drop_cells | `tear[\s\-]?drop\s*cells?`, `teardrop`, `dacrocytes?` | 93 |
| nRBCs | `nrbcs?`, `nucleated\s+rbcs?` | 33 |
| schistocytes | `schistocytes?`, `fragmented\s+rbcs?` | 31 |

Four extra terms are mined and stored in `manifest.csv` for completeness but are **not** model
targets (too sparse): anisopoikilocytosis 26, macrocytosis 14, spherocytes 6, basophilic stippling 4.
`morph_freq.csv` records all raw term counts.

## 4. CBC indices and physiologic-range masking

Seven fields are parsed from the CBC panel text; each is kept only when inside a physiologic range,
otherwise it becomes `NaN` with mask `0` (zero gradient). Unparseable values (`---`, `NaN%`,
`* Not specified`) are logged to `cbc_parse_dropped.csv`.

| Field | Range | Valid / masked | Model target |
|---|---|---:|---|
| HGB | [3, 22] g/dL | 988 / 12 | ✅ |
| HCT | [10, 65] % | 981 / 19 | — |
| MCV | [50, 130] fL | 981 / 19 | ✅ |
| MCH | [15, 45] pg | 929 / 71 | ✅ |
| MCHC | [25, 40] g/dL | 942 / 58 | ✅ |
| RDW-CV | [10, 40] % | 633 / 367 | — |
| RBC | [1.5, 7.5] ×10⁶/µL | 965 / 35 | — |

CBC targets are z-scored with **train-split** statistics (`runs/cbc_stats.json`) and denormalised at
evaluation. Per-fold statistics are recomputed from that fold's training split in
`make_fold_loaders`, so no test information leaks into the normalisation.

## 5. Anemia label

Taken from the cohort folder (`Anemic_individuals` → 1, `Healthy_individuals` → 0). It is **not**
derived from a WHO haemoglobin cut-off, and HGB is never fed to the model — otherwise the "anemia
from an image" claim would be circular.

## 6. Splits

- **Diagnostic 70/15/15 split** (`split.csv`, seed 42, stratified by anemia): train 700 / val 150 /
  test 150. Used only to rank backbones and for the architecture probe.
- **Official 5-fold CV** (`folds.csv`, seed 42, `StratifiedKFold(shuffle=True)`, then a 15 %
  validation carve-out from the remaining pool per fold, `random_state=42+k`): every image is a test
  sample exactly once; per fold **train 680 / val 120 / test 200**.

Per-split label distribution for the 70/15/15 split:

| Split | microcytic-hypo | normocytic-normo | other | elliptocytes | anisocytosis | polychromasia | target | tear-drop | nRBCs | schistocytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train (700) | 316 | 294 | 90 | 304 | 219 | 186 | 150 | 61 | 25 | 24 |
| val (150) | 64 | 64 | 22 | 61 | 52 | 39 | 28 | 20 | 3 | 2 |
| test (150) | 75 | 62 | 13 | 67 | 45 | 37 | 40 | 12 | 5 | 5 |

## 7. Files produced by `src/build_anerbc_i_labels.py`

| File | Rows × cols | Contents |
|---|---|---|
| `manifest.csv` | 1,000 × 27 | one row per image: cohort, anemia, all five paths, mined background, 11 feature flags, 7 CBC values |
| `split.csv` | 1,000 × 2 | `stem`, `split` ∈ {train, val, test} for the 70/15/15 partition |
| `labels_morphology.csv` | 1,000 × 13 | mined background + feature flags |
| `labels_cbc.csv` | 1,000 × 8 | parsed CBC values per field |
| `morph_freq.csv` | 16 × 2 | raw term counts (features + backgrounds) |
| `cbc_parse_dropped.csv` | 27 × 3 | every value rejected during parsing, with the raw string |

`folds.csv` (5 × 1,000 rows) is produced separately by `python -m src.run_cv_study --build-folds-only`.

**Reproducibility check.** Re-running the miner against the original AneRBC-I download regenerates
`manifest.csv` label columns, `split.csv`, `labels_morphology.csv`, `labels_cbc.csv`,
`morph_freq.csv` and `cbc_parse_dropped.csv` **byte-identically**; the only differences on Windows
are CRLF line endings and `\` path separators inside the `*_path` columns. The shipped `folds.csv`
likewise reproduces exactly (CRLF aside).
