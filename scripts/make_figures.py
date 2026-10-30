#!/usr/bin/env python3
"""Regenerate the result figures used in README.md from the shipped CSVs.

Everything here is computed from files committed under runs/ (no checkpoints and
no dataset needed), so the figures can be rebuilt on a fresh clone:

    python scripts/make_figures.py

Outputs (figures/):
    fig_cv_auroc.png         pooled 5-fold anemia AUROC with bootstrap 95% CIs
    fig_ap_bars.png          per-class poikilocyte AP with AneRBC-I support
    fig_cbc_scatter.png      predicted vs. true MCV / HGB (pooled CV)
    fig_training_curves.png  validation macro-F1 / AUROC per fold
    fig_baselines.png        diagnostic backbone ranking (70/15/15 split)
    fig_robustness.png       method vs. baseline AUROC across 5 seeds

If ./AneRBC_dataset is present, fig_dataset_samples.png is written as well;
otherwise that panel is skipped with a notice.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ART = ROOT / "artifacts"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Shared look.
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

BLUE = "#2E5EAA"
RUST = "#C8553D"
SAGE = "#5E8C61"
GREY = "#8A8A8A"
PALETTE = [BLUE, RUST, SAGE, GREY, "#7A5C9E"]

LABELS = {
    "gp_full": "HemaFormer (full)",
    "gp_no_grounding": "HemaFormer (no grounding)",
    "gp_anemia_only": "HemaFormer (anemia only)",
    "convnext@896": "ConvNeXt-Tiny @896 (baseline)",
    "gp_no_cbc": "HemaFormer (no CBC head)",
}
COLORS = {
    "gp_full": BLUE,
    "gp_no_grounding": SAGE,
    "gp_anemia_only": RUST,
    "convnext@896": GREY,
    "gp_no_cbc": BLUE,
}


def _cv_rows() -> pd.DataFrame:
    return pd.read_csv(RUNS / "method" / "cv_results.csv")


def fig_cv_auroc() -> None:
    df = _cv_rows()
    pooled = df[df["row_type"] == "pooled"].set_index("config")
    order = ["convnext@896", "gp_anemia_only", "gp_no_grounding", "gp_full"]

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(order))
    for i, cfg in enumerate(order):
        r = pooled.loc[cfg]
        lo = r["anemia_auroc"] - r["anemia_auroc_ci95_lo"]
        hi = r["anemia_auroc_ci95_hi"] - r["anemia_auroc"]
        ax.errorbar(
            r["anemia_auroc"], i, xerr=[[lo], [hi]],
            fmt="o", markersize=8, color=COLORS[cfg],
            ecolor="#333333", elinewidth=1.3, capsize=4, capthick=1.3,
        )
        ax.text(
            r["anemia_auroc_ci95_hi"] + 0.0012, i,
            f"{r['anemia_auroc']:.3f}  [{r['anemia_auroc_ci95_lo']:.3f}, "
            f"{r['anemia_auroc_ci95_hi']:.3f}]",
            va="center", ha="left", fontsize=8,
        )
    ax.set_yticks(y, [LABELS[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0.932, 0.982)
    ax.set_xlabel("Pooled 5-fold CV anemia AUROC (bootstrap 95% CI, 10k resamples)")
    ax.set_title("Anemia discrimination: statistical parity, not a win", loc="left")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig_cv_auroc.png")
    plt.close(fig)


def fig_ap_bars() -> None:
    df = _cv_rows()
    pooled = df[(df["row_type"] == "pooled") & (df["config"] == "gp_full")].iloc[0]
    manifest = pd.read_csv(ART / "manifest.csv")

    feats = [
        "elliptocytes", "anisocytosis", "polychromasia", "target_cells",
        "tear_drop_cells", "nRBCs", "schistocytes",
    ]
    rows = [{"feature": f, "ap": float(pooled[f"ap_{f}"]), "n": int(manifest[f].sum())} for f in feats]
    d = pd.DataFrame(rows).sort_values("ap", ascending=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = [BLUE if n >= 100 else RUST for n in d["n"]]
    ax.barh(np.arange(len(d)), d["ap"], color=colors, height=0.65, edgecolor="white")
    for i, (ap, n) in enumerate(zip(d["ap"], d["n"])):
        ax.text(ap + 0.012, i, f"{ap:.2f}  (n={n})", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(d)), [f.replace("_", " ") for f in d["feature"]])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average precision (pooled 5-fold CV, HemaFormer full)")
    ax.set_title("Poikilocyte AP tracks label support", loc="left")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig_ap_bars.png")
    plt.close(fig)


def fig_cbc_scatter() -> None:
    df = pd.read_csv(RUNS / "method" / "cv" / "preds_cbc.csv")
    df = df[df["mask"] == 1]
    panels = [("MCV", "MCV (fL)"), ("HGB", "HGB (g/dL)")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (field, label) in zip(axes, panels):
        sub = df[df["field"] == field]
        x, yv = sub["true"].to_numpy(), sub["pred"].to_numpy()
        mae = np.abs(x - yv).mean()
        r = np.corrcoef(x, yv)[0, 1]
        ax.scatter(x, yv, s=7, alpha=0.35, color=BLUE, linewidths=0)
        lo = float(min(x.min(), yv.min()))
        hi = float(max(x.max(), yv.max()))
        ax.plot([lo, hi], [lo, hi], "--", color=RUST, linewidth=1.1, label=r"$y=x$")
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{field}: MAE {mae:.2f}, Pearson $r$ {r:.3f} (n={len(sub)})", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cbc_scatter.png")
    plt.close(fig)


def fig_training_curves() -> None:
    import json

    base = RUNS / "method" / "cv" / "gp_full"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for k in range(5):
        hist_path = base / f"fold{k}" / "history.json"
        if not hist_path.is_file():
            continue
        h = pd.DataFrame(json.loads(hist_path.read_text()))
        axes[0].plot(h["epoch"], h["val_macro_f1"], color=PALETTE[k], linewidth=1.2, label=f"fold {k}")
        axes[1].plot(h["epoch"], h["val_auroc"], color=PALETTE[k], linewidth=1.2, label=f"fold {k}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("Validation macro-F1 (checkpoint criterion)", loc="left")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].set_title("Validation AUROC", loc="left")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_training_curves.png")
    plt.close(fig)


def fig_baselines() -> None:
    df = pd.read_csv(RUNS / "baselines.csv")
    df = df.sort_values(["model", "init"])
    names = [f"{m}\n({i})" for m, i in zip(df["model"], df["init"])]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = [BLUE if p else GREY for p in df["pretrained"]]
    ax.bar(np.arange(len(df)), df["auroc"], color=colors, width=0.65, edgecolor="white")
    for i, v in enumerate(df["auroc"]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(df)), names, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test AUROC (70/15/15 split)")
    ax.set_title("Backbone ranking: ImageNet pretraining dominates", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=GREY),
    ]
    ax.legend(
        handles, ["pretrained", "from scratch"], frameon=False, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
    )
    fig.savefig(FIG / "fig_baselines.png")
    plt.close(fig)


def fig_robustness() -> None:
    df = pd.read_csv(RUNS / "method" / "robustness_per_seed.csv").sort_values("seed")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    w = 0.38
    ax.bar(x - w / 2, df["method_auroc"], width=w, color=BLUE, label="HemaFormer (no CBC head)")
    ax.bar(x + w / 2, df["baseline_auroc"], width=w, color=GREY, label="ConvNeXt-Tiny @896")
    for xi, g, p in zip(x, df["gap"], df["delong_p"]):
        ax.annotate(
            f"Δ{g:+.3f}\np={p:.3f}",
            (xi, 0.973), ha="center", fontsize=7, linespacing=1.4,
        )
    ax.set_xticks(x, [f"seed {s}" for s in df["seed"]])
    ax.set_ylim(0.90, 1.045)
    ax.set_ylabel("Pooled CV anemia AUROC")
    ax.set_title("Multi-seed robustness: no consistent margin", loc="left")
    ax.legend(
        frameon=False, fontsize=8, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig_robustness.png")
    plt.close(fig)


def fig_dataset_samples() -> None:
    root = ROOT / "AneRBC_dataset" / "AneRBC-I"
    picks = [
        ("Anemic_individuals", "086_a", "anemic"),
        ("Anemic_individuals", "329_a", "anemic"),
        ("Healthy_individuals", "456_h", "healthy"),
        ("Healthy_individuals", "192_h", "healthy"),
    ]
    rows = []
    for cohort, stem, label in picks:
        orig = root / cohort / "Original_images" / f"{stem}.png"
        rgb = root / cohort / "RGB_segmented" / f"{stem}.png"
        if orig.is_file() and rgb.is_file():
            rows.append((stem, label, orig, rgb))
    if not rows:
        print("fig_dataset_samples.png skipped (AneRBC_dataset/ not found)")
        return

    from PIL import Image

    fig, axes = plt.subplots(len(rows), 2, figsize=(6.2, 2.0 * len(rows)))
    for r, (stem, label, orig, rgb) in enumerate(rows):
        axes[r, 0].imshow(Image.open(orig))
        axes[r, 0].set_ylabel(f"{stem}\n{label}", fontsize=8)
        axes[r, 1].imshow(Image.open(rgb))
        for c in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Original smear", fontsize=9)
            axes[r, 1].set_title("Dataset segmentation GT", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dataset_samples.png")
    plt.close(fig)


def main() -> None:
    fig_cv_auroc()
    fig_ap_bars()
    fig_cbc_scatter()
    fig_training_curves()
    fig_baselines()
    fig_robustness()
    fig_dataset_samples()
    for p in sorted(FIG.glob("fig_*.png")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
