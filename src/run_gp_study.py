"""Global-pool attribution (Part A) + multi-seed robustness (Part B).

Architecture: ConvNeXt-Tiny global-pool → heads (no TransformerEncoder).
Env: anerbc, CVD=1, bf16 + channels_last, validated 896 recipe.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# GPU selection is left to the caller (the reported runs used CUDA_VISIBLE_DEVICES=1).

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import RUNS, SEED, DataConfig, make_loaders  # noqa: E402
from src.model import HemaFormer, HemaFormerConfig  # noqa: E402
from src.run_baselines import (  # noqa: E402
    build_model,
    ckpt_dir,
    evaluate,
    hard_reset_cuda,
    train_hires_recipe,
)
from src.train_method import (  # noqa: E402
    AblationSpec,
    collect_anemia_probs,
    parse_args as method_parse_args,
    results_to_row,
    set_seed,
    train_one,
    tune_threshold,
)

METHOD_DIR = RUNS / "method"
BASE_DIR = RUNS / "baselines"
log = logging.getLogger("gp_study")


def free_cuda() -> None:
    gc.collect()
    hard_reset_cuda()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# All global-pool; grounding/cbc toggles only.
GP_ABLATIONS = [
    AblationSpec(
        "gp_full",
        grounding=True,
        cbc_head=True,
        use_background=True,
        use_features=True,
        use_transformer=False,
    ),
    AblationSpec(
        "gp_anemia_only",
        grounding=False,
        cbc_head=False,
        use_background=False,
        use_features=False,
        use_transformer=False,
    ),
    AblationSpec(
        "gp_no_grounding",
        grounding=False,
        cbc_head=True,
        use_background=True,
        use_features=True,
        use_transformer=False,
    ),
    AblationSpec(
        "gp_no_cbc",
        grounding=True,
        cbc_head=False,
        use_background=True,
        use_features=True,
        use_transformer=False,
    ),
]


# ---------------------------------------------------------------------------
# Stats: DeLong + bootstrap AUROC gap
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong (handles ties)."""
    order = np.argsort(x)
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        # average rank of ties (1-indexed ranks → convert later if needed)
        mid = 0.5 * (i + j - 1) + 1
        for k in range(i, j):
            ranks[order[k]] = mid
        i = j
    return ranks


def delong_roc_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Paired DeLong test for AUROC(p1) vs AUROC(p2). Returns z, p, aucs."""
    y = np.asarray(y_true).astype(np.int64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 1 or n < 1:
        return {"auc1": float("nan"), "auc2": float("nan"), "z": float("nan"), "p": float("nan")}

    # Structural components V10, V01 for each model
    def structural(probs: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        xp, xn = probs[pos], probs[neg]
        # AUC via midranks on concatenated
        all_s = np.concatenate([xp, xn])
        r_all = _compute_midrank(all_s)
        r_pos = r_all[:m]
        r_neg = r_all[m:]
        # Also midranks within positives / negatives for variance
        r_xp = _compute_midrank(xp)
        r_xn = _compute_midrank(xn)
        auc = (r_pos.sum() - m * (m + 1) / 2.0) / (m * n)
        v10 = (r_pos - r_xp) / n  # DeLong V10
        v01 = 1.0 - (r_neg - r_xn) / m  # DeLong V01
        return float(auc), v10, v01

    auc1, v10_1, v01_1 = structural(p1)
    auc2, v10_2, v01_2 = structural(p2)

    # Covariance of (AUC1 - AUC2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    # Var(d) = S10[0,0]/m + S01[0,0]/n - 2*(S10[0,1]/m + S01[0,1]/n)
    # for difference: L = [1, -1]
    L = np.array([1.0, -1.0])
    var = float(L @ (s10 / m + s01 / n) @ L)
    diff = auc1 - auc2
    if var <= 0 or not np.isfinite(var):
        z = 0.0 if abs(diff) < 1e-12 else np.sign(diff) * np.inf
    else:
        z = diff / np.sqrt(var)
    # two-sided normal p
    from math import erfc

    p = float(erfc(abs(z) / np.sqrt(2.0)))
    return {"auc1": auc1, "auc2": auc2, "diff": float(diff), "z": float(z), "p": p}


def bootstrap_auroc_gap(
    y: np.ndarray,
    p_method: np.ndarray,
    p_base: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Bootstrap 95% CI of AUROC(method) − AUROC(baseline) on paired test scores."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_method = np.asarray(p_method, dtype=np.float64)
    p_base = np.asarray(p_base, dtype=np.float64)
    n = len(y)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        gaps.append(roc_auc_score(yy, p_method[idx]) - roc_auc_score(yy, p_base[idx]))
    gaps = np.asarray(gaps, dtype=np.float64)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    point = float(roc_auc_score(y, p_method) - roc_auc_score(y, p_base))
    return {
        "gap": point,
        "boot_mean": float(gaps.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot_used": int(len(gaps)),
    }


# ---------------------------------------------------------------------------
# Part A — GP ablation
# ---------------------------------------------------------------------------

def reuse_no_transformer_as_gp_full() -> dict | None:
    src = METHOD_DIR / "no_transformer"
    dst = METHOD_DIR / "gp_full"
    src_res = src / "results.json"
    if not src_res.is_file():
        return None
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "results.json", "history.json", "config.json"):
        s, d = src / name, dst / name
        if s.is_file() and (not d.is_file() or name == "results.json"):
            shutil.copy2(s, d)
    results = json.loads((dst / "results.json").read_text(encoding="utf-8"))
    results["tag"] = "gp_full"
    results["reused_from"] = "no_transformer"
    results["use_transformer"] = False
    (dst / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # ensure test scores for Part B
    if not (dst / "test_scores.npz").is_file() and (dst / "best.pt").is_file():
        export_method_scores(dst, results)
    return results


def export_method_scores(out_dir: Path, results: dict | None = None) -> None:
    """Export test_scores.npz from an existing gp ckpt."""
    device = torch.device("cuda:0")
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg") or {}
    if not cfg:
        # fall back to config.json ablation
        conf = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        ab = conf.get("ablation") or {}
        cfg = {
            "backbone": conf.get("backbone", "convnext_tiny"),
            "n_layers": conf.get("n_layers", 3),
            "cbc_head": ab.get("cbc_head", True),
            "grounding": ab.get("grounding", True),
            "use_transformer": ab.get("use_transformer", False),
            "use_background": ab.get("use_background", True),
            "use_features": ab.get("use_features", True),
        }
    model = HemaFormer(HemaFormerConfig(**{
        k: cfg[k] for k in (
            "backbone", "n_layers", "cbc_head", "grounding",
            "use_transformer", "use_background", "use_features",
        ) if k in cfg
    }))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device=device, memory_format=torch.channels_last)
    data_cfg = DataConfig(long_side=896, batch_size=4, num_workers=8, cache_ram=False)
    _, _, test_loader, _ = make_loaders(data_cfg, balanced_train=False)
    y, p, _, _ = collect_anemia_probs(model, test_loader, device)
    np.savez_compressed(out_dir / "test_scores.npz", y=y, prob=p)
    del model, test_loader
    free_cuda()
    log.info("Exported test_scores.npz → %s (auroc≈%.4f)", out_dir, roc_auc_score(y, p))


def run_part_a(args) -> pd.DataFrame:
    device = torch.device("cuda:0")
    rows = []
    for spec in GP_ABLATIONS:
        out_dir = METHOD_DIR / spec.tag
        if spec.tag == "gp_full":
            reused = reuse_no_transformer_as_gp_full()
            if reused is not None and (out_dir / "results.json").is_file():
                log.info("Reused no_transformer → gp_full (auroc=%.4f)", reused["anemia"]["auroc"])
                rows.append(results_to_row(reused))
                continue
        if (out_dir / "results.json").is_file():
            log.info("Skip existing %s", out_dir)
            results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
            if "acc_at_tuned_thr" in results.get("anemia", {}):
                rows.append(results_to_row(results))
                if not (out_dir / "test_scores.npz").is_file() and (out_dir / "best.pt").is_file():
                    export_method_scores(out_dir, results)
                continue
        results = train_one(spec, args, device, out_dir, seed=SEED)
        # strip arrays already handled inside train_one for JSON; row from cleaned results
        clean = {k: v for k, v in results.items() if k not in ("test_y", "test_prob")}
        rows.append(results_to_row(clean))
        free_cuda()

    free_cuda()
    df = pd.DataFrame(rows)
    df = df.sort_values("anemia_auroc", ascending=False, na_position="last")
    csv_path = METHOD_DIR / "gp_ablation.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)
    print("\n" + "=" * 110)
    print("PART A — GP ABLATION (sorted by anemia AUROC)")
    print("=" * 110)
    show = df.copy()
    for c in show.columns:
        if c in ("tag", "pred_sha1"):
            continue
        show[c] = show[c].apply(
            lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v))
            else (f"{v:.4f}" if isinstance(v, float) else v)
        )
    if "pred_sha1" in show.columns:
        show["pred_sha1"] = show["pred_sha1"].astype(str).str[:12]
    cols = [
        "tag", "anemia_auroc", "anemia_acc_tuned",
        "background_macro_f1", "feature_macro_mAP",
        "ap_elliptocytes", "ap_target_cells", "ap_anisocytosis",
        "cbc_HGB_mae", "cbc_MCV_mae", "best_epoch", "pred_sha1",
    ]
    print(show[[c for c in cols if c in show.columns]].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Part B — robustness
# ---------------------------------------------------------------------------

def method_spec_from_tag(tag: str) -> AblationSpec:
    match = [s for s in GP_ABLATIONS if s.tag == tag]
    if not match:
        raise ValueError(f"Unknown gp tag {tag}")
    return match[0]


def baseline_out_dir(seed: int) -> Path:
    base = ckpt_dir("convnext_tiny", True, 896)
    if seed == SEED:
        return base
    return base.parent / f"{base.name}_seed{seed}"


def method_seed_dir(tag: str, seed: int) -> Path:
    if seed == SEED:
        return METHOD_DIR / tag
    return METHOD_DIR / f"{tag}_seed{seed}"


def ensure_baseline_scores(seed: int, device: torch.device, args) -> dict:
    """Train/reuse convnext@896 for seed; return metrics + ensure test_scores.npz.

    Also computes VAL-tuned thr → test acc@thr for fairness with method.
    """
    free_cuda()
    out_dir = baseline_out_dir(seed)
    metrics_path = out_dir / "test_metrics.json"
    ckpt_path = out_dir / "best.pt"
    scores_path = out_dir / "test_scores.npz"

    if metrics_path.is_file() and ckpt_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            scores_path.is_file()
            and metrics.get("acc_at_tuned_thr") is not None
            and metrics.get("threshold") is not None
        ):
            log.info("Reuse baseline seed=%d (scores+tuned-thr ready)", seed)
            metrics["seed"] = seed
            return metrics

    if not (metrics_path.is_file() and ckpt_path.is_file()):
        row = train_hires_recipe(
            "convnext_tiny",
            device=device,
            long_side=896,
            batch_size=8,
            grad_accum=3,
            seed=seed,
            out_dir=out_dir,
            skip_if_exists=False,
            cache_ram=False,
            num_workers=args.num_workers,
            pred_hash_registry={},
        )
        if row is None:
            raise RuntimeError(f"Baseline seed={seed} failed")
        free_cuda()
    else:
        log.info("Reuse baseline ckpt seed=%d @ %s — computing tuned thr", seed, out_dir)

    # VAL-tuned thr for acc@thr (eval batch 4 to avoid OOM after method runs)
    free_cuda()
    data_cfg = DataConfig(
        long_side=896, batch_size=4, num_workers=args.num_workers, cache_ram=False
    )
    _, val_loader, test_loader, _ = make_loaders(data_cfg)
    model = build_model("convnext_tiny", pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last)

    def _probs(loader):
        model.eval()
        ys, ps = [], []
        for batch in loader:
            x = batch["image"].to(device).to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)[:, 1]
            ys.append(batch["anemia"].numpy())
            ps.append(p.detach().cpu().numpy())
        return np.concatenate(ys), np.concatenate(ps)

    y_val, p_val = _probs(val_loader)
    thr, _ = tune_threshold(y_val, p_val)
    y_te, p_te = _probs(test_loader)
    acc_thr = float(accuracy_score(y_te, (p_te >= thr).astype(np.int64)))
    auroc = float(roc_auc_score(y_te, p_te))
    np.savez_compressed(scores_path, y=y_te, prob=p_te)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics.update({
        "acc_at_tuned_thr": acc_thr,
        "threshold": float(thr),
        "auroc": auroc,
        "seed": seed,
        "model": "convnext_tiny",
        "long_side": 896,
    })
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    del model, val_loader, test_loader
    free_cuda()
    return metrics


def ensure_method_seed(tag: str, seed: int, device: torch.device, args) -> dict:
    free_cuda()
    out_dir = method_seed_dir(tag, seed)
    res_path = out_dir / "results.json"
    if res_path.is_file() and "acc_at_tuned_thr" in json.loads(res_path.read_text())["anemia"]:
        log.info("Reuse method %s seed=%d", tag, seed)
        results = json.loads(res_path.read_text(encoding="utf-8"))
        if not (out_dir / "test_scores.npz").is_file():
            export_method_scores(out_dir, results)
        free_cuda()
        return results
    spec = method_spec_from_tag(tag)
    # clone tag for logging but keep architecture; train_one writes under out_dir
    run_spec = AblationSpec(
        tag=f"{tag}_seed{seed}" if seed != SEED else tag,
        grounding=spec.grounding,
        cbc_head=spec.cbc_head,
        use_background=spec.use_background,
        use_features=spec.use_features,
        use_transformer=False,
    )
    results = train_one(run_spec, args, device, out_dir, seed=seed)
    clean = {k: v for k, v in results.items() if k not in ("test_y", "test_prob")}
    # normalize tag in results for aggregation
    clean["tag"] = tag
    (out_dir / "results.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    free_cuda()
    return clean


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["y"], z["prob"]


def fmt_mean_std(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return ""
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def run_part_b(winner_tag: str, args, seeds: list[int] | None = None) -> pd.DataFrame:
    device = torch.device("cuda:0")
    if seeds is None:
        seeds = [42, 1, 2]

    method_rows = []
    base_rows = []
    per_seed_stats = []

    for seed in seeds:
        log.info("=== Robustness seed=%d | method=%s vs convnext@896 ===", seed, winner_tag)
        free_cuda()
        m = ensure_method_seed(winner_tag, seed, device, args)
        free_cuda()
        b = ensure_baseline_scores(seed, device, args)
        free_cuda()

        my, mp = load_scores(method_seed_dir(winner_tag, seed) / "test_scores.npz")
        by, bp = load_scores(baseline_out_dir(seed) / "test_scores.npz")
        assert np.array_equal(my, by), f"label mismatch seed={seed}"

        dl = delong_roc_test(my, mp, bp)
        gap = float(roc_auc_score(my, mp) - roc_auc_score(by, bp))
        per_seed_stats.append({
            "seed": seed,
            "method_auroc": float(roc_auc_score(my, mp)),
            "baseline_auroc": float(roc_auc_score(by, bp)),
            "gap": gap,
            "delong_z": dl["z"],
            "delong_p": dl["p"],
        })

        a = m["anemia"]
        method_rows.append({
            "config": winner_tag,
            "seed": seed,
            "anemia_auroc": a["auroc"],
            "anemia_acc_tuned": a["acc_at_tuned_thr"],
            "background_macro_f1": m["background_3"]["macro_f1"],
            "feature_macro_mAP": m["features"]["macro_mAP"],
            "cbc_MCV_mae": (m.get("cbc_mae_original") or {}).get("MCV", {}).get("mae")
            if m.get("cbc_mae_original") else None,
        })
        base_rows.append({
            "config": "convnext@896",
            "seed": seed,
            "anemia_auroc": b["auroc"],
            "anemia_acc_tuned": b["acc_at_tuned_thr"],
            "background_macro_f1": None,
            "feature_macro_mAP": None,
            "cbc_MCV_mae": None,
        })

    # Extend seeds if gap is marginal
    gaps = [r["gap"] for r in per_seed_stats]
    mean_gap = float(np.mean(gaps))
    # pooled (average probs across seeds, same test order)
    probs_m = []
    probs_b = []
    y_ref = None
    for seed in seeds:
        y, pm = load_scores(method_seed_dir(winner_tag, seed) / "test_scores.npz")
        _, pb = load_scores(baseline_out_dir(seed) / "test_scores.npz")
        if y_ref is None:
            y_ref = y
        probs_m.append(pm)
        probs_b.append(pb)
    p_m_avg = np.mean(np.stack(probs_m, 0), axis=0)
    p_b_avg = np.mean(np.stack(probs_b, 0), axis=0)
    boot = bootstrap_auroc_gap(y_ref, p_m_avg, p_b_avg, n_boot=10000, seed=0)

    marginal = (not boot["excludes_zero"]) or abs(mean_gap) < 0.01
    extra = [s for s in (3, 4) if s not in seeds]
    if marginal and extra:
        log.info(
            "Gap marginal (mean=%.4f, CI excludes 0=%s) — extending seeds %s",
            mean_gap, boot["excludes_zero"], extra,
        )
        return run_part_b(winner_tag, args, seeds=seeds + extra)

    # summary table
    summary = []
    for config, rows in ((winner_tag, method_rows), ("convnext@896", base_rows)):
        summary.append({
            "config": config,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "anemia_auroc": fmt_mean_std([r["anemia_auroc"] for r in rows]),
            "anemia_acc_tuned": fmt_mean_std([r["anemia_acc_tuned"] for r in rows]),
            "background_macro_f1": fmt_mean_std(
                [r["background_macro_f1"] for r in rows if r["background_macro_f1"] is not None]
            ) if any(r["background_macro_f1"] is not None for r in rows) else "",
            "feature_macro_mAP": fmt_mean_std(
                [r["feature_macro_mAP"] for r in rows if r["feature_macro_mAP"] is not None]
            ) if any(r["feature_macro_mAP"] is not None for r in rows) else "",
            "cbc_MCV_mae": fmt_mean_std(
                [r["cbc_MCV_mae"] for r in rows if r["cbc_MCV_mae"] is not None]
            ) if any(r["cbc_MCV_mae"] is not None for r in rows) else "",
            "gap_mean": mean_gap if config == winner_tag else "",
            "gap_ci95_lo": boot["ci95_lo"] if config == winner_tag else "",
            "gap_ci95_hi": boot["ci95_hi"] if config == winner_tag else "",
            "gap_ci_excludes_zero": boot["excludes_zero"] if config == winner_tag else "",
            "delong_p_per_seed": (
                ";".join(f"s{r['seed']}:{r['delong_p']:.4g}" for r in per_seed_stats)
                if config == winner_tag else ""
            ),
        })

    df = pd.DataFrame(summary)
    csv_path = METHOD_DIR / "robustness.csv"
    df.to_csv(csv_path, index=False)

    # also dump per-seed detail
    detail = pd.DataFrame(per_seed_stats)
    detail.to_csv(METHOD_DIR / "robustness_per_seed.csv", index=False)
    (METHOD_DIR / "robustness_bootstrap.json").write_text(
        json.dumps(
            {
                "winner": winner_tag,
                "seeds": seeds,
                "mean_gap_across_seeds": mean_gap,
                "std_gap_across_seeds": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
                "pooled_avg_probs_bootstrap": boot,
                "per_seed": per_seed_stats,
                "verdict": (
                    f"Bootstrap 95% CI of anemia-AUROC gap (method−baseline) on pooled "
                    f"(seed-averaged) test scores: [{boot['ci95_lo']:.4f}, {boot['ci95_hi']:.4f}]; "
                    f"excludes 0: {boot['excludes_zero']}."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("PART B — ROBUSTNESS")
    print("=" * 110)
    print(df.to_string(index=False))
    print("\nPer-seed gaps / DeLong:")
    print(detail.to_string(index=False))
    print(f"\n*** {json.loads((METHOD_DIR / 'robustness_bootstrap.json').read_text())['verdict']} ***")
    log.info("Wrote %s", csv_path)
    return df


def parse_study_args():
    # reuse method hyperparams; add study flags
    sys_argv_bak = sys.argv
    # strip our flags before method parser sees them
    our = argparse.ArgumentParser(add_help=False)
    our.add_argument("--part", choices=["a", "b", "all"], default="all")
    our.add_argument("--winner", type=str, default=None, help="Override Part-A winner tag")
    our.add_argument("--seeds", type=str, default="42,1,2")
    known, rest = our.parse_known_args()
    sys.argv = [sys_argv_bak[0]] + rest
    args = method_parse_args()
    sys.argv = sys_argv_bak
    args.part = known.part
    args.winner = known.winner
    args.seeds = [int(x) for x in known.seeds.split(",") if x.strip()]
    return args


def main() -> None:
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(METHOD_DIR / "gp_study.log", mode="a"),
        ],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (set CUDA_VISIBLE_DEVICES to pick a GPU)")
    args = parse_study_args()
    log.info("Device cuda:0 (%s) part=%s", torch.cuda.get_device_name(0), args.part)

    winner = args.winner
    if args.part in ("a", "all"):
        df_a = run_part_a(args)
        winner = winner or str(df_a.iloc[0]["tag"])
        log.info("Part-A winner: %s", winner)

    if args.part in ("b", "all"):
        if not winner:
            # fall back to gp_ablation.csv
            csv = METHOD_DIR / "gp_ablation.csv"
            if csv.is_file():
                winner = str(pd.read_csv(csv).iloc[0]["tag"])
            else:
                winner = "gp_full"
        run_part_b(winner, args, seeds=args.seeds)


if __name__ == "__main__":
    main()
