"""5-fold stratified CV over all 1000 AneRBC-I images (global-pool recipe @896).

Builds artifacts/folds.csv, trains gp_full / gp_no_grounding / gp_anemia_only /
convnext@896 on each fold, pools held-out predictions, reports bootstrap CIs.
Env: anerbc, CVD=1, bf16 + channels_last.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# GPU selection is left to the caller (the reported runs used CUDA_VISIBLE_DEVICES=1).

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    CBC_MODEL_FIELDS,
    FEATURE_COLS,
    RUNS,
    SEED,
    DataConfig,
    build_folds_csv,
    feature_pos_weight_from_df,
    joined_for_fold,
    make_fold_loaders,
)
from src.model import HemaFormer, HemaFormerConfig  # noqa: E402
from src.run_baselines import build_model, hard_reset_cuda  # noqa: E402
from src.train_method import (  # noqa: E402
    AblationSpec,
    compute_loss,
    eval_anemia_for_earlystop,
    eval_full,
    feature_pos_weight,
    parse_args as method_parse_args,
    results_to_row,
    set_seed,
    sha1_array,
    train_one,
    tune_threshold,
)

METHOD_DIR = RUNS / "method"
CV_DIR = METHOD_DIR / "cv"
log = logging.getLogger("cv_study")

N_FOLDS = 5
VAL_FRAC = 0.15

CV_CONFIGS: dict[str, dict] = {
    "gp_full": {
        "type": "method",
        "spec": AblationSpec(
            "gp_full",
            grounding=True,
            cbc_head=True,
            use_background=True,
            use_features=True,
            use_transformer=False,
        ),
    },
    "gp_no_grounding": {
        "type": "method",
        "spec": AblationSpec(
            "gp_no_grounding",
            grounding=False,
            cbc_head=True,
            use_background=True,
            use_features=True,
            use_transformer=False,
        ),
    },
    "gp_anemia_only": {
        "type": "method",
        "spec": AblationSpec(
            "gp_anemia_only",
            grounding=False,
            cbc_head=False,
            use_background=False,
            use_features=False,
            use_transformer=False,
        ),
    },
    "convnext@896": {"type": "baseline"},
}


def free_cuda() -> None:
    gc.collect()
    hard_reset_cuda()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def bootstrap_ci(
    fn,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Generic bootstrap CI for scalar metric fn() -> float."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"point": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "excludes_zero": False}
    a = np.asarray(vals, dtype=np.float64)
    lo, hi = np.percentile(a, [2.5, 97.5])
    point = float(a.mean())
    return {
        "point": point,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot_used": len(vals),
    }


def bootstrap_auroc_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    n = len(y)
    point = float(roc_auc_score(y, prob))

    def _fn(rng):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], prob[idx]
        if yy.min() == yy.max():
            return None
        return float(roc_auc_score(yy, pp))

    out = bootstrap_ci(_fn, n_boot=n_boot, seed=seed)
    out["point"] = point
    return out


def bootstrap_acc_ci_from_pred(y: np.ndarray, pred: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=np.int64)
    point = float(accuracy_score(y, pred))
    n = len(y)

    def _fn(rng):
        idx = rng.integers(0, n, size=n)
        return float(accuracy_score(y[idx], pred[idx]))

    out = bootstrap_ci(_fn, n_boot=n_boot, seed=seed)
    out["point"] = point
    return out


def bootstrap_gap_ci(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    n = len(y)
    point = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))

    def _fn(rng):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            return None
        return float(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))

    out = bootstrap_ci(_fn, n_boot=n_boot, seed=seed)
    out["point"] = point
    return out


def fold_out_dir(config: str, fold_id: int) -> Path:
    safe = config.replace("@", "_at_").replace("/", "_")
    return CV_DIR / safe / f"fold{fold_id}"


@torch.no_grad()
def eval_baseline_split(model, loader, device: torch.device) -> dict:
    model.eval()
    ys, probs, preds = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = batch["anemia"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits_f = logits.float()
        p = torch.softmax(logits_f, dim=1)[:, 1]
        pred = logits_f.argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    from sklearn.metrics import recall_score as rs

    rec_pc = rs(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "recall_healthy": float(rec_pc[0]),
        "recall_anemic": float(rec_pc[1]),
    }


@torch.no_grad()
def eval_baseline_probs(model, loader, device: torch.device):
    model.eval()
    ys, ps, stems = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        ys.append(batch["anemia"].numpy())
        ps.append(p.detach().cpu().numpy())
        stems.extend(batch["stem"])
    return np.concatenate(ys), np.concatenate(ps), np.asarray(stems)


def train_baseline_fold(
    fold_id: int,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """convnext@896 hires recipe on one CV fold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    free_cuda()

    loaders = make_fold_loaders(
        fold_id,
        long_side=896,
        batch_size=8,
        num_workers=args.num_workers,
        cache_ram=False,
        balanced_train=True,
    )
    train_loader, val_loader, test_loader, _, fold_df = loaders

    model = build_model("convnext_tiny", pretrained=True)
    set_seed(SEED)
    model = model.to(device=device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    peak_lr = 5e-5
    weight_decay = 1e-4
    warmup_epochs = args.warmup_epochs
    min_epochs = args.min_epochs
    patience = args.patience
    max_epochs = args.max_epochs
    grad_accum = 3

    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    warmup = LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    ckpt_path = out_dir / "best.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()

    best_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = batch["anemia"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * grad_accum * x.size(0)
            n += x.size(0)
        if len(train_loader) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_m = eval_baseline_split(model, val_loader, device)
        history.append({"epoch": epoch, "val_macro_f1": val_m["macro_f1"], **val_m})
        log.info(
            "[convnext@896 f%d] ep %03d val_mF1=%.4f rec_h=%.3f rec_a=%.3f",
            fold_id, epoch, val_m["macro_f1"], val_m["recall_healthy"], val_m["recall_anemic"],
        )
        if val_m["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1 = val_m["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                },
                ckpt_path,
            )
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last)

    y_val, p_val, _ = eval_baseline_probs(model, val_loader, device)
    thr, _ = tune_threshold(y_val, p_val)
    y_te, p_te, stems_te = eval_baseline_probs(model, test_loader, device)
    pred_te = (p_te >= thr).astype(np.int64)

    results = {
        "tag": "convnext@896",
        "fold": fold_id,
        "anemia": {
            "auroc": float(roc_auc_score(y_te, p_te)),
            "threshold": float(thr),
            "acc_at_tuned_thr": float(accuracy_score(y_te, pred_te)),
            "pred_sha1": sha1_array(pred_te),
        },
        "background_3": {"macro_f1": None},
        "features": {"macro_mAP": None, "ap_per_class": {}},
        "cbc_mae_original": {},
        "best_epoch": best_epoch,
        "seconds": time.time() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "test_scores.npz",
        y=y_te,
        prob=p_te,
        pred=pred_te,
        stems=stems_te,
        fold=fold_id,
    )

    del model, train_loader, val_loader, test_loader
    free_cuda()
    return results


def train_method_fold(
    config: str,
    spec: AblationSpec,
    fold_id: int,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    free_cuda()
    loaders = make_fold_loaders(
        fold_id,
        long_side=args.long_side,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_ram=args.cache_ram,
        balanced_train=True,
    )
    train_loader, val_loader, test_loader, cbc_stats, fold_df = loaders
    train_df = fold_df[fold_df["role"] == "train"]
    pos_w = feature_pos_weight_from_df(train_df, device)

    results = train_one(
        spec,
        args,
        device,
        out_dir,
        seed=SEED,
        loaders=(train_loader, val_loader, test_loader),
        cbc_stats=cbc_stats,
        pos_w=pos_w,
    )
    results["fold"] = fold_id
    results["config"] = config
    clean = {k: v for k, v in results.items() if k not in ("test_y", "test_prob", "test_stems")}
    (out_dir / "results.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


def run_fold(config: str, fold_id: int, args, device: torch.device, skip_existing: bool) -> dict:
    out_dir = fold_out_dir(config, fold_id)
    res_path = out_dir / "results.json"
    if skip_existing and res_path.is_file() and (out_dir / "test_scores.npz").is_file():
        log.info("Skip existing %s fold%d", config, fold_id)
        return json.loads(res_path.read_text(encoding="utf-8"))

    log.info("=== Train %s fold %d ===", config, fold_id)
    cfg = CV_CONFIGS[config]
    if cfg["type"] == "baseline":
        return train_baseline_fold(fold_id, args, device, out_dir)
    return train_method_fold(config, cfg["spec"], fold_id, args, device, out_dir)


def load_pooled_scores(config: str) -> dict:
    """Concatenate held-out test predictions across folds."""
    ys, ps, preds, stems, folds = [], [], [], [], []
    for k in range(N_FOLDS):
        path = fold_out_dir(config, k) / "test_scores.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        z = np.load(path, allow_pickle=True)
        ys.append(z["y"])
        ps.append(z["prob"])
        preds.append(z["pred"] if "pred" in z else (z["prob"] >= 0.5).astype(np.int64))
        stems.append(z["stems"])
        folds.append(np.full(len(z["y"]), k, dtype=np.int64))
    y = np.concatenate(ys)
    prob = np.concatenate(ps)
    pred = np.concatenate(preds)
    stem = np.concatenate(stems)
    fold = np.concatenate(folds)
    if len(stem) != len(set(stem.tolist())):
        raise RuntimeError(f"{config}: duplicate stems in pooled predictions")
    return {"y": y, "prob": prob, "pred": pred, "stems": stem, "fold": fold}


def pooled_threshold(config: str) -> float:
    """Average per-fold val-tuned thresholds (applied per-fold at eval time; for pooled acc CI use pooled preds with mean thr)."""
    thrs = []
    for k in range(N_FOLDS):
        r = json.loads((fold_out_dir(config, k) / "results.json").read_text())
        thrs.append(float(r["anemia"]["threshold"]))
    return float(np.mean(thrs))


def aggregate_aux_pooled(config: str) -> dict:
    """Pool auxiliary-task arrays saved per fold (method configs only)."""
    bg_y, bg_p = [], []
    feat_y, feat_logit = [], []

    for k in range(N_FOLDS):
        aux_path = fold_out_dir(config, k) / "aux_arrays.npz"
        if not aux_path.is_file():
            continue
        z = np.load(aux_path, allow_pickle=True)
        if "bg_y" in z:
            bg_y.append(z["bg_y"])
            bg_p.append(z["bg_p"])
        if "feat_y" in z:
            feat_y.append(z["feat_y"])
            feat_logit.append(z["feat_logit"])

    out: dict = {}
    if bg_y:
        y_b, p_b = np.concatenate(bg_y), np.concatenate(bg_p)
        out["background_macro_f1"] = float(f1_score(y_b, p_b, average="macro", zero_division=0))
    if feat_y:
        y_f = np.concatenate(feat_y)
        prob_f = 1.0 / (1.0 + np.exp(-np.concatenate(feat_logit)))
        aps = []
        ap_per = {}
        for i, name in enumerate(FEATURE_COLS):
            if y_f[:, i].sum() < 1:
                ap_per[name] = float("nan")
                continue
            ap = float(average_precision_score(y_f[:, i], prob_f[:, i]))
            ap_per[name] = ap
            aps.append(ap)
        out["feature_macro_mAP"] = float(np.mean(aps)) if aps else None
        # Supported subset: classes with n>=100 positives in AneRBC-I.
        supported = ["elliptocytes", "anisocytosis", "polychromasia", "target_cells"]
        supported_aps = [ap_per[n] for n in supported if n in ap_per and np.isfinite(ap_per[n])]
        out["feature_mAP_supported"] = float(np.mean(supported_aps)) if supported_aps else None
        out["ap_per_class"] = ap_per

    # Denormalize per fold then pool (each fold uses its own train CBC stats).
    errs: dict[str, list[np.ndarray]] = {f: [] for f in CBC_MODEL_FIELDS}
    for k in range(N_FOLDS):
        aux_path = fold_out_dir(config, k) / "aux_arrays.npz"
        if not aux_path.is_file():
            continue
        z = np.load(aux_path, allow_pickle=True)
        if "cbc_pred_z" not in z:
            continue
        stats = json.loads(str(z["cbc_stats_json"]))
        pz = z["cbc_pred_z"]
        tz = z["cbc_true_z"]
        m = z["cbc_mask"]
        for i, field in enumerate(CBC_MODEL_FIELDS):
            mean, std = stats[field]["mean"], stats[field]["std"]
            valid = m[:, i] > 0.5
            if valid.sum() == 0:
                continue
            err = np.abs((pz[:, i] * std + mean)[valid] - (tz[:, i] * std + mean)[valid])
            errs[field].append(err)
    cbc_out: dict[str, dict[str, float]] = {}
    for field, chunks in errs.items():
        if not chunks:
            continue
        cat = np.concatenate(chunks)
        cbc_out[field] = {"mae": float(cat.mean()), "n": int(cat.size)}
    out["cbc_mae"] = cbc_out
    return out


@torch.no_grad()
def save_aux_arrays(config: str, fold_id: int, spec: AblationSpec, args, device: torch.device) -> None:
    """After training, dump aux-task test arrays for pooling."""
    out_dir = fold_out_dir(config, fold_id)
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model = HemaFormer(
        HemaFormerConfig(
            backbone=args.backbone,
            n_layers=args.n_layers,
            cbc_head=spec.cbc_head,
            grounding=spec.grounding,
            use_transformer=False,
            use_background=spec.use_background,
            use_features=spec.use_features,
        )
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device=device, memory_format=torch.channels_last)

    _, _, test_loader, cbc_stats, _ = make_fold_loaders(
        fold_id,
        long_side=args.long_side,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_ram=False,
    )

    bg_y, bg_p = [], []
    feat_y, feat_logit = [], []
    cbc_pred_z, cbc_true_z, cbc_mask = [], [], []

    for batch in test_loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, report_texts=None, return_grounding=False)
        if spec.use_background and out["logits_background"] is not None:
            bg_y.append(batch["background_3"].numpy())
            bg_p.append(out["logits_background"].float().argmax(1).cpu().numpy())
        if spec.use_features and out["logits_features"] is not None:
            feat_y.append(batch["features"].numpy())
            feat_logit.append(out["logits_features"].float().cpu().numpy())
        if spec.cbc_head and out["pred_cbc"] is not None:
            cbc_pred_z.append(out["pred_cbc"].float().cpu().numpy())
            cbc_true_z.append(batch["cbc_value"].numpy())
            cbc_mask.append(batch["cbc_mask"].numpy())

    save: dict = {"cbc_stats_json": json.dumps(cbc_stats)}
    if bg_y:
        save["bg_y"] = np.concatenate(bg_y)
        save["bg_p"] = np.concatenate(bg_p)
    if feat_y:
        save["feat_y"] = np.concatenate(feat_y)
        save["feat_logit"] = np.concatenate(feat_logit)
    if cbc_pred_z:
        save["cbc_pred_z"] = np.concatenate(cbc_pred_z)
        save["cbc_true_z"] = np.concatenate(cbc_true_z)
        save["cbc_mask"] = np.concatenate(cbc_mask)
    if len(save) > 1:
        np.savez_compressed(out_dir / "aux_arrays.npz", **save)

    del model, test_loader
    free_cuda()


def fold_metrics_row(config: str, fold_id: int, results: dict) -> dict:
    a = results["anemia"]
    ap = results.get("features", {}).get("ap_per_class") or {}
    cbc = results.get("cbc_mae_original") or {}

    def _cbc_mae(field: str):
        v = cbc.get(field)
        if isinstance(v, dict):
            return v.get("mae")
        return v

    def _cbc_n(field: str):
        v = cbc.get(field)
        if isinstance(v, dict):
            return v.get("n")
        return None

    row = {
        "row_type": "fold",
        "config": config,
        "fold": fold_id,
        "anemia_auroc": a["auroc"],
        "anemia_acc_tuned": a.get("acc_at_tuned_thr"),
        "anemia_thr": a.get("threshold"),
        "background_macro_f1": (results.get("background_3") or {}).get("macro_f1"),
        "feature_macro_mAP": (results.get("features") or {}).get("macro_mAP"),
        "pred_sha1": a.get("pred_sha1"),
        "best_epoch": results.get("best_epoch"),
    }
    for name in FEATURE_COLS:
        row[f"ap_{name}"] = ap.get(name)
    for field in CBC_MODEL_FIELDS:
        row[f"cbc_{field}_mae"] = _cbc_mae(field)
        row[f"cbc_{field}_n"] = _cbc_n(field)
    return row


def fmt_mean_std(vals: list[float]) -> str:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(a) == 0:
        return ""
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def build_cv_results(configs: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    pooled_scores: dict[str, dict] = {}

    for config in configs:
        pooled = load_pooled_scores(config)
        pooled_scores[config] = pooled
        y, prob, pred = pooled["y"], pooled["prob"], pooled["pred"]

        auroc_ci = bootstrap_auroc_ci(y, prob, n_boot=10000, seed=0)
        acc_ci = bootstrap_acc_ci_from_pred(y, pred, n_boot=10000, seed=1)
        aux = aggregate_aux_pooled(config) if config != "convnext@896" else {}

        fold_rows = []
        for k in range(N_FOLDS):
            r = json.loads((fold_out_dir(config, k) / "results.json").read_text())
            fold_rows.append(fold_metrics_row(config, k, r))
            rows.append(fold_rows[-1])

        fr = pd.DataFrame(fold_rows)
        pooled_row = {
            "row_type": "pooled",
            "config": config,
            "fold": "all",
            "n_test": len(y),
            "anemia_auroc": auroc_ci["point"],
            "anemia_auroc_ci95_lo": auroc_ci["ci95_lo"],
            "anemia_auroc_ci95_hi": auroc_ci["ci95_hi"],
            "anemia_acc_tuned": acc_ci["point"],
            "anemia_acc_ci95_lo": acc_ci["ci95_lo"],
            "anemia_acc_ci95_hi": acc_ci["ci95_hi"],
            "anemia_thr_mean": pooled_threshold(config),
            "background_macro_f1": aux.get("background_macro_f1"),
            "feature_macro_mAP": aux.get("feature_macro_mAP"),
            "feature_mAP_supported": aux.get("feature_mAP_supported"),
            "anemia_auroc_mean_std_folds": fmt_mean_std(fr["anemia_auroc"].tolist()),
            "anemia_acc_mean_std_folds": fmt_mean_std(fr["anemia_acc_tuned"].tolist()),
            "bg_mf1_mean_std_folds": fmt_mean_std(fr["background_macro_f1"].tolist()),
            "feat_mAP_mean_std_folds": fmt_mean_std(fr["feature_macro_mAP"].tolist()),
            "mcv_mae_mean_std_folds": fmt_mean_std(fr["cbc_MCV_mae"].tolist()) if "cbc_MCV_mae" in fr else "",
            "pred_sha1_pooled": hashlib.sha1(prob.tobytes()).hexdigest()[:16],
        }
        ap_per = aux.get("ap_per_class") or {}
        for name in FEATURE_COLS:
            pooled_row[f"ap_{name}"] = ap_per.get(name)
        cbc_mae = aux.get("cbc_mae") or {}
        for field in CBC_MODEL_FIELDS:
            entry = cbc_mae.get(field) or {}
            if isinstance(entry, dict):
                pooled_row[f"cbc_{field}_mae"] = entry.get("mae")
                pooled_row[f"cbc_{field}_n"] = entry.get("n")
            else:
                pooled_row[f"cbc_{field}_mae"] = entry
                pooled_row[f"cbc_{field}_n"] = None
        rows.append(pooled_row)

    # contrasts on pooled paired predictions
    contrasts = [
        ("grounding", "gp_full", "gp_no_grounding"),
        ("multi_task", "gp_full", "gp_anemia_only"),
        ("vs_baseline", "gp_full", "convnext@896"),
    ]
    y_ref = pooled_scores["gp_full"]["y"]
    for name, a_cfg, b_cfg in contrasts:
        pa = pooled_scores[a_cfg]["prob"]
        pb = pooled_scores[b_cfg]["prob"]
        assert np.array_equal(y_ref, pooled_scores[b_cfg]["y"])
        gap_ci = bootstrap_gap_ci(y_ref, pa, pb, n_boot=10000, seed=hash(name) % 10000)
        rows.append({
            "row_type": "contrast",
            "contrast": name,
            "config_a": a_cfg,
            "config_b": b_cfg,
            "gap_auroc": gap_ci["point"],
            "gap_ci95_lo": gap_ci["ci95_lo"],
            "gap_ci95_hi": gap_ci["ci95_hi"],
            "gap_ci_excludes_zero": gap_ci["excludes_zero"],
            "n_pooled": len(y_ref),
        })

    df = pd.DataFrame(rows)
    out_path = METHOD_DIR / "cv_results.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    return df


def parse_cv_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--build-folds-only", action="store_true")
    p.add_argument("--rebuild-csv-only", action="store_true",
                   help="Skip training/inference; rebuild cv_results.csv from existing fold outputs.")
    p.add_argument("--configs", type=str, default="gp_full,gp_no_grounding,gp_anemia_only,convnext@896")
    p.add_argument("--folds", type=str, default="0,1,2,3,4")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    known, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = method_parse_args()
    args.build_folds_only = known.build_folds_only
    args.rebuild_csv_only = known.rebuild_csv_only
    args.cv_configs = [c.strip() for c in known.configs.split(",") if c.strip()]
    args.cv_folds = [int(x) for x in known.folds.split(",") if x.strip()]
    args.skip_existing = known.skip_existing
    return args


def main() -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(CV_DIR / "cv_study.log", mode="a"),
        ],
    )
    args = parse_cv_args()

    folds_path = ROOT / "artifacts" / "folds.csv"
    if not folds_path.is_file():
        log.info("Building %s", folds_path)
        build_folds_csv(n_folds=N_FOLDS, val_frac=VAL_FRAC, seed=SEED)
    else:
        log.info("Using existing %s", folds_path)

    if args.build_folds_only:
        df = pd.read_csv(folds_path)
        print(df.groupby(["fold", "role"]).size().unstack(fill_value=0))
        return

    for config in args.cv_configs:
        if config not in CV_CONFIGS:
            raise SystemExit(f"Unknown config {config}; choose from {list(CV_CONFIGS)}")

    if args.rebuild_csv_only:
        df = build_cv_results(args.cv_configs)
        print("\n" + "=" * 100)
        print("5-FOLD CV — POOLED RESULTS (rebuilt from existing fold outputs)")
        print("=" * 100)
        pooled = df[df["row_type"] == "pooled"]
        cols = [
            "config", "n_test", "anemia_auroc", "background_macro_f1",
            "feature_macro_mAP", "feature_mAP_supported",
            "ap_elliptocytes", "ap_polychromasia", "ap_schistocytes",
            "cbc_HGB_mae", "cbc_MCV_mae", "cbc_MCH_mae", "cbc_MCHC_mae",
            "cbc_HGB_n", "cbc_MCV_n", "cbc_MCH_n", "cbc_MCHC_n",
        ]
        print(pooled[[c for c in cols if c in pooled.columns]].to_string(index=False))
        print("\nContrasts:")
        print(df[df["row_type"] == "contrast"].to_string(index=False))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (set CUDA_VISIBLE_DEVICES to pick a GPU)")
    device = torch.device("cuda:0")
    log.info("CV study on %s", torch.cuda.get_device_name(0))

    for config in args.cv_configs:
        for fold_id in args.cv_folds:
            results = run_fold(config, fold_id, args, device, args.skip_existing)
            if CV_CONFIGS[config]["type"] == "method":
                spec = CV_CONFIGS[config]["spec"]
                if spec.use_background or spec.use_features or spec.cbc_head:
                    aux_path = fold_out_dir(config, fold_id) / "aux_arrays.npz"
                    if not aux_path.is_file():
                        save_aux_arrays(config, fold_id, spec, args, device)
            free_cuda()

    df = build_cv_results(args.cv_configs)

    print("\n" + "=" * 100)
    print("5-FOLD CV — POOLED RESULTS")
    print("=" * 100)
    pooled = df[df["row_type"] == "pooled"]
    print(pooled[["config", "n_test", "anemia_auroc", "anemia_auroc_ci95_lo", "anemia_auroc_ci95_hi",
                  "anemia_acc_tuned", "anemia_acc_ci95_lo", "anemia_acc_ci95_hi"]].to_string(index=False))
    print("\nContrasts (AUROC gap bootstrap 95% CI):")
    print(df[df["row_type"] == "contrast"].to_string(index=False))


if __name__ == "__main__":
    main()
