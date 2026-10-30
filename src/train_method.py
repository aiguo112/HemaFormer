"""Train / evaluate HemaFormer + ablation grid on AneRBC-I (long_side=896).

Validated recipe: batch4×accum6, AdamW bb1e-5/new5e-5, warmup5/cosine,
WeightedRandomSampler, min15/patience12/max60, collapse guard.
Adds VAL-tuned anemia threshold → test acc@thr; AUROC is threshold-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# GPU selection is left to the caller: the reported runs pinned GPU 1 of a
# multi-GPU host via CUDA_VISIBLE_DEVICES=1. Nothing is forced here so that
# single-GPU machines work out of the box.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    BACKGROUND_3_CLASSES,
    CBC_MODEL_FIELDS,
    FEATURE_COLS,
    RUNS,
    SEED,
    DataConfig,
    ensure_cbc_stats,
    load_joined_tables,
    make_loaders,
)
from src.model import HemaFormer, HemaFormerConfig, info_nce_loss, masked_mse  # noqa: E402

METHOD_DIR = RUNS / "method"
log = logging.getLogger("train_method")

CONVNEXT_896_ACC = 0.853
CONVNEXT_896_AUROC = 0.939
BEST_BASELINE_AUROC = 0.94


@dataclass
class AblationSpec:
    tag: str
    grounding: bool = True
    cbc_head: bool = True
    use_background: bool = True
    use_features: bool = True
    use_transformer: bool = True


ABLATIONS = [
    AblationSpec("full"),
    AblationSpec(
        "anemia_only",
        grounding=False,
        cbc_head=False,
        use_background=False,
        use_features=False,
        use_transformer=True,
    ),
    AblationSpec("no_grounding", grounding=False),
    AblationSpec("no_cbc", cbc_head=False),
    AblationSpec("no_transformer", use_transformer=False),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha1_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def collect_versions() -> dict[str, str | None]:
    import importlib.metadata as md

    names = [
        "torch", "torchvision", "timm", "transformers", "tokenizers", "einops",
        "pandas", "numpy", "pillow", "scikit-learn", "albumentations",
    ]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def feature_pos_weight(device: torch.device) -> torch.Tensor:
    df = load_joined_tables()
    train = df[df["split"] == "train"]
    n = len(train)
    weights = []
    for c in FEATURE_COLS:
        pos = float(train[c].sum())
        weights.append((n - pos) / max(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Choose threshold maximizing macro-F1 on VAL. Returns (thr, best_macro_f1)."""
    best_thr, best_f1 = 0.5, -1.0
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob]))
    for thr in candidates:
        thr = float(thr)
        if thr <= 0.0 or thr >= 1.0:
            continue
        pred = (y_prob >= thr).astype(np.int64)
        f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def compute_loss(
    out: dict,
    batch: dict,
    *,
    pos_weight: torch.Tensor,
    use_cbc: bool,
    use_grounding: bool,
    use_background: bool,
    use_features: bool,
    criterion_anemia: nn.Module,
    criterion_bg: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_a = criterion_anemia(out["logits_anemia"], batch["anemia"])
    total = 1.0 * loss_a
    parts = {
        "loss_anemia": float(loss_a.detach()),
        "loss_background": 0.0,
        "loss_features": 0.0,
        "loss_cbc": 0.0,
        "loss_grounding": 0.0,
    }

    if use_background and out.get("logits_background") is not None:
        loss_bg = criterion_bg(out["logits_background"], batch["background_3"])
        total = total + 0.5 * loss_bg
        parts["loss_background"] = float(loss_bg.detach())

    if use_features and out.get("logits_features") is not None:
        loss_f = F.binary_cross_entropy_with_logits(
            out["logits_features"], batch["features"], pos_weight=pos_weight
        )
        total = total + 0.5 * loss_f
        parts["loss_features"] = float(loss_f.detach())

    if use_cbc and out.get("pred_cbc") is not None:
        loss_c = masked_mse(out["pred_cbc"], batch["cbc_value"], batch["cbc_mask"])
        total = total + 0.3 * loss_c
        parts["loss_cbc"] = float(loss_c.detach())

    if use_grounding and "z_img_proj" in out and "z_text_proj" in out:
        loss_g = info_nce_loss(out["z_img_proj"], out["z_text_proj"], temperature=0.07)
        total = total + 0.3 * loss_g
        parts["loss_grounding"] = float(loss_g.detach())

    parts["loss_total"] = float(total.detach())
    return total, parts


@torch.no_grad()
def collect_anemia_probs(model: HemaFormer, loader, device: torch.device):
    model.eval()
    ys, probs, logits_list, stems = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, report_texts=None, return_grounding=False)
            logits = out["logits_anemia"].float()
        ys.append(batch["anemia"].numpy())
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        logits_list.append(logits.cpu().numpy().astype(np.float32))
        stems.extend(batch["stem"])
    return (
        np.concatenate(ys),
        np.concatenate(probs),
        np.concatenate(logits_list),
        np.asarray(stems),
    )


@torch.no_grad()
def eval_anemia_for_earlystop(model, loader, device) -> dict:
    y, prob, logits, _ = collect_anemia_probs(model, loader, device)
    pred = (prob >= 0.5).astype(np.int64)  # early-stop uses default thr; final uses tuned
    # Actually early-stop on macro-F1 of argmax / 0.5 is fine; use argmax for consistency with before
    pred = logits.argmax(axis=1).astype(np.int64)
    rec_pc = recall_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)),
        "recall_healthy": float(rec_pc[0]),
        "recall_anemic": float(rec_pc[1]),
    }


@torch.no_grad()
def eval_full(
    model: HemaFormer,
    val_loader,
    test_loader,
    device: torch.device,
    cbc_stats: dict,
    *,
    use_cbc: bool,
    use_background: bool,
    use_features: bool,
) -> dict:
    """Tune thr on VAL macro-F1; apply to TEST. Also aux-task metrics on TEST."""
    y_val, p_val, _, _ = collect_anemia_probs(model, val_loader, device)
    thr, thr_val_f1 = tune_threshold(y_val, p_val)

    y_te, p_te, logits_te, stems_te = collect_anemia_probs(model, test_loader, device)
    pred_te = (p_te >= thr).astype(np.int64)
    anemia = {
        "auroc": float(roc_auc_score(y_te, p_te)),
        "threshold": float(thr),
        "val_macro_f1_at_thr": float(thr_val_f1),
        "acc_at_tuned_thr": float(accuracy_score(y_te, pred_te)),
        "precision_at_tuned_thr": float(
            precision_score(y_te, pred_te, average="binary", zero_division=0)
        ),
        "recall_at_tuned_thr": float(
            recall_score(y_te, pred_te, average="binary", zero_division=0)
        ),
        "f1_at_tuned_thr": float(f1_score(y_te, pred_te, average="binary", zero_division=0)),
        "macro_f1_at_tuned_thr": float(
            f1_score(y_te, pred_te, average="macro", zero_division=0)
        ),
        "pred_sha1": sha1_array(pred_te.astype(np.int64)),
        "logits_sha1": sha1_array(logits_te.astype(np.float32)),
    }

    # Aux tasks on test (only if heads exist)
    bg_y, bg_p = [], []
    feat_y, feat_logit = [], []
    cbc_pred_z, cbc_true_z, cbc_mask = [], [], []
    model.eval()
    for batch in test_loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, report_texts=None, return_grounding=False)
        if use_background and out["logits_background"] is not None:
            bg_y.append(batch["background_3"].numpy())
            bg_p.append(out["logits_background"].float().argmax(1).cpu().numpy())
        if use_features and out["logits_features"] is not None:
            feat_y.append(batch["features"].numpy())
            feat_logit.append(out["logits_features"].float().cpu().numpy())
        if use_cbc and out["pred_cbc"] is not None:
            cbc_pred_z.append(out["pred_cbc"].float().cpu().numpy())
            cbc_true_z.append(batch["cbc_value"].numpy())
            cbc_mask.append(batch["cbc_mask"].numpy())

    if bg_y:
        y_b, p_b = np.concatenate(bg_y), np.concatenate(bg_p)
        background = {
            "accuracy": float(accuracy_score(y_b, p_b)),
            "macro_f1": float(f1_score(y_b, p_b, average="macro", zero_division=0)),
        }
    else:
        background = {"accuracy": None, "macro_f1": None}

    if feat_y:
        y_f = np.concatenate(feat_y)
        prob_f = 1.0 / (1.0 + np.exp(-np.concatenate(feat_logit)))
        ap_per, aps = {}, []
        for i, name in enumerate(FEATURE_COLS):
            if y_f[:, i].sum() < 1:
                ap_per[name] = float("nan")
                continue
            ap = float(average_precision_score(y_f[:, i], prob_f[:, i]))
            ap_per[name] = ap
            aps.append(ap)
        features = {"ap_per_class": ap_per, "macro_mAP": float(np.mean(aps)) if aps else None}
    else:
        features = {"ap_per_class": {}, "macro_mAP": None}

    cbc_out: dict = {}
    if cbc_pred_z:
        pz = np.concatenate(cbc_pred_z)
        tz = np.concatenate(cbc_true_z)
        m = np.concatenate(cbc_mask)
        for i, field in enumerate(CBC_MODEL_FIELDS):
            mean, std = cbc_stats[field]["mean"], cbc_stats[field]["std"]
            valid = m[:, i] > 0.5
            if valid.sum() == 0:
                cbc_out[field] = {"mae": None, "n": 0}
            else:
                mae = float(
                    np.abs((pz[:, i] * std + mean)[valid] - (tz[:, i] * std + mean)[valid]).mean()
                )
                cbc_out[field] = {"mae": mae, "n": int(valid.sum())}

    return {
        "anemia": anemia,
        "background_3": background,
        "features": features,
        "cbc_mae_original": cbc_out,
        # numpy scores for robustness / DeLong (stripped before JSON write)
        "test_y": y_te,
        "test_prob": p_te,
        "test_stems": stems_te,
    }


def train_one(
    spec: AblationSpec,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    seed: int = SEED,
    *,
    loaders: tuple | None = None,
    cbc_stats: dict | None = None,
    pos_w: torch.Tensor | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    batch_size = args.batch_size
    grad_accum = args.grad_accum
    long_side = args.long_side

    model_cfg = HemaFormerConfig(
        backbone=args.backbone,
        n_layers=args.n_layers,
        cbc_head=spec.cbc_head,
        grounding=spec.grounding,
        use_transformer=spec.use_transformer,
        use_background=spec.use_background,
        use_features=spec.use_features,
    )
    model = HemaFormer(model_cfg).to(device=device, memory_format=torch.channels_last)
    set_seed(seed)

    grounding_on = bool(model.grounding_enabled)
    cbc_on = bool(model.head_cbc is not None)
    use_bg = bool(model.head_background is not None)
    use_feat = bool(model.head_features is not None)

    config = {
        "tag": spec.tag,
        "ablation": asdict(spec),
        "env": "anerbc",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "seed": seed,
        "amp": "bf16",
        "memory_format": "channels_last",
        "long_side": long_side,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "backbone_lr": args.backbone_lr,
        "new_lr": args.new_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "max_epochs": args.max_epochs,
        "backbone": args.backbone,
        "n_layers": args.n_layers,
        "cbc_head": cbc_on,
        "grounding": grounding_on,
        "use_transformer": spec.use_transformer,
        "use_background": use_bg,
        "use_features": use_feat,
        "packages": collect_versions(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log.info(
        "=== %s | grounding=%s cbc=%s bg=%s feat=%s transformer=%s ===",
        spec.tag, grounding_on, cbc_on, use_bg, use_feat, spec.use_transformer,
    )

    data_cfg = DataConfig(
        long_side=long_side,
        batch_size=batch_size,
        num_workers=args.num_workers,
        cache_ram=args.cache_ram,
        drop_last_train=True,
    )
    if loaders is None:
        train_loader, val_loader, test_loader, cbc_stats = make_loaders(
            data_cfg, balanced_train=True
        )
    else:
        train_loader, val_loader, test_loader = loaders
        if cbc_stats is None:
            cbc_stats = ensure_cbc_stats()

    # smoke
    batch0 = next(iter(train_loader))
    x0 = batch0["image"].to(device).to(memory_format=torch.channels_last)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(
                x0,
                report_texts=batch0["report_text"] if grounding_on else None,
                return_grounding=grounding_on,
            )
        torch.cuda.synchronize()
        log.info("Smoke OK batch=%d", batch_size)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        raise RuntimeError(f"{spec.tag}: OOM at batch={batch_size}") from e
    finally:
        del x0, batch0
        torch.cuda.empty_cache()

    if pos_w is None:
        pos_w = feature_pos_weight(device)
    criterion_anemia = nn.CrossEntropyLoss()
    criterion_bg = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.param_groups(args.backbone_lr, args.new_lr),
        weight_decay=args.weight_decay,
    )
    warmup = LinearLR(
        optimizer, start_factor=1.0 / max(args.warmup_epochs, 1), total_iters=args.warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs - args.warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    best_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    ckpt_path = out_dir / "best.pt"
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            batch_gpu = {
                "anemia": batch["anemia"].to(device, non_blocking=True),
                "background_3": batch["background_3"].to(device, non_blocking=True),
                "features": batch["features"].to(device, non_blocking=True),
                "cbc_value": batch["cbc_value"].to(device, non_blocking=True),
                "cbc_mask": batch["cbc_mask"].to(device, non_blocking=True),
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    x,
                    report_texts=batch["report_text"] if grounding_on else None,
                    return_grounding=grounding_on,
                )
                loss, parts = compute_loss(
                    out,
                    batch_gpu,
                    pos_weight=pos_w,
                    use_cbc=cbc_on,
                    use_grounding=grounding_on,
                    use_background=use_bg,
                    use_features=use_feat,
                    criterion_anemia=criterion_anemia,
                    criterion_bg=criterion_bg,
                )
                loss = loss / grad_accum
            loss.backward()
            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(parts["loss_total"]) * x.size(0)
            n += x.size(0)
        if len(train_loader) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_loss = running / max(n, 1)
        val_m = eval_anemia_for_earlystop(model, val_loader, device)
        hist = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(hist)
        log.info(
            "[%s] ep %03d/%d loss=%.4f val_mF1=%.4f acc=%.4f rec[h]=%.3f rec[a]=%.3f",
            spec.tag, epoch, args.max_epochs, train_loss,
            val_m["macro_f1"], val_m["accuracy"],
            val_m["recall_healthy"], val_m["recall_anemic"],
        )
        if epoch > 10 and (val_m["recall_healthy"] < 0.15 or val_m["recall_anemic"] < 0.15):
            log.warning(
                "!!! COLLAPSE WARNING [%s] ep %d rec_h=%.3f rec_a=%.3f !!!",
                spec.tag, epoch, val_m["recall_healthy"], val_m["recall_anemic"],
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
                    "config": config,
                    "model_cfg": asdict(model_cfg),
                    "tag": spec.tag,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                log.info("[%s] early stop ep=%d best=%d mF1=%.4f",
                         spec.tag, epoch, best_epoch, best_macro_f1)
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last)

    results = eval_full(
        model, val_loader, test_loader, device, cbc_stats,
        use_cbc=cbc_on, use_background=use_bg, use_features=use_feat,
    )
    results["tag"] = spec.tag
    results["best_epoch"] = best_epoch
    results["best_val_macro_f1"] = best_macro_f1
    results["seconds"] = time.time() - t0
    results["grounding"] = grounding_on
    results["cbc_head"] = cbc_on
    results["use_transformer"] = spec.use_transformer
    results["seed"] = seed

    test_y = results.pop("test_y")
    test_prob = results.pop("test_prob")
    test_stems = results.pop("test_stems", None)
    save_kw = {"y": test_y, "prob": test_prob}
    if test_stems is not None:
        save_kw["stems"] = test_stems
    save_kw["pred"] = (test_prob >= results["anemia"]["threshold"]).astype(np.int64)
    np.savez_compressed(out_dir / "test_scores.npz", **save_kw)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    a = results["anemia"]
    log.info(
        "[%s] TEST auroc=%.4f acc@thr=%.4f thr=%.3f bg_mF1=%s feat_mAP=%s "
        "HGB_mae=%s MCV_mae=%s pred_sha1=%s",
        spec.tag, a["auroc"], a["acc_at_tuned_thr"], a["threshold"],
        results["background_3"]["macro_f1"],
        results["features"]["macro_mAP"],
        (results["cbc_mae_original"].get("HGB") or {}).get("mae"),
        (results["cbc_mae_original"].get("MCV") or {}).get("mae"),
        a["pred_sha1"],
    )

    results["test_y"] = test_y
    results["test_prob"] = test_prob
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    return results


def reeval_existing_full(args, device) -> dict | None:
    """Reuse legacy runs/method/best.pt as 'full' with threshold tuning."""
    legacy = METHOD_DIR / "best.pt"
    if not legacy.is_file():
        return None
    out_dir = METHOD_DIR / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "best.pt"
    if not dest.exists():
        shutil.copy2(legacy, dest)
    for name in ("config.json", "history.json"):
        src = METHOD_DIR / name
        if src.is_file() and not (out_dir / name).exists():
            shutil.copy2(src, out_dir / name)

    log.info("Re-evaluating legacy full ckpt with threshold tuning → %s", out_dir)
    set_seed(SEED)
    spec = AblationSpec("full")
    model_cfg = HemaFormerConfig(
        backbone=args.backbone,
        n_layers=args.n_layers,
        cbc_head=True,
        grounding=True,
        use_transformer=True,
        use_background=True,
        use_features=True,
    )
    model = HemaFormer(model_cfg)
    ckpt = torch.load(dest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device=device, memory_format=torch.channels_last)

    data_cfg = DataConfig(
        long_side=args.long_side,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_ram=args.cache_ram,
    )
    _, val_loader, test_loader, _ = make_loaders(data_cfg, balanced_train=False)
    cbc_stats = ensure_cbc_stats()
    results = eval_full(
        model, val_loader, test_loader, device, cbc_stats,
        use_cbc=True, use_background=True, use_features=True,
    )
    results["tag"] = "full"
    results["best_epoch"] = ckpt.get("epoch")
    results["best_val_macro_f1"] = ckpt.get("val_macro_f1")
    results["seconds"] = None
    results["grounding"] = True
    results["cbc_head"] = True
    results["use_transformer"] = True
    results["reused_legacy_ckpt"] = True
    results.pop("test_y", None)
    results.pop("test_prob", None)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # write config snapshot
    cfg = {
        "tag": "full",
        "reused_legacy_ckpt": True,
        "ablation": asdict(spec),
        "long_side": args.long_side,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "packages": collect_versions(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    del model, val_loader, test_loader
    torch.cuda.empty_cache()
    return results


def results_to_row(results: dict) -> dict:
    a = results["anemia"]
    ap = results["features"].get("ap_per_class") or {}
    cbc = results.get("cbc_mae_original") or {}
    return {
        "tag": results["tag"],
        "anemia_auroc": a["auroc"],
        "anemia_acc_tuned": a["acc_at_tuned_thr"],
        "anemia_thr": a["threshold"],
        "background_macro_f1": results["background_3"]["macro_f1"],
        "feature_macro_mAP": results["features"]["macro_mAP"],
        "ap_elliptocytes": ap.get("elliptocytes"),
        "ap_target_cells": ap.get("target_cells"),
        "ap_anisocytosis": ap.get("anisocytosis"),
        "cbc_HGB_mae": (cbc.get("HGB") or {}).get("mae"),
        "cbc_MCV_mae": (cbc.get("MCV") or {}).get("mae"),
        "best_epoch": results.get("best_epoch"),
        "pred_sha1": a["pred_sha1"],
        "grounding": results.get("grounding"),
        "cbc_head": results.get("cbc_head"),
        "use_transformer": results.get("use_transformer"),
    }


def print_ablation_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    # reference row
    ref = {
        "tag": "convnext@896 (ref)",
        "anemia_auroc": CONVNEXT_896_AUROC,
        "anemia_acc_tuned": CONVNEXT_896_ACC,
        "anemia_thr": None,
        "background_macro_f1": None,
        "feature_macro_mAP": None,
        "ap_elliptocytes": None,
        "ap_target_cells": None,
        "ap_anisocytosis": None,
        "cbc_HGB_mae": None,
        "cbc_MCV_mae": None,
        "best_epoch": None,
        "pred_sha1": "—",
    }
    df = pd.concat([df, pd.DataFrame([ref])], ignore_index=True)
    df = df.sort_values("anemia_auroc", ascending=False, na_position="last")
    cols = [
        "tag", "anemia_auroc", "anemia_acc_tuned", "anemia_thr",
        "background_macro_f1", "feature_macro_mAP",
        "ap_elliptocytes", "ap_target_cells", "ap_anisocytosis",
        "cbc_HGB_mae", "cbc_MCV_mae", "best_epoch", "pred_sha1",
    ]
    print("\n" + "=" * 120)
    print("ABLATION TABLE (sorted by anemia AUROC)")
    print("=" * 120)
    show = df[cols].copy()
    for c in show.columns:
        if c in ("tag", "pred_sha1"):
            continue
        show[c] = show[c].apply(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (
            f"{v:.4f}" if isinstance(v, float) else v
        ))
    show["pred_sha1"] = show["pred_sha1"].astype(str).str[:12]
    print(show.to_string(index=False))
    print(f"\nHeadline anchor: best baseline AUROC ≈ {BEST_BASELINE_AUROC:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", action="store_true", help="Run full ablation grid")
    p.add_argument("--tag", type=str, default=None, help="Single ablation tag")
    p.add_argument("--reuse-full", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grounding", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cbc_head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--backbone", type=str, default="convnext_tiny")
    p.add_argument("--long-side", type=int, default=896)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--new-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-ram", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(METHOD_DIR / "ablation_train.log", mode="a"),
        ],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (set CUDA_VISIBLE_DEVICES to pick a GPU)")
    device = torch.device("cuda:0")
    log.info("Device %s (%s)", device, torch.cuda.get_device_name(0))

    if args.ablation:
        specs = list(ABLATIONS)
    elif args.tag:
        match = [s for s in ABLATIONS if s.tag == args.tag]
        if not match:
            raise SystemExit(f"Unknown tag {args.tag}; choose from {[s.tag for s in ABLATIONS]}")
        specs = match
    else:
        # default single full-like from CLI flags
        specs = [
            AblationSpec(
                tag="full",
                grounding=args.grounding,
                cbc_head=args.cbc_head,
            )
        ]

    all_rows = []
    for spec in specs:
        out_dir = METHOD_DIR / spec.tag
        if (
            spec.tag == "full"
            and args.reuse_full
            and (METHOD_DIR / "best.pt").is_file()
            and not (out_dir / "results.json").is_file()
        ):
            results = reeval_existing_full(args, device)
            if results is not None:
                all_rows.append(results_to_row(results))
                continue
        if (out_dir / "results.json").is_file() and args.ablation:
            log.info("Skip existing %s", out_dir / "results.json")
            results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
            # ensure threshold fields exist; if legacy without tuned thr, re-eval
            if "acc_at_tuned_thr" not in results.get("anemia", {}):
                log.info("Legacy results lack tuned thr — will retrain/reeval %s", spec.tag)
            else:
                all_rows.append(results_to_row(results))
                continue
        results = train_one(spec, args, device, out_dir)
        all_rows.append(results_to_row(results))

    df = pd.DataFrame(all_rows)
    csv_path = METHOD_DIR / "ablation.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)
    print_ablation_table(all_rows)


if __name__ == "__main__":
    main()
