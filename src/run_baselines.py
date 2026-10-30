"""Baseline anemia binary classifiers on our AneRBC-I split.

The reported runs used CUDA_VISIBLE_DEVICES=1, bf16 AMP + channels_last, seed=42.
Integrity guard: per-run isolation + sha1 of test logits/predictions;
scratch runs must not accidentally load ImageNet weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# GPU selection is left to the caller (the reported runs used CUDA_VISIBLE_DEVICES=1).

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
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
    FEATURE_COLS,
    RUNS,
    SEED,
    DataConfig,
    make_loaders,
    summarize_folds,
)

MODELS = ["vgg16", "mobilenetv2_100", "resnet152"]

PAPER_METRICS = {
    "vgg16": {"accuracy": 0.91, "precision": 0.91, "recall": 0.96, "f1": 0.93},
    "mobilenetv2_100": {"accuracy": 0.83, "precision": 0.95, "recall": 0.68, "f1": 0.78},
    "resnet152": {"accuracy": 0.77, "precision": 0.70, "recall": 0.95, "f1": 0.86},
}

CSV_COLS = [
    "model",
    "init",
    "long_side",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "pred_sha1",
    "logits_sha1",
    "pretrained",
    "macro_f1",
    "best_epoch",
    "val_macro_f1",
    "seconds",
    "ckpt",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_tag(model_name: str, pretrained: bool, long_side: int) -> str:
    init = "pretrained" if pretrained else "scratch"
    return f"{model_name}_{init}_ls{long_side}"


def ckpt_dir(model_name: str, pretrained: bool, long_side: int) -> Path:
    """Unique per (model, init, long_side)."""
    return RUNS / "baselines" / run_tag(model_name, pretrained, long_side)


def build_model(name: str, pretrained: bool, num_classes: int = 2) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def first_conv_param(model: nn.Module) -> tuple[str, torch.Tensor]:
    for name, p in model.named_parameters():
        if p.ndim == 4:
            return name, p.detach()
    raise RuntimeError("No 4D conv weight found for integrity check")


def first_conv_stat(model: nn.Module) -> dict:
    name, w = first_conv_param(model)
    wf = w.float().cpu()
    return {
        "name": name,
        "mean_abs": float(wf.abs().mean()),
        "std": float(wf.std()),
        "sha1": hashlib.sha1(wf.numpy().tobytes()).hexdigest()[:16],
    }


def assert_scratch_not_pretrained(model: nn.Module, model_name: str) -> None:
    """Scratch backbone first-conv weight must differ from a freshly pretrained load."""
    ref = build_model(model_name, pretrained=True, num_classes=2)
    try:
        n_m, w_m = first_conv_param(model)
        n_r, w_r = first_conv_param(ref)
        assert n_m == n_r, f"first-conv name mismatch: {n_m} vs {n_r}"
        # Exact equality would mean ImageNet weights were loaded into a "scratch" run.
        same = torch.allclose(w_m.float().cpu(), w_r.float().cpu(), atol=0.0, rtol=0.0)
        if same:
            raise AssertionError(
                f"INTEGRITY FAIL: scratch {model_name} first-conv '{n_m}' is identical "
                f"to a freshly pretrained load (accidental ImageNet init)."
            )
        # Soft check: mean|w| / std should also differ meaningfully
        sm, sr = first_conv_stat(model), first_conv_stat(ref)
        print(
            f"  scratch-guard OK: first_conv={n_m} "
            f"scratch(mean_abs={sm['mean_abs']:.5f},std={sm['std']:.5f},sha1={sm['sha1']}) "
            f"vs pretrained(mean_abs={sr['mean_abs']:.5f},std={sr['std']:.5f},sha1={sr['sha1']})"
        )
    finally:
        del ref
        gc.collect()


def sha1_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """Return metrics + raw logits/preds + sha1 hashes. Fresh accumulators every call."""
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        y = batch["anemia"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        logits_f = logits.float()
        p = torch.softmax(logits_f, dim=1)[:, 1]
        pred = logits_f.argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy().astype(np.int64))
        probs.append(p.detach().cpu().numpy().astype(np.float64))
        logits_list.append(logits_f.detach().cpu().numpy().astype(np.float32))

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    logits_all = np.concatenate(logits_list)

    # Per-class recall (collapse guard): index 0=healthy, 1=anemic
    rec_pc = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_healthy": float(rec_pc[0]),
        "recall_anemic": float(rec_pc[1]),
        "logits": logits_all,
        "preds": y_pred,
        "logits_sha1": sha1_array(logits_all.astype(np.float32)),
        "pred_sha1": sha1_array(y_pred.astype(np.int64)),
    }
    out["y_true"] = y_true
    out["y_prob"] = y_prob
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def hard_reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def train_one(
    model_name: str,
    pretrained: bool,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    long_side: int,
    num_workers: int,
    cache_ram: bool,
    pred_hash_registry: dict[str, dict],
) -> dict | None:
    """Train one model in isolation. Returns row dict, or None if OOM before progress."""
    hard_reset_cuda()
    set_seed(SEED)

    init = "pretrained" if pretrained else "scratch"
    tag = run_tag(model_name, pretrained, long_side)
    out_dir = ckpt_dir(model_name, pretrained, long_side)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    # Remove stale checkpoint so we never accidentally reload another run's weights.
    if ckpt_path.exists():
        ckpt_path.unlink()

    config = {
        "model": model_name,
        "pretrained": pretrained,
        "init": init,
        "seed": SEED,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "long_side": long_side,
        "amp": "bf16",
        "memory_format": "channels_last",
        "optimizer": "AdamW",
        "early_stop_metric": "val_macro_f1",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "tag": tag,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Fresh model first (weights fixed before any integrity side-effects).
    model = build_model(model_name, pretrained=pretrained)
    if not pretrained:
        assert_scratch_not_pretrained(model, model_name)
        # Re-seed after the pretrained reference load so DataLoader shuffle / dropout
        # RNG is not contaminated by the integrity check.
        set_seed(SEED)

    data_cfg = DataConfig(
        long_side=long_side,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_ram=cache_ram,
    )
    train_loader, val_loader, test_loader, _ = make_loaders(data_cfg)

    model = model.to(device=device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    use_cuda = device.type == "cuda"

    # Reset metric accumulators explicitly per run.
    best_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    print(
        f"\n===== Training {tag} on {device} "
        f"(bf16 + channels_last, long_side={long_side}, batch={batch_size}) ====="
    )
    t0 = time.time()
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            n = 0
            for batch in train_loader:
                x = batch["image"].to(device, non_blocking=True)
                x = x.to(memory_format=torch.channels_last)
                y = batch["anemia"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    logits = model(x)
                    loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * x.size(0)
                n += x.size(0)
            scheduler.step()
            train_loss = running / max(n, 1)

            # Fresh val accumulators inside evaluate()
            val_metrics = evaluate(model, val_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{
                        f"val_{k}": v
                        for k, v in val_metrics.items()
                        if k not in ("logits", "preds")
                    },
                }
            )
            print(
                f"  epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            if val_metrics["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "pretrained": pretrained,
                        "epoch": epoch,
                        "val_macro_f1": best_macro_f1,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config,
                    },
                    ckpt_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    print(
                        f"  early stop at epoch {epoch} "
                        f"(best epoch={best_epoch}, macro_f1={best_macro_f1:.4f})"
                    )
                    break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM during training {tag} — dropping this config.")
            hard_reset_cuda()
            del model, optimizer, scheduler
            hard_reset_cuda()
            return None
        raise

    if not ckpt_path.is_file():
        print(f"  No checkpoint saved for {tag}; skipping.")
        del model, optimizer, scheduler
        hard_reset_cuda()
        return None

    # Reload best weights into a fresh metric pass (isolation from last-epoch state).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last)

    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    logits_sha1 = test_metrics["logits_sha1"]
    pred_sha1 = test_metrics["pred_sha1"]
    print(
        f"  TEST {tag}: acc={test_metrics['accuracy']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f} AUROC={test_metrics['auroc']:.4f} "
        f"(best_epoch={best_epoch}, {elapsed/60:.1f} min)"
    )
    print(f"  integrity logits_sha1={logits_sha1}")
    print(f"  integrity pred_sha1  ={pred_sha1}")

    # Assert uniqueness of thresholded predictions across models in this session / table.
    if pred_sha1 in pred_hash_registry:
        other = pred_hash_registry[pred_sha1]
        dump = {
            "current": config,
            "duplicate_of": other.get("config"),
            "pred_sha1": pred_sha1,
            "logits_sha1_current": logits_sha1,
            "logits_sha1_other": other.get("logits_sha1"),
        }
        dump_path = RUNS / "baselines" / f"INTEGRITY_FAIL_{tag}.json"
        dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        raise AssertionError(
            f"INTEGRITY FAIL: identical prediction hash between {tag} and "
            f"{other.get('tag')} (pred_sha1={pred_sha1}). "
            f"Configs dumped to {dump_path}"
        )
    pred_hash_registry[pred_sha1] = {
        "tag": tag,
        "config": config,
        "logits_sha1": logits_sha1,
    }

    row = {
        "model": model_name,
        "init": init,
        "long_side": int(long_side),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auroc": test_metrics["auroc"],
        "pred_sha1": pred_sha1,
        "logits_sha1": logits_sha1,
        "pretrained": bool(pretrained),
        "macro_f1": test_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "seconds": elapsed,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
    }

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(
        json.dumps({k: v for k, v in row.items()}, indent=2),
        encoding="utf-8",
    )

    # Tear down before next run.
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    hard_reset_cuda()
    return row


def train_hires_recipe(
    model_name: str = "convnext_tiny",
    *,
    device: torch.device,
    long_side: int = 896,
    batch_size: int = 8,
    grad_accum: int = 3,
    peak_lr: float = 5e-5,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    min_epochs: int = 15,
    patience: int = 12,
    max_epochs: int = 60,
    num_workers: int = 8,
    cache_ram: bool = True,
    pred_hash_registry: dict[str, dict] | None = None,
    seed: int = SEED,
    out_dir: Path | None = None,
    skip_if_exists: bool = False,
) -> dict | None:
    """Robust high-res recipe (method validation): balanced sampler, warmup+cosine,
    grad-accum, collapse guard, delayed early-stop.
    """
    if pred_hash_registry is None:
        pred_hash_registry = {}

    hard_reset_cuda()
    set_seed(seed)

    pretrained = True
    init = "pretrained"
    tag = run_tag(model_name, pretrained, long_side)
    if out_dir is None:
        out_dir = ckpt_dir(model_name, pretrained, long_side)
        if seed != SEED:
            out_dir = out_dir.parent / f"{out_dir.name}_seed{seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "test_metrics.json"
    if skip_if_exists and metrics_path.is_file() and ckpt_path.is_file():
        print(f"  Skip existing hires {out_dir.name}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if ckpt_path.exists() and not skip_if_exists:
        ckpt_path.unlink()

    eff_batch = batch_size * grad_accum
    config = {
        "model": model_name,
        "pretrained": pretrained,
        "init": init,
        "recipe": "hires_robust",
        "seed": seed,
        "long_side": long_side,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": eff_batch,
        "peak_lr": peak_lr,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "max_epochs": max_epochs,
        "sampler": "WeightedRandomSampler",
        "amp": "bf16",
        "memory_format": "channels_last",
        "optimizer": "AdamW",
        "scheduler": "linear_warmup_cosine",
        "early_stop_metric": "val_macro_f1",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "tag": tag,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = build_model(model_name, pretrained=True)
    set_seed(seed)  # deterministic after weight download side-effects

    data_cfg = DataConfig(
        long_side=long_side,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_ram=cache_ram,
        drop_last_train=True,
    )
    train_loader, val_loader, test_loader, _ = make_loaders(
        data_cfg, balanced_train=True
    )

    model = model.to(device=device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    # Linear warmup over first `warmup_epochs`, then cosine over the rest.
    # start_factor: epoch-1 lr ≈ peak/warmup_epochs → peak after warmup.
    warmup = LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    use_cuda = device.type == "cuda"

    best_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    print(
        f"\n===== HIRES {tag} seed={seed} on {device} =====\n"
        f"  out={out_dir.name} long_side={long_side} batch={batch_size} accum={grad_accum} "
        f"eff={eff_batch} peak_lr={peak_lr} warmup={warmup_epochs} "
        f"min_epochs={min_epochs} patience={patience} max={max_epochs}\n"
        f"  sampler=WeightedRandomSampler  amp=bf16  channels_last"
    )
    t0 = time.time()
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            n = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                x = batch["image"].to(device, non_blocking=True)
                x = x.to(memory_format=torch.channels_last)
                y = batch["anemia"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    logits = model(x)
                    loss = criterion(logits, y) / grad_accum
                loss.backward()
                if step % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running += float(loss.item()) * grad_accum * x.size(0)
                n += x.size(0)
            # Flush leftover grads if last incomplete accum window
            if len(train_loader) % grad_accum != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            train_loss = running / max(n, 1)

            val_metrics = evaluate(model, val_loader, device)
            rh = val_metrics["recall_healthy"]
            ra = val_metrics["recall_anemic"]
            hist_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": cur_lr,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
                "val_recall_healthy": rh,
                "val_recall_anemic": ra,
            }
            history.append(hist_row)

            print(
                f"  epoch {epoch:03d}/{max_epochs}  loss={train_loss:.4f}  lr={cur_lr:.2e}  "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
                f"val_rec[healthy]={rh:.3f} val_rec[anemic]={ra:.3f}"
            )

            # COLLAPSE GUARD
            if epoch > 10 and (rh < 0.15 or ra < 0.15):
                print(
                    "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"  !!! COLLAPSE WARNING @ epoch {epoch}: "
                    f"val recall healthy={rh:.3f} anemic={ra:.3f} "
                    f"(threshold 0.15) — model may be trivial\n"
                    "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

            if val_metrics["macro_f1"] > best_macro_f1 + 1e-6:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "pretrained": pretrained,
                        "epoch": epoch,
                        "val_macro_f1": best_macro_f1,
                        "state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "config": config,
                    },
                    ckpt_path,
                )
            else:
                stale += 1
                # Never early-stop before min_epochs; require patience after that.
                if epoch >= min_epochs and stale >= patience:
                    print(
                        f"  early stop at epoch {epoch} "
                        f"(best epoch={best_epoch}, macro_f1={best_macro_f1:.4f}; "
                        f"min_epochs={min_epochs}, patience={patience})"
                    )
                    break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM during hires training {tag} — aborting.")
            hard_reset_cuda()
            return None
        raise

    if not ckpt_path.is_file():
        print(f"  No checkpoint for {tag}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    y_true = test_metrics.pop("y_true")
    y_prob = test_metrics.pop("y_prob")
    # drop bulky arrays from metrics dict before hashing / JSON
    test_metrics.pop("logits", None)
    test_metrics.pop("preds", None)
    np.savez_compressed(out_dir / "test_scores.npz", y=y_true, prob=y_prob)

    logits_sha1 = test_metrics["logits_sha1"]
    pred_sha1 = test_metrics["pred_sha1"]
    print(
        f"  TEST {tag}: acc={test_metrics['accuracy']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f} AUROC={test_metrics['auroc']:.4f} "
        f"(best_epoch={best_epoch}, {elapsed/60:.1f} min)"
    )
    print(f"  integrity logits_sha1={logits_sha1}")
    print(f"  integrity pred_sha1  ={pred_sha1}")

    if pred_sha1 in pred_hash_registry:
        other = pred_hash_registry[pred_sha1]
        dump_path = RUNS / "baselines" / f"INTEGRITY_FAIL_{tag}.json"
        dump_path.write_text(
            json.dumps(
                {"current": config, "duplicate_of": other.get("config"), "pred_sha1": pred_sha1},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        raise AssertionError(
            f"INTEGRITY FAIL: identical pred_sha1 between {tag} and {other.get('tag')}"
        )
    pred_hash_registry[pred_sha1] = {
        "tag": tag,
        "config": config,
        "logits_sha1": logits_sha1,
    }

    row = {
        "model": model_name,
        "init": init,
        "long_side": int(long_side),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auroc": test_metrics["auroc"],
        "pred_sha1": pred_sha1,
        "logits_sha1": logits_sha1,
        "pretrained": True,
        "macro_f1": test_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "seconds": elapsed,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
        "recipe": "hires_robust",
        "seed": seed,
    }

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (out_dir / "val_macro_f1_curve.json").write_text(
        json.dumps(
            [{"epoch": h["epoch"], "val_macro_f1": h["val_macro_f1"],
              "val_recall_healthy": h["val_recall_healthy"],
              "val_recall_anemic": h["val_recall_anemic"],
              "lr": h["lr"]} for h in history],
            indent=2,
        ),
        encoding="utf-8",
    )

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    hard_reset_cuda()
    return row


@torch.no_grad()
def eval_existing_ckpt(
    model_name: str,
    pretrained: bool,
    long_side: int,
    ckpt_path: Path,
    device: torch.device,
    num_workers: int,
    cache_ram: bool,
    pred_hash_registry: dict[str, dict],
) -> dict:
    """Recompute test metrics + hashes from an existing checkpoint (no training)."""
    hard_reset_cuda()
    init = "pretrained" if pretrained else "scratch"
    tag = run_tag(model_name, pretrained, long_side)

    data_cfg = DataConfig(
        long_side=long_side,
        batch_size=16 if long_side <= 512 else 8,
        num_workers=num_workers,
        cache_ram=cache_ram,
    )
    _, _, test_loader, _ = make_loaders(data_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    if not pretrained:
        # Still verify saved scratch weights ≠ ImageNet init
        assert_scratch_not_pretrained(model, model_name)
    model = model.to(device=device, memory_format=torch.channels_last)

    test_metrics = evaluate(model, test_loader, device)
    pred_sha1 = test_metrics["pred_sha1"]
    logits_sha1 = test_metrics["logits_sha1"]
    print(
        f"  EVAL {tag} from {ckpt_path}: "
        f"acc={test_metrics['accuracy']:.4f} pred_sha1={pred_sha1} logits_sha1={logits_sha1}"
    )

    config = {
        "model": model_name,
        "pretrained": pretrained,
        "init": init,
        "long_side": long_side,
        "tag": tag,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
        "source": "existing_ckpt_eval",
    }
    if pred_sha1 in pred_hash_registry:
        other = pred_hash_registry[pred_sha1]
        dump = {"current": config, "duplicate_of": other.get("config"), "pred_sha1": pred_sha1}
        dump_path = RUNS / "baselines" / f"INTEGRITY_FAIL_{tag}.json"
        dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        raise AssertionError(
            f"INTEGRITY FAIL: identical prediction hash between {tag} and "
            f"{other.get('tag')} (pred_sha1={pred_sha1}). Dumped {dump_path}"
        )
    pred_hash_registry[pred_sha1] = {
        "tag": tag,
        "config": config,
        "logits_sha1": logits_sha1,
    }

    row = {
        "model": model_name,
        "init": init,
        "long_side": int(long_side),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auroc": test_metrics["auroc"],
        "pred_sha1": pred_sha1,
        "logits_sha1": logits_sha1,
        "pretrained": bool(pretrained),
        "macro_f1": test_metrics["macro_f1"],
        "best_epoch": ckpt.get("epoch"),
        "val_macro_f1": ckpt.get("val_macro_f1"),
        "seconds": None,
        "ckpt": str(ckpt_path.relative_to(ROOT)),
    }
    del model, test_loader
    hard_reset_cuda()
    return row


def find_legacy_ckpt(model_name: str, pretrained: bool) -> Path | None:
    """Locate pre-guard ckpt paths (without _ls{N} suffix)."""
    init = "pretrained" if pretrained else "scratch"
    candidates = [
        RUNS / "baselines" / f"{model_name}_{init}" / "best.pt",
        RUNS / "baselines" / f"{model_name}_{init}_ls512" / "best.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def collect_package_versions() -> dict[str, str | None]:
    import importlib.metadata as md

    names = [
        "torch", "torchvision", "timm", "transformers", "tokenizers", "einops",
        "pandas", "numpy", "pillow", "opencv-python-headless", "scikit-learn",
        "albumentations", "tqdm", "matplotlib", "tensorboard",
    ]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def write_global_config(args: argparse.Namespace, device: torch.device, jobs: list) -> Path:
    cfg = {
        "env": "anerbc",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False,
        "amp": "bf16",
        "memory_format": "channels_last",
        "seed": SEED,
        "jobs": jobs,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "lr_scratch": args.lr_scratch,
        "lr_pretrained": args.lr_pretrained,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "cache_ram": args.cache_ram,
        "packages": collect_package_versions(),
    }
    path = RUNS / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def print_tables(df: pd.DataFrame) -> None:
    show = df.copy()
    show["pred_sha1_short"] = show["pred_sha1"].astype(str).str[:12]
    cols = ["model", "init", "long_side", "accuracy", "precision", "recall", "f1", "auroc", "pred_sha1"]
    print("\n" + "=" * 100)
    print("FULL BASELINE TABLE")
    print("=" * 100)
    print(show[cols].to_string(index=False))

    print("\nBaseline table — reported(paper) vs ours(our split)")
    print("(Splits differ; do NOT claim equivalence.)\n")
    header = (
        f"{'model':16s} {'init':12s} {'ls':>4s} {'source':18s} "
        f"{'acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'AUROC':>7s}"
    )
    print(header)
    print("-" * len(header))
    for model_name, paper in PAPER_METRICS.items():
        print(
            f"{model_name:16s} {'scratch':12s} {'512':>4s} {'reported(paper)':18s} "
            f"{paper['accuracy']:6.2f} {paper['precision']:6.2f} {paper['recall']:6.2f} "
            f"{paper['f1']:6.2f} {'—':>7s}"
        )
        sub = df[(df.model == model_name) & (df.long_side == 512)]
        for _, ours in sub.sort_values("init").iterrows():
            print(
                f"{model_name:16s} {ours['init']:12s} {int(ours['long_side']):>4d} "
                f"{'ours(our split)':18s} "
                f"{ours['accuracy']:6.4f} {ours['precision']:6.4f} {ours['recall']:6.4f} "
                f"{ours['f1']:6.4f} {ours['auroc']:7.4f}"
            )
    # ConvNeXt (no paper numbers)
    sub = df[df.model == "convnext_tiny"].sort_values(["long_side", "init"])
    for _, ours in sub.iterrows():
        print(
            f"{'convnext_tiny':16s} {ours['init']:12s} {int(ours['long_side']):>4d} "
            f"{'ours(our split)':18s} "
            f"{ours['accuracy']:6.4f} {ours['precision']:6.4f} {ours['recall']:6.4f} "
            f"{ours['f1']:6.4f} {ours['auroc']:7.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr-scratch", type=float, default=1e-3)
    parser.add_argument("--lr-pretrained", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache-ram", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["full", "fix_extend", "convnext_hires"],
        default="fix_extend",
        help="full | fix_extend | convnext_hires (retrain convnext@896 with robust recipe)",
    )
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (set CUDA_VISIBLE_DEVICES to pick a GPU)")
    device = torch.device("cuda:0")
    print(f"Device: {device}  name={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")

    pred_hash_registry: dict[str, dict] = {}
    rows: list[dict] = []

    if args.mode == "convnext_hires":
        # Merge-overwrite only the convnext@896 row; keep the rest of baselines.csv.
        csv_path = RUNS / "baselines.csv"
        if csv_path.is_file():
            existing = pd.read_csv(csv_path)
            keep = existing[
                ~((existing.model == "convnext_tiny") & (existing.long_side == 896))
            ].copy()
            for _, r in keep.iterrows():
                pred_hash_registry[str(r["pred_sha1"])] = {
                    "tag": f"{r['model']}_{r['init']}_ls{int(r['long_side'])}",
                    "config": r.to_dict(),
                    "logits_sha1": r.get("logits_sha1"),
                }
        else:
            keep = pd.DataFrame()

        write_global_config(
            args,
            device,
            [{"model": "convnext_tiny", "long_side": 896, "action": "hires_robust"}],
        )
        row = train_hires_recipe(
            "convnext_tiny",
            device=device,
            long_side=896,
            batch_size=8,
            grad_accum=3,
            peak_lr=5e-5,
            weight_decay=args.weight_decay,
            warmup_epochs=5,
            min_epochs=15,
            patience=12,
            max_epochs=60,
            num_workers=args.num_workers,
            cache_ram=args.cache_ram,
            pred_hash_registry=pred_hash_registry,
        )
        if row is None:
            raise RuntimeError("convnext hires training failed (OOM or no ckpt)")

        df = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values(["model", "long_side", "init"]).reset_index(drop=True)
        ordered = [c for c in CSV_COLS if c in df.columns]
        # preserve recipe col if present
        extra = [c for c in df.columns if c not in ordered]
        df[ordered + extra].to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path} ({len(df)} rows) — overwritten convnext_tiny@896")

        print("\n=== Retrained convnext_tiny@896 row ===")
        print(pd.DataFrame([row])[
            ["model", "init", "long_side", "accuracy", "precision", "recall",
             "f1", "auroc", "pred_sha1", "best_epoch", "val_macro_f1"]
        ].to_string(index=False))

        curve_path = ckpt_dir("convnext_tiny", True, 896) / "val_macro_f1_curve.json"
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        print("\n=== Val macro-F1 curve ===")
        print(f"{'ep':>4s} {'macroF1':>8s} {'rec_h':>7s} {'rec_a':>7s} {'lr':>10s}")
        for h in curve:
            print(
                f"{h['epoch']:4d} {h['val_macro_f1']:8.4f} "
                f"{h['val_recall_healthy']:7.3f} {h['val_recall_anemic']:7.3f} "
                f"{h['lr']:10.2e}"
            )
        print_tables(df)
        return

    if args.mode == "fix_extend":
        jobs = [
            {"model": "vgg16", "pretrained": False, "long_side": 512, "action": "eval_existing"},
            {"model": "vgg16", "pretrained": True, "long_side": 512, "action": "eval_existing"},
            {"model": "mobilenetv2_100", "pretrained": False, "long_side": 512, "action": "eval_existing"},
            {"model": "mobilenetv2_100", "pretrained": True, "long_side": 512, "action": "eval_existing"},
            {"model": "resnet152", "pretrained": False, "long_side": 512, "action": "train", "batch": 16},
            {"model": "resnet152", "pretrained": True, "long_side": 512, "action": "train", "batch": 16},
            {"model": "convnext_tiny", "pretrained": True, "long_side": 512, "action": "train", "batch": 16},
            {"model": "convnext_tiny", "pretrained": True, "long_side": 896, "action": "train", "batch": 8},
        ]
    else:
        jobs = []
        for m in MODELS:
            for pt in (False, True):
                jobs.append(
                    {"model": m, "pretrained": pt, "long_side": 512, "action": "train", "batch": 16}
                )
        jobs.append(
            {"model": "convnext_tiny", "pretrained": True, "long_side": 512, "action": "train", "batch": 16}
        )
        jobs.append(
            {"model": "convnext_tiny", "pretrained": True, "long_side": 896, "action": "train", "batch": 8}
        )

    write_global_config(args, device, jobs)
    print(f"Jobs ({len(jobs)}):")
    for j in jobs:
        print(f"  {j}")

    for job in jobs:
        model_name = job["model"]
        pretrained = bool(job["pretrained"])
        long_side = int(job["long_side"])
        action = job["action"]

        if action == "eval_existing":
            path = find_legacy_ckpt(model_name, pretrained)
            if path is None:
                raise FileNotFoundError(
                    f"Missing checkpoint for {model_name} pretrained={pretrained}"
                )
            row = eval_existing_ckpt(
                model_name,
                pretrained,
                long_side,
                path,
                device,
                num_workers=args.num_workers,
                cache_ram=args.cache_ram,
                pred_hash_registry=pred_hash_registry,
            )
            # Enrich legacy test_metrics with hashes
            out_dir = path.parent
            (out_dir / "test_metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            rows.append(row)
            continue

        batch = int(job.get("batch", 16 if long_side <= 512 else 8))
        lr = args.lr_pretrained if pretrained else args.lr_scratch
        row = train_one(
            model_name,
            pretrained,
            device=device,
            epochs=args.epochs,
            batch_size=batch,
            lr=lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            long_side=long_side,
            num_workers=args.num_workers,
            cache_ram=args.cache_ram,
            pred_hash_registry=pred_hash_registry,
        )
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    # Stable sort
    df = df.sort_values(["model", "long_side", "init"]).reset_index(drop=True)
    out_csv = RUNS / "baselines.csv"
    ordered = [c for c in CSV_COLS if c in df.columns]
    df[ordered].to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")

    # Final cross-check: unique pred hashes
    if df["pred_sha1"].duplicated().any():
        dups = df[df["pred_sha1"].duplicated(keep=False)]
        raise AssertionError(f"Duplicate pred_sha1 in final table:\n{dups}")

    summary = summarize_folds()
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("Split sizes:", summary["split_sizes"])
    print_tables(df)


if __name__ == "__main__":
    main()
