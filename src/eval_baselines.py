"""Evaluate baseline CNN checkpoints on our fixed AneRBC-I split.

- Reads artifacts/manifest.csv + split.csv via src.data
- Computes accuracy/precision/recall/F1/AUC on test
- Prints SUMMARY and saves runs/baselines.csv

If resnet152_pretrained checkpoint is missing, this script optionally trains it
(with small epochs/patience) to complete the table.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import RUNS, SEED, make_loaders, summarize_folds

MODELS = ["vgg16", "mobilenetv2_100", "resnet152"]

PAPER_METRICS = {
    # (model, pretrained) -> dict
    ("vgg16", False): {"accuracy": 0.91, "precision": 0.91, "recall": 0.96, "f1": 0.93},
    ("mobilenetv2_100", False): {"accuracy": 0.83, "precision": 0.95, "recall": 0.68, "f1": 0.78},
    ("resnet152", False): {"accuracy": 0.77, "precision": 0.70, "recall": 0.95, "f1": 0.86},
    ("vgg16", True): {"accuracy": 0.86, "precision": 0.88, "recall": 0.78, "f1": 0.92},
    ("mobilenetv2_100", True): {"accuracy": 0.71, "precision": 0.73, "recall": 0.52, "f1": 0.60},
    ("resnet152", True): {"accuracy": 0.70, "precision": 0.69, "recall": 0.86, "f1": 0.77},
}


@dataclass
class ResultRow:
    model: str
    pretrained: bool
    source: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float | None
    auc: float | None


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["anemia"].to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
        probs.append(p.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def default_batch_size(model_name: str) -> int:
    # Keep it small for ResNet152
    if model_name in ("vgg16", "resnet152"):
        return 4 if model_name == "vgg16" else 2
    return 8


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    obj = torch.load(ckpt_path, map_location="cpu")
    model_name = obj["model_name"]
    pretrained = bool(obj["pretrained"])
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=2)
    state_dict = obj["state_dict"]
    # Handle DataParallel checkpoints (\"module.\" prefix)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, obj


def train_resnet152_pretrained_if_missing(image_size: int = 1024) -> Path | None:
    ckpt_dir = RUNS / "checkpoints"
    ckpt_path = ckpt_dir / "resnet152_pretrained.pt"
    if ckpt_path.is_file():
        return ckpt_path

    if os.environ.get("SKIP_MISSING_RESNET152_PRETRAINED", "0") == "1":
        return None

    print("[WARN] Missing resnet152_pretrained checkpoint; training a small baseline to fill the table...")

    # Hyperparams (small)
    epochs = int(os.environ.get("RESNET152_PRETRAINED_EPOCHS", "15"))
    patience = int(os.environ.get("RESNET152_PRETRAINED_PATIENCE", "4"))
    lr = float(os.environ.get("RESNET152_PRETRAINED_LR", "1e-4"))
    weight_decay = float(os.environ.get("RESNET152_PRETRAINED_WD", "1e-4"))
    max_train_batches = int(os.environ.get("RESNET152_PRETRAINED_MAX_TRAIN_BATCHES", "0"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    train_loader, val_loader, test_loader, _ = make_loaders(
        batch_size=default_batch_size("resnet152"),
        num_workers=2,
        long_side=image_size,
    )

    model = timm.create_model("resnet152", pretrained=True, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1 = -1.0
    best_epoch = -1
    stale = 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["anemia"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
            if max_train_batches > 0 and (i + 1) >= max_train_batches:
                break

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        print(f"  epoch {epoch:03d}/{epochs} loss={running/max(n,1):.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_name": "resnet152",
                    "pretrained": True,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "state_dict": model.state_dict(),
                    "config": {
                        "epochs": epochs,
                        "patience": patience,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "image_size": image_size,
                    },
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch} (best epoch={best_epoch}, macro_f1={best_macro_f1:.4f})")
                break

    return ckpt_path if ckpt_path.is_file() else None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_dir = RUNS / "checkpoints"

    resnet_pretrained_ckpt = train_resnet152_pretrained_if_missing(image_size=1024)

    specs = []
    # scratch
    specs.append(("mobilenetv2_100", False, ckpt_dir / "mobilenetv2_100_scratch.pt"))
    specs.append(("vgg16", False, ckpt_dir / "vgg16_scratch.pt"))
    specs.append(("resnet152", False, ckpt_dir / "resnet152_scratch.pt"))
    # pretrained
    specs.append(("mobilenetv2_100", True, ckpt_dir / "mobilenetv2_100_pretrained.pt"))
    specs.append(("vgg16", True, ckpt_dir / "vgg16_pretrained.pt"))
    specs.append(("resnet152", True, resnet_pretrained_ckpt))

    results = []
    for model_name, pretrained, ckpt_path in specs:
        if ckpt_path is None or not ckpt_path.is_file():
            print(f"[WARN] Missing ckpt: {ckpt_path}; skipping {model_name} pretrained={pretrained}")
            continue

        # Data loaders (test only for speed; make_loaders also computes stats if missing)
        _, _, test_loader, _ = make_loaders(
            batch_size=default_batch_size(model_name),
            num_workers=2,
            image_size=1024,
        )

        model, obj = load_checkpoint(ckpt_path, device)
        metrics = evaluate(model, test_loader, device)

        results.append(
            ResultRow(
                model=model_name,
                pretrained=pretrained,
                source="ours (our split)",
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                macro_f1=metrics["macro_f1"],
                auc=metrics["auc"],
            )
        )

    # Save baselines.csv
    df = pd.DataFrame(
        [r.__dict__ for r in results]
    )
    # Rename
    if len(df) == 0:
        raise RuntimeError("No baseline results produced.")

    out_df = pd.DataFrame(
        {
            "model": df["model"],
            "pretrained": df["pretrained"],
            "source": df["source"],
            "accuracy": df["accuracy"],
            "precision": df["precision"],
            "recall": df["recall"],
            "f1": df["f1"],
            "auc": df["auc"],
        }
    )
    out_path = RUNS / "baselines.csv"
    out_df.to_csv(out_path, index=False)
    print("Saved", out_path)

    # Paper vs ours side-by-side for each model
    print("\n=== Paper vs ours (AneRBC-I) ===")
    for model_name in ["vgg16", "mobilenetv2_100", "resnet152"]:
        for pretrained in [False, True]:
            paper = PAPER_METRICS[(model_name, pretrained)]
            label = "ImageNet-pretrained" if pretrained else "from-scratch"
            ours_sel = out_df[(out_df.model == model_name) & (out_df.pretrained == pretrained)]
            print(f"\n  {model_name} | {label}")
            print(
                f"    reported (paper):  acc={paper['accuracy']:.2f}  P={paper['precision']:.2f}  R={paper['recall']:.2f}  F1={paper['f1']:.2f}"
            )
            if len(ours_sel) == 0:
                print("    ours (our split):  MISSING (checkpoint not available / not trained)")
            else:
                ours = ours_sel.iloc[0]
                print(
                    f"    ours (our split):  acc={ours['accuracy']:.4f}  P={ours['precision']:.4f}  R={ours['recall']:.4f}  F1={ours['f1']:.4f}  AUC={ours['auc']:.4f}"
                )

    # Summary
    summary = summarize_folds()
    print("\n================ SUMMARY ================")
    print("Split sizes:", summary["split_sizes"])
    print("\nbackground_3 counts per split:")
    print(pd.DataFrame(summary["background_3"]).to_string())
    print("\nFeature positive counts per split:")
    print(pd.DataFrame(summary["features"]).T.to_string())
    print("\nCBC valid/masked per field (by split):")
    for split in ("train", "val", "test"):
        print(f"  [{split}]")
        for f, d in summary["cbc"][split].items():
            print(f"    {f:8s} valid={d['valid']:4d} masked={d['masked']:4d}")


if __name__ == "__main__":
    main()
