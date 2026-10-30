"""AneRBC-I data module: dataset, CBC cleaning, transforms, loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUNS = ROOT / "runs"

# Cleaning ranges (all CBC fields); model tensors use CBC_MODEL_FIELDS only.
CBC_FIELDS = ["HGB", "HCT", "MCV", "MCH", "MCHC", "RDW_CV", "RBC"]
CBC_MODEL_FIELDS = ["HGB", "MCV", "MCH", "MCHC"]
CBC_RANGES = {
    "HGB": (3.0, 22.0),
    "HCT": (10.0, 65.0),
    "MCV": (50.0, 130.0),
    "MCH": (15.0, 45.0),
    "MCHC": (25.0, 40.0),
    "RDW_CV": (10.0, 40.0),
    "RBC": (1.5, 7.5),
}

FEATURE_COLS = [
    "elliptocytes",
    "anisocytosis",
    "polychromasia",
    "target_cells",
    "tear_drop_cells",
    "nRBCs",
    "schistocytes",
]

BACKGROUND_3_CLASSES = [
    "microcytic_hypochromic",
    "normocytic_normochromic",
    "other",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42


@dataclass
class DataConfig:
    long_side: int = 896
    batch_size: int = 8
    num_workers: int = 8
    cache_ram: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True
    seed: int = SEED


def map_background_3(bg: str) -> str:
    if bg in ("microcytic_hypochromic", "normocytic_normochromic"):
        return bg
    # dimorphic / macrocytic / existing-other → other
    return "other"


def load_joined_tables(
    manifest_path: Path | None = None,
    split_path: Path | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path or ARTIFACTS / "manifest.csv")
    split = pd.read_csv(split_path or ARTIFACTS / "split.csv")
    df = manifest.merge(split, on="stem", how="inner")
    if len(df) != len(manifest):
        missing = set(manifest["stem"]) - set(split["stem"])
        raise RuntimeError(f"split.csv missing stems: {sorted(missing)[:10]}")
    return df.sort_values("stem").reset_index(drop=True)


def load_manifest_only(manifest_path: Path | None = None) -> pd.DataFrame:
    """All AneRBC-I rows (no split column)."""
    return pd.read_csv(manifest_path or ARTIFACTS / "manifest.csv").sort_values("stem").reset_index(
        drop=True
    )


def load_folds_table(folds_path: Path | None = None) -> pd.DataFrame:
    path = folds_path or ARTIFACTS / "folds.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run build_folds_csv() first.")
    return pd.read_csv(path)


def build_folds_csv(
    *,
    n_folds: int = 5,
    val_frac: float = 0.15,
    seed: int = SEED,
    folds_path: Path | None = None,
) -> pd.DataFrame:
    """Stratified K-fold assignment + per-fold train/val/test roles → artifacts/folds.csv."""
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    manifest = load_manifest_only()
    labels = _anemia_labels_from_df(manifest)
    stems = manifest["stem"].to_numpy()

    fold_id = np.full(len(stems), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(skf.split(np.arange(len(stems)), labels)):
        fold_id[test_idx] = k
    if (fold_id < 0).any():
        raise RuntimeError("Incomplete fold assignment")

    rows: list[dict[str, Any]] = []
    for k in range(n_folds):
        test_mask = fold_id == k
        pool_mask = ~test_mask
        pool_idx = np.where(pool_mask)[0]
        pool_y = labels[pool_mask]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=seed + k
        )
        train_rel, val_rel = next(sss.split(pool_idx, pool_y))
        train_idx = pool_idx[train_rel]
        val_idx = pool_idx[val_rel]
        for i in test_mask.nonzero()[0]:
            rows.append({"stem": stems[i], "fold": int(k), "role": "test", "anemia": int(labels[i])})
        for i in train_idx:
            rows.append({"stem": stems[i], "fold": int(k), "role": "train", "anemia": int(labels[i])})
        for i in val_idx:
            rows.append({"stem": stems[i], "fold": int(k), "role": "val", "anemia": int(labels[i])})

    out = pd.DataFrame(rows).sort_values(["fold", "role", "stem"]).reset_index(drop=True)
    path = folds_path or ARTIFACTS / "folds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    # sanity: each stem appears once per fold block; 1000 * 5 rows
    expected = len(manifest) * n_folds
    if len(out) != expected:
        raise RuntimeError(f"folds.csv row count {len(out)} != expected {expected}")
    for k in range(n_folds):
        sub = out[out["fold"] == k]
        n_test = int((sub["role"] == "test").sum())
        n_train = int((sub["role"] == "train").sum())
        n_val = int((sub["role"] == "val").sum())
        if n_test + n_train + n_val != len(manifest):
            raise RuntimeError(f"fold {k} partition size mismatch")
    return out


def joined_for_fold(fold_id: int, folds_path: Path | None = None) -> pd.DataFrame:
    """Manifest + role (train/val/test) for one CV fold."""
    manifest = load_manifest_only()
    folds = load_folds_table(folds_path)
    sub = folds[folds["fold"] == fold_id][["stem", "role"]]
    df = manifest.merge(sub, on="stem", how="inner")
    if len(df) != len(manifest):
        raise RuntimeError(f"fold {fold_id}: merge lost rows")
    return df.sort_values("stem").reset_index(drop=True)


def compute_cbc_stats_from_df(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return compute_cbc_stats(train_df)


def clean_cbc_row(row: pd.Series) -> tuple[dict[str, float], dict[str, int]]:
    """Apply physiologic ranges; NaN + mask=0 if out of range."""
    values: dict[str, float] = {}
    masks: dict[str, int] = {}
    for f in CBC_FIELDS:
        lo, hi = CBC_RANGES[f]
        v = row.get(f, float("nan"))
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        if v != v or v < lo or v > hi:
            values[f] = float("nan")
            masks[f] = 0
        else:
            values[f] = v
            masks[f] = 1
    return values, masks


def compute_cbc_stats(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Train-split in-range mean/std per CBC field."""
    stats: dict[str, dict[str, float]] = {}
    for f in CBC_FIELDS:
        lo, hi = CBC_RANGES[f]
        s = pd.to_numeric(train_df[f], errors="coerce")
        in_range = s[(s >= lo) & (s <= hi)]
        if len(in_range) == 0:
            mean, std = 0.0, 1.0
        else:
            mean = float(in_range.mean())
            std = float(in_range.std(ddof=0))
            if std < 1e-8:
                std = 1.0
        stats[f] = {"mean": mean, "std": std, "n": int(len(in_range))}
    return stats


def save_cbc_stats(stats: dict[str, dict[str, float]], path: Path | None = None) -> Path:
    path = path or (RUNS / "cbc_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return path


def load_cbc_stats(path: Path | None = None) -> dict[str, dict[str, float]]:
    path = path or (RUNS / "cbc_stats.json")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def ensure_cbc_stats(df: pd.DataFrame | None = None, force: bool = False) -> dict[str, dict[str, float]]:
    path = RUNS / "cbc_stats.json"
    if path.is_file() and not force:
        return load_cbc_stats(path)
    df = df if df is not None else load_joined_tables()
    stats = compute_cbc_stats(df[df["split"] == "train"])
    save_cbc_stats(stats, path)
    return stats


def standardize_cbc(
    values: dict[str, float],
    masks: dict[str, int],
    stats: dict[str, dict[str, float]],
    fields: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (values_z, mask) for model CBC fields. Masked slots → 0.0."""
    fields = fields or CBC_MODEL_FIELDS
    z_list, m_list = [], []
    for f in fields:
        m = int(masks[f])
        m_list.append(float(m))
        if m == 0:
            z_list.append(0.0)
        else:
            z_list.append((values[f] - stats[f]["mean"]) / stats[f]["std"])
    return (
        torch.tensor(z_list, dtype=torch.float32),
        torch.tensor(m_list, dtype=torch.float32),
    )


class ResizeLongSidePad32:
    """Resize so long side == size, pad H/W to a multiple of 32 (keep aspect)."""

    def __init__(self, long_side: int = 896, fill: int = 0):
        self.long_side = long_side
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        long = max(w, h)
        scale = self.long_side / float(long)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        pad_w = (32 - new_w % 32) % 32
        pad_h = (32 - new_h % 32) % 32
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return TF.pad(img, [left, top, right, bottom], fill=self.fill)


def build_transforms(train: bool, long_side: int = 896) -> transforms.Compose:
    ops: list[Any] = [ResizeLongSidePad32(long_side)]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


class AneRBCDataset(Dataset):
    """Per-stem sample from manifest (+ split or explicit dataframe)."""

    def __init__(
        self,
        split: str | None = None,
        *,
        df: pd.DataFrame | None = None,
        role_col: str = "role",
        root: Path | None = None,
        cbc_stats: dict[str, dict[str, float]] | None = None,
        long_side: int = 896,
        transform: transforms.Compose | None = None,
        cache_ram: bool = False,
        image_cache: dict[str, np.ndarray] | None = None,
    ):
        self.root = root or ROOT
        if df is not None:
            self.df = df.reset_index(drop=True)
            self.split = str(split or "custom")
        else:
            if split is None:
                raise ValueError("Provide split= or df=")
            self.split = split
            joined = load_joined_tables()
            self.df = joined[joined["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split}")

        train_flag = False
        if df is not None and role_col in self.df.columns:
            roles = set(self.df[role_col].astype(str).unique())
            train_flag = roles == {"train"}
        elif split is not None:
            train_flag = split == "train"
        self.cbc_stats = cbc_stats if cbc_stats is not None else ensure_cbc_stats()
        self.transform = transform or build_transforms(train=train_flag, long_side=long_side)
        self.cache_ram = cache_ram
        self.image_cache = image_cache if image_cache is not None else {}
        if cache_ram and not self.image_cache:
            self._preload_images()

    def _preload_images(self) -> None:
        for i in range(len(self.df)):
            stem = str(self.df.iloc[i]["stem"])
            if stem in self.image_cache:
                continue
            path = self.root / str(self.df.iloc[i]["image_path"])
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            self.image_cache[stem] = arr

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row: pd.Series) -> Image.Image:
        stem = str(row["stem"])
        if stem in self.image_cache:
            return Image.fromarray(self.image_cache[stem], mode="RGB")
        path = self.root / str(row["image_path"])
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        stem = str(row["stem"])
        image = self._load_image(row)
        # CHW here; channels_last applied on NCHW in collate_anerbc
        image_t = self.transform(image)

        # anemia 0/1 from cohort (Anemic_individuals → 1)
        anemia = 1 if str(row["cohort"]) == "Anemic_individuals" else 0

        bg3 = map_background_3(str(row["background"]))
        bg_idx = BACKGROUND_3_CLASSES.index(bg3)
        feats = torch.tensor([float(row[c]) for c in FEATURE_COLS], dtype=torch.float32)

        values, masks = clean_cbc_row(row)
        cbc_value, cbc_mask = standardize_cbc(values, masks, self.cbc_stats)

        morph_path = self.root / str(row["morphology_path"])
        try:
            report_text = morph_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            report_text = ""

        return {
            "stem": stem,
            "image": image_t,
            "anemia": torch.tensor(anemia, dtype=torch.long),
            "background_3": torch.tensor(bg_idx, dtype=torch.long),
            "background_3_name": bg3,
            "features": feats,
            "cbc_value": cbc_value,
            "cbc_mask": cbc_mask,
            "cbc_raw": {f: values[f] for f in CBC_MODEL_FIELDS},
            "report_text": report_text,
        }


def collate_anerbc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    images = images.contiguous(memory_format=torch.channels_last)
    return {
        "stem": [b["stem"] for b in batch],
        "image": images,
        "anemia": torch.stack([b["anemia"] for b in batch], dim=0),
        "background_3": torch.stack([b["background_3"] for b in batch], dim=0),
        "background_3_name": [b["background_3_name"] for b in batch],
        "features": torch.stack([b["features"] for b in batch], dim=0),
        "cbc_value": torch.stack([b["cbc_value"] for b in batch], dim=0),
        "cbc_mask": torch.stack([b["cbc_mask"] for b in batch], dim=0),
        "cbc_raw": [b["cbc_raw"] for b in batch],
        "report_text": [b["report_text"] for b in batch],
    }


def _anemia_labels_from_df(df: pd.DataFrame) -> np.ndarray:
    """0=healthy, 1=anemic from cohort (fallback to anemia column)."""
    if "cohort" in df.columns:
        return (df["cohort"].astype(str) == "Anemic_individuals").astype(np.int64).to_numpy()
    return (df["anemia"].astype(str) == "anemic").astype(np.int64).to_numpy()


def make_balanced_sampler(ds: AneRBCDataset) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-frequency weights so each batch tends to see both classes."""
    labels = _anemia_labels_from_df(ds.df)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / counts
    weights = class_w[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def make_loaders(
    cfg: DataConfig | None = None,
    *,
    long_side: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_ram: bool | None = None,
    cbc_stats: dict[str, dict[str, float]] | None = None,
    balanced_train: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    cfg = cfg or DataConfig()
    if long_side is not None:
        cfg.long_side = long_side
    if batch_size is not None:
        cfg.batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
    if cache_ram is not None:
        cfg.cache_ram = cache_ram

    RUNS.mkdir(parents=True, exist_ok=True)
    df = load_joined_tables()
    if cbc_stats is None:
        cbc_stats = ensure_cbc_stats(df)

    shared_cache: dict[str, np.ndarray] = {}
    loaders: dict[str, DataLoader] = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = AneRBCDataset(
            split,
            cbc_stats=cbc_stats,
            long_side=cfg.long_side,
            cache_ram=cfg.cache_ram,
            image_cache=shared_cache if cfg.cache_ram else None,
        )
        if cfg.cache_ram:
            # ensure this split is cached (shared dict grows across splits)
            ds._preload_images()
            shared_cache = ds.image_cache

        nw = cfg.num_workers
        sampler = None
        use_shuffle = shuffle
        drop_last = cfg.drop_last_train and split == "train"
        if split == "train" and balanced_train:
            sampler = make_balanced_sampler(ds)
            use_shuffle = False  # mutually exclusive with sampler
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=use_shuffle,
            sampler=sampler,
            num_workers=nw,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            persistent_workers=(cfg.persistent_workers and nw > 0),
            drop_last=drop_last,
            collate_fn=collate_anerbc,
        )
    return loaders["train"], loaders["val"], loaders["test"], cbc_stats


def _make_loader_from_df(
    df: pd.DataFrame,
    *,
    cfg: DataConfig,
    cbc_stats: dict,
    balanced_train: bool,
    shuffle: bool,
    drop_last: bool,
    shared_cache: dict[str, np.ndarray] | None,
) -> DataLoader:
    ds = AneRBCDataset(
        df=df,
        cbc_stats=cbc_stats,
        long_side=cfg.long_side,
        cache_ram=cfg.cache_ram,
        image_cache=shared_cache if cfg.cache_ram else None,
    )
    if cfg.cache_ram:
        ds._preload_images()
        if shared_cache is not None:
            shared_cache.update(ds.image_cache)

    nw = cfg.num_workers
    sampler = None
    use_shuffle = shuffle
    if balanced_train:
        sampler = make_balanced_sampler(ds)
        use_shuffle = False
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        persistent_workers=(cfg.persistent_workers and nw > 0),
        drop_last=drop_last,
        collate_fn=collate_anerbc,
    )


def make_fold_loaders(
    fold_id: int,
    cfg: DataConfig | None = None,
    *,
    long_side: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_ram: bool | None = None,
    balanced_train: bool = True,
    folds_path: Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, pd.DataFrame]:
    """Train/val/test loaders for one CV fold; CBC stats from fold-train only."""
    cfg = cfg or DataConfig()
    if long_side is not None:
        cfg.long_side = long_side
    if batch_size is not None:
        cfg.batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
    if cache_ram is not None:
        cfg.cache_ram = cache_ram

    df = joined_for_fold(fold_id, folds_path)
    train_df = df[df["role"] == "train"].reset_index(drop=True)
    val_df = df[df["role"] == "val"].reset_index(drop=True)
    test_df = df[df["role"] == "test"].reset_index(drop=True)
    cbc_stats = compute_cbc_stats_from_df(train_df)

    shared_cache: dict[str, np.ndarray] = {}
    train_loader = _make_loader_from_df(
        train_df,
        cfg=cfg,
        cbc_stats=cbc_stats,
        balanced_train=balanced_train,
        shuffle=True,
        drop_last=cfg.drop_last_train,
        shared_cache=shared_cache,
    )
    val_loader = _make_loader_from_df(
        val_df,
        cfg=cfg,
        cbc_stats=cbc_stats,
        balanced_train=False,
        shuffle=False,
        drop_last=False,
        shared_cache=shared_cache if cfg.cache_ram else None,
    )
    test_loader = _make_loader_from_df(
        test_df,
        cfg=cfg,
        cbc_stats=cbc_stats,
        balanced_train=False,
        shuffle=False,
        drop_last=False,
        shared_cache=shared_cache if cfg.cache_ram else None,
    )
    return train_loader, val_loader, test_loader, cbc_stats, df


def feature_pos_weight_from_df(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    n = len(train_df)
    weights = []
    for c in FEATURE_COLS:
        pos = float(train_df[c].sum())
        weights.append((n - pos) / max(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def summarize_folds(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = df if df is not None else load_joined_tables()
    out: dict[str, Any] = {"split_sizes": df["split"].value_counts().to_dict()}

    df = df.copy()
    df["background_3"] = df["background"].map(map_background_3)
    bg_counts = (
        df.groupby(["split", "background_3"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"])
        .reindex(columns=BACKGROUND_3_CLASSES, fill_value=0)
    )
    out["background_3"] = bg_counts.to_dict()

    feat_counts = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        feat_counts[split] = {c: int(sub[c].sum()) for c in FEATURE_COLS}
    out["features"] = feat_counts

    cbc_valid: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        cbc_valid[split] = {}
        for f in CBC_FIELDS:
            lo, hi = CBC_RANGES[f]
            s = pd.to_numeric(sub[f], errors="coerce")
            valid = int(((s >= lo) & (s <= hi)).sum())
            masked = int(len(sub) - valid)
            cbc_valid[split][f] = {"valid": valid, "masked": masked}
    out["cbc"] = cbc_valid
    return out


def decode_label_row(sample: dict[str, Any]) -> dict[str, Any]:
    """Fully decoded human-readable label row (batch idx 0 or single sample)."""
    anemia_t = sample["anemia"]
    bg_t = sample["background_3"]
    feats = sample["features"]
    cbc_v = sample["cbc_value"]
    cbc_m = sample["cbc_mask"]

    if torch.is_tensor(anemia_t) and anemia_t.ndim > 0 and anemia_t.numel() > 1:
        anemia = int(anemia_t[0].item())
        bg_idx = int(bg_t[0].item())
        feats = feats[0]
        cbc_v = cbc_v[0]
        cbc_m = cbc_m[0]
        stem = sample["stem"][0]
        report = sample["report_text"][0]
        bg_name = sample.get("background_3_name", [BACKGROUND_3_CLASSES[bg_idx]])[0]
    elif torch.is_tensor(anemia_t):
        anemia = int(anemia_t.item())
        bg_idx = int(bg_t.item())
        stem = sample["stem"] if isinstance(sample["stem"], str) else sample["stem"][0]
        report = (
            sample["report_text"]
            if isinstance(sample["report_text"], str)
            else sample["report_text"][0]
        )
        bg_name = sample.get("background_3_name", BACKGROUND_3_CLASSES[bg_idx])
        if isinstance(bg_name, list):
            bg_name = bg_name[0]
    else:
        anemia = int(anemia_t)
        bg_idx = int(bg_t)
        stem = sample["stem"]
        report = sample["report_text"]
        bg_name = sample.get("background_3_name", BACKGROUND_3_CLASSES[bg_idx])

    feat_vals = {
        FEATURE_COLS[i]: int(v > 0.5) for i, v in enumerate(feats.tolist())
    }
    cbc_decoded = {
        f: {"z": float(cbc_v[i]), "mask": int(cbc_m[i])}
        for i, f in enumerate(CBC_MODEL_FIELDS)
    }
    return {
        "stem": stem,
        "anemia": anemia,
        "anemia_name": "anemic" if anemia == 1 else "healthy",
        "background_3": bg_name,
        "background_3_index": bg_idx,
        "features": feat_vals,
        "features_on": [k for k, v in feat_vals.items() if v],
        "cbc": cbc_decoded,
        "report_text_preview": report[:240].replace("\n", " "),
    }


def _print_batch_demo() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig(long_side=896, batch_size=8, num_workers=8, cache_ram=False)
    train_loader, val_loader, test_loader, stats = make_loaders(cfg)
    batch = next(iter(train_loader))
    print("=== One training batch (long_side=896, batch=8) ===")
    print(f"image:            {tuple(batch['image'].shape)}  dtype={batch['image'].dtype}  "
          f"channels_last={batch['image'].is_contiguous(memory_format=torch.channels_last)}")
    print(f"anemia:           {tuple(batch['anemia'].shape)}  values={batch['anemia'].tolist()}")
    print(f"background_3:     {tuple(batch['background_3'].shape)}  values={batch['background_3'].tolist()}")
    print(f"features:         {tuple(batch['features'].shape)}")
    print(f"cbc_value:        {tuple(batch['cbc_value'].shape)}  fields={CBC_MODEL_FIELDS}")
    print(f"cbc_mask:         {tuple(batch['cbc_mask'].shape)}")
    print(f"stems:            {batch['stem']}")
    print(f"report_text[0] len={len(batch['report_text'][0])}")
    print("\nFully decoded label row [0]:")
    decoded = decode_label_row(batch)
    for k, v in decoded.items():
        print(f"  {k}: {v}")

    print(f"\nCBC stats -> {RUNS / 'cbc_stats.json'}")
    print(json.dumps(stats, indent=2))

    summary = summarize_folds()
    print("\n=== CBC valid/masked counts per field per split ===")
    for split in ("train", "val", "test"):
        print(f"  [{split}] n={summary['split_sizes'][split]}")
        for f, d in summary["cbc"][split].items():
            print(f"    {f:8s}  valid={d['valid']:4d}  masked={d['masked']:4d}")


if __name__ == "__main__":
    _print_batch_demo()
