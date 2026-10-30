#!/usr/bin/env python3
"""Build supervised morphology/CBC labels + seeded stratified split for AneRBC-I.

Read-only on AneRBC_dataset/. Writes only under ./artifacts/.
Requires: pandas + stdlib. Does NOT train anything.

Run from the repository root:

    python -m src.build_anerbc_i_labels
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "AneRBC_dataset" / "AneRBC-I"
ART = ROOT / "artifacts"
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

FEATURE_COLS = [
    "elliptocytes",
    "target_cells",
    "tear_drop_cells",
    "polychromasia",
    "anisocytosis",
    "anisopoikilocytosis",
    "basophilic_stippling",
    "schistocytes",
    "spherocytes",
    "nRBCs",
    "macrocytosis",
]

# Lexicon with variants / misspellings (matched as whole phrases, case-insensitive).
FEATURE_PATTERNS: dict[str, list[str]] = {
    "elliptocytes": [r"elliptocytes?", r"ovalocytes?"],
    "target_cells": [r"target\s*cells?"],
    "tear_drop_cells": [
        r"tear[\s\-]?drop\s*cells?",
        r"teardrop\s*cells?",
        r"dacrocytes?",
    ],
    "polychromasia": [r"polychromasia", r"polycromasia", r"polychromatophilia"],
    "anisocytosis": [r"anisocytosis"],
    "anisopoikilocytosis": [r"anisopoikilocytosis"],
    "basophilic_stippling": [r"basophilic\s+stippling"],
    "schistocytes": [r"schistocytes?", r"fragmented\s+rbcs?"],
    "spherocytes": [r"spherocytes?", r"sphrerocytes?"],
    "nRBCs": [r"nrbcs?", r"nucleated\s+rbcs?"],
    "macrocytosis": [r"macrocytosis", r"macro[\s\-]?ovalocytosis", r"macrocytic"],
}

CBC_FIELDS = ["HGB", "HCT", "MCV", "MCH", "MCHC", "RDW_CV", "RBC"]
# Map raw first-column names -> canonical
CBC_NAME_MAP = {
    "HGB": "HGB",
    "HB": "HGB",
    "HEMOGLOBIN": "HGB",
    "HCT": "HCT",
    "HEMATOCRIT": "HCT",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "RDW---CV": "RDW_CV",
    "RDW--CV": "RDW_CV",
    "RDW-CV": "RDW_CV",
    "RDWCV": "RDW_CV",
    "RDW": "RDW_CV",
    "%RDW---CV": "RDW_CV",
    "%RDW-CV": "RDW_CV",
    "%RDW": "RDW_CV",
    "RBC": "RBC",
}

# Unit tokens we accept / normalize away before float coercion
UNIT_TOKEN_RE = re.compile(
    r"""
    (?:
        x10\.e\s*[36]\s*/\s*[μµu]?l
      | x10\.e
      | g/d[lL]
      | [fFiI]?[lL]          # fL / FL / fl / L / IL typos for MCV/MPV
      | pg
      | %
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def list_stems() -> list[tuple[str, str]]:
    """Return sorted list of (stem, cohort) for all original images."""
    if not DATASET.is_dir():
        raise SystemExit(
            f"AneRBC-I not found at {DATASET}\n"
            "Place the public AneRBC dataset at ./AneRBC_dataset (see docs/DATASET.md)."
        )
    rows = []
    for cohort, flag in (
        ("Healthy_individuals", "healthy"),
        ("Anemic_individuals", "anemic"),
    ):
        d = DATASET / cohort / "Original_images"
        for p in sorted(d.glob("*.png")):
            rows.append((p.stem, cohort))
    if not rows:
        raise SystemExit(f"No *.png under {DATASET}/*/Original_images")
    return rows


def extract_smear_clause(text: str) -> str:
    """Pull the peripheral-smear description; stop before WBC/platelet sections."""
    # Normalize whitespace for matching, but work on a flattened copy
    flat = re.sub(r"[\r\n]+", " ", text)
    flat = re.sub(r"\s+", " ", flat)

    m = re.search(
        r"Peripheral\s+smear\s+shows\s+(.+?)(?="
        r"\s*WBCs?\s+MORPHOLOGY|"
        r"\s*PLATELETS|"
        r"\s*NOTE\s*:|"
        r"$)",
        flat,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(" .;")

    # Fallback: any "shows ..." before WBC section
    m = re.search(
        r"shows\s+(.+?)(?=\s*WBCs?\s+MORPHOLOGY|\s*PLATELETS|\s*NOTE\s*:|$)",
        flat,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(" .;")
    return ""


def classify_background(clause: str) -> str:
    c = clause.lower()
    # Order matters: more specific / priority first
    if re.search(r"\bdimorphic\b", c):
        return "dimorphic"
    if re.search(r"normocytic\s+normochromic", c):
        return "normocytic_normochromic"
    if re.search(r"microcytic\s+hypochromic", c):
        return "microcytic_hypochromic"
    # macrocytic family (picture or -cytosis / ovalocytosis)
    if re.search(r"\bmacrocytic\b|\bmacrocytosis\b|macro[\s\-]?ovalocytosis", c):
        return "macrocytic"
    return "other"


def feature_flags(clause: str, full_text: str) -> dict[str, int]:
    """Multi-label flags; search clause + full report text."""
    hay_l = f"{clause}\n{full_text}".lower()
    out = {}
    for feat, pats in FEATURE_PATTERNS.items():
        hit = 0
        for pat in pats:
            bounded = rf"(?<![a-z])(?:{pat})(?![a-z])"
            if re.search(bounded, hay_l, flags=re.IGNORECASE):
                hit = 1
                break
        out[feat] = hit
    return out


def parse_morphology() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for stem, cohort in list_stems():
        path = DATASET / cohort / "Morphology_reports" / f"{stem}.txt"
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        clause = extract_smear_clause(text) if text else ""
        bg = classify_background(clause) if clause else "other"
        flags = feature_flags(clause, text) if text else {f: 0 for f in FEATURE_COLS}
        # If no report file, still emit row with zeros / other
        if not path.is_file():
            bg = "other"
        rows.append({"stem": stem, "background": bg, **flags})

    df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)

    # Frequency over feature columns
    freq_rows = []
    for feat in FEATURE_COLS:
        freq_rows.append({"term": feat, "count": int(df[feat].sum())})
    # Also count backgrounds
    for bg, cnt in df["background"].value_counts().items():
        freq_rows.append({"term": f"background:{bg}", "count": int(cnt)})
    freq = pd.DataFrame(freq_rows)

    return df, freq


def normalize_cbc_value(raw: str) -> tuple[float | None, str | None]:
    """Strip *, units, coerce to float. Return (value, drop_reason_or_None)."""
    if raw is None:
        return None, "empty"
    s = str(raw).strip()
    if not s:
        return None, "empty"

    # Remove asterisks / flag markers
    s = s.replace("*", " ").strip()
    # Normalize micro-sign variants already in string; collapse spaces
    s = s.replace("µ", "μ").replace("uL", "μl").replace("ul", "μl")
    s = re.sub(r"\s+", " ", s)

    # Some broken cells look like ranges "0.02 --- 0.09 x10.e 3/μl" with no single value
    # Prefer the first number if a clear primary number exists before "---"
    # Drop if the cell is ONLY a range with no leading value pattern we trust.

    # Strip known unit tokens repeatedly
    cleaned = s
    cleaned = UNIT_TOKEN_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("---", " ").replace("--", " ")
    cleaned = re.sub(r"[^0-9.\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None, "no_numeric"

    # If multiple numbers remain (e.g. mangled range), take the first
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None, "no_numeric"
    try:
        val = float(nums[0])
    except ValueError:
        return None, "float_fail"

    # Sanity: reject absurd values for typical CBC scales (still keep; log as suspicious?)
    # User asked NaN+log failures only on parse failure — keep value.
    return val, None


def parse_cbc() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = []
    dropped = []
    stats_prep: dict[str, list] = {f: [] for f in CBC_FIELDS}

    for stem, cohort in list_stems():
        path = DATASET / cohort / "CBC_reports" / f"{stem}.txt"
        row = {f: float("nan") for f in CBC_FIELDS}
        row["stem"] = stem
        if not path.is_file():
            for f in CBC_FIELDS:
                dropped.append({"stem": stem, "field": f, "raw": "<MISSING_FILE>"})
            records.append(row)
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            for f in CBC_FIELDS:
                dropped.append({"stem": stem, "field": f, "raw": f"<READ_ERROR:{e}>"})
            records.append(row)
            continue

        seen = set()
        for line_no, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line or line.lower().startswith("test"):
                continue
            # Defensive CSV split — never crash
            try:
                parts = [p.strip() for p in line.split(",")]
            except Exception:
                dropped.append({"stem": stem, "field": "<line>", "raw": line})
                continue
            if len(parts) < 2:
                continue
            raw_name = parts[0].lstrip("*%").strip()
            key = CBC_NAME_MAP.get(raw_name.upper())
            if key is None:
                norm = re.sub(r"\s+", "", raw_name.upper())
                key = CBC_NAME_MAP.get(norm)
            if key is None and re.match(r"^%?RDW", raw_name, re.I):
                key = "RDW_CV"
            if key is None and re.match(r"^HEMOGLOB", raw_name, re.I):
                key = "HGB"
            if key is None or key not in CBC_FIELDS:
                continue
            if key in seen:
                continue  # first wins
            seen.add(key)

            raw_val = parts[1]
            val, reason = normalize_cbc_value(raw_val)
            if reason is not None:
                dropped.append({"stem": stem, "field": key, "raw": raw_val})
                continue
            row[key] = val
            stats_prep[key].append(val)

        # Fields never seen in file → not a drop (just missing); only log if we attempted
        records.append(row)

    df = pd.DataFrame(records)
    # column order
    df = df[["stem"] + CBC_FIELDS].sort_values("stem").reset_index(drop=True)
    drop_df = pd.DataFrame(dropped, columns=["stem", "field", "raw"])

    summary = {}
    for f in CBC_FIELDS:
        s = df[f]
        nn = int(s.notna().sum())
        dropped_n = int((drop_df["field"] == f).sum()) if len(drop_df) else 0
        if nn:
            summary[f] = {
                "non_null": nn,
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
                "dropped": dropped_n,
            }
        else:
            summary[f] = {
                "non_null": 0,
                "min": None,
                "median": None,
                "max": None,
                "dropped": dropped_n,
            }
    return df, drop_df, summary


def build_manifest(
    morph: pd.DataFrame, cbc: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    rows = []
    morph_i = morph.set_index("stem")
    cbc_i = cbc.set_index("stem")

    for stem, cohort in list_stems():
        anemia = "anemic" if cohort == "Anemic_individuals" else "healthy"
        base = DATASET / cohort
        img = base / "Original_images" / f"{stem}.png"
        binary = base / "Binary_segmented" / f"{stem}.png"
        rgb = base / "RGB_segmented" / f"{stem}.png"
        cbc_p = base / "CBC_reports" / f"{stem}.txt"
        morph_p = base / "Morphology_reports" / f"{stem}.txt"

        missing = []
        for label, p in (
            ("image", img),
            ("binary_gt", binary),
            ("rgb_gt", rgb),
            ("cbc", cbc_p),
            ("morphology", morph_p),
        ):
            if not p.is_file():
                missing.append(label)
        if missing:
            problems.append(f"{stem}: missing {', '.join(missing)}")

        mrow = morph_i.loc[stem] if stem in morph_i.index else None
        crow = cbc_i.loc[stem] if stem in cbc_i.index else None

        rec = {
            "stem": stem,
            "cohort": cohort,
            "anemia": anemia,
            "image_path": str(img.relative_to(ROOT)) if img.is_file() else "",
            "binary_gt_path": str(binary.relative_to(ROOT)) if binary.is_file() else "",
            "rgb_gt_path": str(rgb.relative_to(ROOT)) if rgb.is_file() else "",
            "cbc_path": str(cbc_p.relative_to(ROOT)) if cbc_p.is_file() else "",
            "morphology_path": str(morph_p.relative_to(ROOT)) if morph_p.is_file() else "",
            "background": mrow["background"] if mrow is not None else "other",
        }
        for feat in FEATURE_COLS:
            rec[feat] = int(mrow[feat]) if mrow is not None else 0
        for f in CBC_FIELDS:
            rec[f] = float(crow[f]) if crow is not None else float("nan")
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    return df, problems


def stratified_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Seeded 70/15/15 stratified by anemia. Sort stems before shuffling."""
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9
    parts = []
    for anemia_label, group in manifest.groupby("anemia", sort=True):
        stems = sorted(group["stem"].tolist())
        # deterministic shuffle via pandas/numpy-free: use Random
        import random

        rng = random.Random(SEED)
        stems = list(stems)
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
        # remainder to test to ensure sum == n
        n_test = n - n_train - n_val
        # Adjust if rounding pushed over
        if n_test < 0:
            n_train += n_test
            n_test = 0

        train_stems = stems[:n_train]
        val_stems = stems[n_train : n_train + n_val]
        test_stems = stems[n_train + n_val :]

        for s in train_stems:
            parts.append({"stem": s, "split": "train"})
        for s in val_stems:
            parts.append({"stem": s, "split": "val"})
        for s in test_stems:
            parts.append({"stem": s, "split": "test"})

    split_df = pd.DataFrame(parts).sort_values("stem").reset_index(drop=True)
    return split_df


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)

    print("=== A) MORPHOLOGY ===")
    morph, freq = parse_morphology()
    morph_path = ART / "labels_morphology.csv"
    freq_path = ART / "morph_freq.csv"
    morph.to_csv(morph_path, index=False)
    freq.to_csv(freq_path, index=False)
    print(f"Wrote {morph_path} ({len(morph)} rows)")
    print(f"Wrote {freq_path}")
    print("\nFeature frequencies:")
    rare = []
    for feat in FEATURE_COLS:
        c = int(morph[feat].sum())
        flag = "  *** <30 positives ***" if c < 30 else ""
        print(f"  {feat:24s} {c:4d}{flag}")
        if c < 30:
            rare.append(feat)
    print("\nBackground distribution:")
    print(morph["background"].value_counts().to_string())
    if rare:
        print(f"\nFeatures with <30 positives: {rare}")
    else:
        print("\nNo features below 30 positives.")

    print("\n=== B) CBC ===")
    cbc, dropped, summary = parse_cbc()
    cbc_path = ART / "labels_cbc.csv"
    drop_path = ART / "cbc_parse_dropped.csv"
    cbc.to_csv(cbc_path, index=False)
    dropped.to_csv(drop_path, index=False)
    print(f"Wrote {cbc_path} ({len(cbc)} rows)")
    print(f"Wrote {drop_path} ({len(dropped)} drop records)")
    print("\nPer-field stats:")
    print(f"{'field':10s} {'non_null':>8s} {'min':>10s} {'median':>10s} {'max':>10s} {'dropped':>8s}")
    for f in CBC_FIELDS:
        s = summary[f]
        mn = f"{s['min']:.4g}" if s["min"] is not None else "NA"
        md = f"{s['median']:.4g}" if s["median"] is not None else "NA"
        mx = f"{s['max']:.4g}" if s["max"] is not None else "NA"
        print(f"{f:10s} {s['non_null']:8d} {mn:>10s} {md:>10s} {mx:>10s} {s['dropped']:8d}")

    print("\n=== C) MANIFEST + SPLIT ===")
    manifest, problems = build_manifest(morph, cbc)
    man_path = ART / "manifest.csv"
    manifest.to_csv(man_path, index=False)
    print(f"Wrote {man_path} ({len(manifest)} rows)")

    # Assert completeness
    print("\nCompleteness check (binary GT + RGB GT + CBC + morphology):")
    if problems:
        print(f"  MISSING for {len(problems)} stems:")
        for p in problems[:50]:
            print(f"    {p}")
        if len(problems) > 50:
            print(f"    ... and {len(problems) - 50} more")
    else:
        print("  OK — every stem has image, binary GT, RGB GT, CBC, morphology.")

    split = stratified_split(manifest)
    split_path = ART / "split.csv"
    split.to_csv(split_path, index=False)
    print(f"Wrote {split_path} ({len(split)} rows)")

    # Class balance per split
    merged = manifest.merge(split, on="stem", how="inner")
    print("\nClass balance (anemia) per split:")
    bal = (
        merged.groupby(["split", "anemia"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"])
    )
    print(bal.to_string())
    print("\nBackground × split:")
    bg_bal = (
        merged.groupby(["split", "background"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"])
    )
    print(bg_bal.to_string())

    print("\n========== SUMMARY ==========")
    print(f"Stems: {len(manifest)} (expected 1000)")
    print(f"Anemia: {manifest['anemia'].value_counts().to_dict()}")
    print(f"Background: {manifest['background'].value_counts().to_dict()}")
    print(f"Rare morph features (<30): {rare}")
    print(
        f"Split seed={SEED}, ratios={TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC}, "
        f"stratified by anemia, stems sorted then shuffled."
    )
    print(
        "Split sizes: "
        + ", ".join(
            f"{sp}={int((split['split'] == sp).sum())}"
            for sp in ("train", "val", "test")
        )
    )
    print(f"CBC drop records: {len(dropped)}")
    print(f"Completeness problems: {len(problems)}")
    print(f"Artifacts dir: {ART}")


if __name__ == "__main__":
    main()
