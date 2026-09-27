"""
Stage 1: load, validate and split the raw variant file.

Steps
-----
1. Read the raw CSV. Only empty cells count as missing, so the valid
   Alcohol_Use category "None" is not silently turned into NaN.
2. Validate the schema: required columns, unique IDs, allowed target levels,
   numeric ranges for bounded scores.
3. Write the validated file to data/interim/.
4. Separate the unlabelled VUS rows (to be reclassified at the end) from the
   labelled Pathogenic/Benign rows, then split the labelled rows 80/20 into
   train and test, stratified by class. Written to data/processed/.

Run:  python -m src.data.make_dataset
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as cfg
from src.utils import get_logger

log = get_logger(__name__)

# Bounded scores: values outside these ranges indicate a corrupt file.
VALID_RANGES = {
    "REVEL_Score": (0, 1), "SIFT_Score": (0, 1), "PolyPhen2_Score": (0, 1),
    "SpliceAI_Score": (0, 1), "gnomAD_AF": (0, 1), "Variant_Allele_Freq": (0, 1),
    "Age": (0, 120), "Ki67_Index": (0, 100), "PD_L1_TPS": (0, 100),
}


def load_raw(path=cfg.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV, treating only blank cells as missing."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    log.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise a ValueError if the file does not match the expected schema."""
    required = [cfg.ID_COL, cfg.TARGET_COL] + cfg.ALL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[cfg.ID_COL].duplicated().any():
        raise ValueError("Duplicate Variant_ID values found")
    if df[cfg.TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values")
    bad = set(df[cfg.TARGET_COL].unique()) - set(cfg.TARGET_LEVELS)
    if bad:
        raise ValueError(f"Unexpected target levels: {bad}")
    for col, (lo, hi) in VALID_RANGES.items():
        s = df[col].dropna()
        if ((s < lo) | (s > hi)).any():
            raise ValueError(f"{col} has values outside [{lo}, {hi}]")
    # Make sure numeric columns really are numeric.
    for col in cfg.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    log.info("Schema validation passed. Class counts: %s",
             df[cfg.TARGET_COL].value_counts().to_dict())
    log.info("Overall missing cells: %.1f%%", 100 * df.isna().mean().mean())
    return df


def split(df: pd.DataFrame):
    """Return (train, test, vus) data frames."""
    vus = df[df[cfg.TARGET_COL] == cfg.UNLABELLED_CLASS].reset_index(drop=True)
    labelled = df[df[cfg.TARGET_COL] != cfg.UNLABELLED_CLASS]
    train, test = train_test_split(
        labelled, test_size=cfg.TEST_SIZE, stratify=labelled[cfg.TARGET_COL],
        random_state=cfg.RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True), vus


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split ('train', 'test' or 'vus') with the right NA rules."""
    path = {"train": cfg.TRAIN_FILE, "test": cfg.TEST_FILE, "vus": cfg.VUS_FILE}[name]
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def get_xy(df: pd.DataFrame, features=None):
    """Split a frame into X (features) and y (1 = Pathogenic, 0 = Benign)."""
    features = features or cfg.ALL_FEATURES
    X = df[features].copy()
    y = (df[cfg.TARGET_COL] == cfg.POSITIVE_CLASS).astype(int)
    return X, y


def main(raw_path=cfg.RAW_FILE):
    cfg.ensure_dirs()
    df = validate(load_raw(raw_path))
    df.to_csv(cfg.INTERIM_FILE, index=False)
    train, test, vus = split(df)
    train.to_csv(cfg.TRAIN_FILE, index=False)
    test.to_csv(cfg.TEST_FILE, index=False)
    vus.to_csv(cfg.VUS_FILE, index=False)
    log.info("Train %d | Test %d | VUS %d rows written to %s",
             len(train), len(test), len(vus), cfg.PROCESSED_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--raw", default=str(cfg.RAW_FILE), help="path to raw CSV")
    from pathlib import Path
    main(Path(p.parse_args().raw))
