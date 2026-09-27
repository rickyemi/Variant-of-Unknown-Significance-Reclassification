"""Data loading, validation and splitting."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data.make_dataset import load_raw, split, validate


def test_raw_file_shape_and_classes():
    df = load_raw()
    assert df.shape == (3410, 30)
    share = df[cfg.TARGET_COL].value_counts(normalize=True).round(2).to_dict()
    assert share == {"Benign": 0.70, "Pathogenic": 0.20, "VUS": 0.10}


def test_none_category_is_not_missing():
    """'None' is a valid Alcohol_Use level and must not be read as NaN."""
    df = load_raw()
    assert "None" in set(df["Alcohol_Use"].dropna())
    assert abs(df.isna().mean().mean() - 0.13) < 0.005


def test_validate_rejects_bad_target():
    df = load_raw().head(50).copy()
    df.loc[0, cfg.TARGET_COL] = "Likely_Pathogenic"
    with pytest.raises(ValueError, match="Unexpected target"):
        validate(df)


def test_validate_rejects_out_of_range_score():
    df = load_raw().head(50).copy()
    df.loc[0, "REVEL_Score"] = 1.7
    with pytest.raises(ValueError, match="REVEL_Score"):
        validate(df)


def test_split_is_stratified_and_disjoint():
    train, test, vus = split(validate(load_raw()))
    assert len(train) + len(test) + len(vus) == 3410
    assert set(train[cfg.ID_COL]).isdisjoint(test[cfg.ID_COL])
    assert (vus[cfg.TARGET_COL] == "VUS").all()
    assert "VUS" not in set(train[cfg.TARGET_COL]) | set(test[cfg.TARGET_COL])
    p_tr = (train[cfg.TARGET_COL] == "Pathogenic").mean()
    p_te = (test[cfg.TARGET_COL] == "Pathogenic").mean()
    assert abs(p_tr - p_te) < 0.01
