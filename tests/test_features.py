"""Preprocessing transformers and feature selection."""
import numpy as np
import pandas as pd

from src import config as cfg
from src.data.make_dataset import get_xy
from src.features.build_features import (IQRCapper, Log10Transformer, build_preprocessor,
                                         select_features, split_types)


def test_iqr_capper_clips_to_training_fences():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0], [np.nan]])
    cap = IQRCapper().fit(X)
    out = cap.transform(X)
    assert out[4, 0] == cap.upper_[0]           # extreme value capped
    assert np.isnan(out[5, 0])                  # NaN left for the imputer
    assert np.allclose(out[:4, 0], X[:4, 0])    # normal values untouched


def test_log10_transformer_only_touches_named_columns():
    X = pd.DataFrame({"gnomAD_AF": [1e-4, 1e-2], "Age": [40, 60]})
    out = Log10Transformer(["gnomAD_AF"]).fit(X).transform(X)
    assert np.allclose(out["gnomAD_AF"], [-4, -2])
    assert out["Age"].tolist() == [40, 60]


def test_preprocessor_outputs_complete_zscored_matrix(splits):
    feats = cfg.ALL_FEATURES
    num, cat = split_types(feats)
    X, _ = get_xy(splits["train"], feats)
    prep = build_preprocessor(num, cat).fit(X)
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    assert not Xt.isna().any().any()
    assert np.allclose(Xt[num].mean(), 0, atol=1e-6)
    assert np.allclose(Xt[num].std(ddof=0), 1, atol=1e-3)


def test_preprocessor_handles_unseen_category(splits):
    num, cat = split_types(cfg.ALL_FEATURES)
    X, _ = get_xy(splits["train"], cfg.ALL_FEATURES)
    prep = build_preprocessor(num, cat).fit(X)
    Xn = X.head(3).copy()
    Xn["Gene"] = "NEW_GENE"
    assert np.isfinite(prep.transform(Xn)).all()


def test_feature_selection_drops_pure_noise(splits):
    sel = select_features(splits["train"])["selected"]
    noise = {"BMI", "Sex", "Ethnicity", "Read_Depth", "PD_L1_TPS", "Cancer_Type", "Alcohol_Use"}
    assert noise.isdisjoint(sel)
    for strong in ["REVEL_Score", "Variant_Type", "gnomAD_AF", "CADD_Phred"]:
        assert strong in sel
