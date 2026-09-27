"""Saved models, metrics and VUS predictions."""
import numpy as np
import pandas as pd

from src import config as cfg
from src.data.make_dataset import get_xy
from src.models.evaluate_model import MODEL_ORDER, compute_metrics
from src.models.predict_model import confidence_tier, reclassify
from src.models.transformer_model import FTTransformerClassifier


def test_compute_metrics_on_known_confusion_matrix():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .2, .1, .1, .2, .3, .6, .4])   # TP3 FN1 FP1 TN5 at 0.5
    m = compute_metrics(y, p, 0.5)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (3, 1, 1, 5)
    assert np.isclose(m["Sensitivity"], 0.75) and np.isclose(m["Specificity"], 5 / 6)
    assert np.isclose(m["PPV"], 0.75) and np.isclose(m["NPV"], 5 / 6)


def test_transformer_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = FTTransformerClassifier(max_epochs=40, patience=10, random_state=0).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all()
    assert ((p >= 0.5) == y).mean() > 0.85


def test_saved_models_load_and_score_test_set(trained, splits):
    models, meta = trained
    X, y = get_xy(splits["test"], meta["features"])
    for m in MODEL_ORDER:
        p = models[m].predict_proba(X)[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        thr = meta["models"][m]["decision_threshold"]
        met = compute_metrics(y, p, thr)
        assert met["AUC"] > 0.90, f"{m} AUC regressed"
        assert met["Accuracy"] > 0.85, f"{m} accuracy regressed"


def test_models_accept_raw_rows_with_missing_values(trained, splits):
    models, meta = trained
    X = splits["test"][meta["features"]].head(5).copy()
    X.iloc[:, :] = np.nan
    for m in MODEL_ORDER:
        assert np.isfinite(models[m].predict_proba(X)).all()


def test_vus_reclassification_output(trained, splits):
    models, meta = trained
    res = reclassify(splits["vus"], models, meta)
    assert len(res) == len(splits["vus"])
    assert set(res["Reclassified_As"]) <= {"Pathogenic", "Benign"}
    assert np.allclose(res["P_Pathogenic_Ensemble"] + res["P_Benign_Ensemble"], 1, atol=1e-3)
    assert set(res["Model_Agreement"]) <= {"3/3", "2/3"}


def test_confidence_tiers():
    tiers = confidence_tier(np.array([0.95, 0.05, 0.7, 0.2, 0.5]))
    assert tiers.tolist() == ["High", "High", "Medium", "Medium", "Low"]
