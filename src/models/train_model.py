"""
Stage 4: train the three classifiers and save them as joblib files.

Each model is a single scikit-learn Pipeline:
    preprocessing (log10 -> IQR capping -> imputation -> z-score -> one-hot)
    -> classifier
so the saved .joblib file takes raw variant columns and returns probabilities,
with no separate preprocessing step to forget in production.

Models
------
1. XGBoost      RandomizedSearchCV over 9 hyperparameters (40 candidates x 5 folds)
2. SVM (RBF)    RandomizedSearchCV over C, gamma, class_weight (30 candidates x 5 folds)
3. Transformer  FT-Transformer (PyTorch) with early stopping

Candidate choice guards against overfitting: among candidates whose mean
train-vs-validation AUC gap is <= 0.03, the one with the best validation AUC
wins (if none qualify, the smallest gap wins). This keeps Train and Test
performance close.

Decision threshold: the probability cut-off is picked on out-of-fold
training predictions (never the test set) to maximise Youden's J
(sensitivity + specificity - 1), which balances missed pathogenic variants
against false alarms given the 22% Pathogenic prevalence.

Run:  python -m src.models.train_model
"""
import platform
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import config as cfg
from src.data.make_dataset import get_xy, load_split
from src.features.build_features import build_preprocessor, split_types
from src.models.transformer_model import FTTransformerClassifier
from src.utils import get_logger, load_json, save_json, set_seed
from src.visualization.visualize import save_fig

log = get_logger(__name__)
torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
def make_pipeline(model, numeric, categorical) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", model)])


XGB_SPACE = {
    "clf__n_estimators": randint(100, 600),
    "clf__max_depth": randint(2, 6),
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__min_child_weight": randint(1, 30),
    "clf__gamma": uniform(0, 5),
    "clf__reg_lambda": loguniform(1e-2, 20),
    "clf__reg_alpha": loguniform(1e-3, 5),
}
SVM_SPACE = {
    "clf__C": loguniform(1e-2, 1e2),
    "clf__gamma": loguniform(1e-4, 1),
    "clf__class_weight": [None, "balanced"],
}
TRANSFORMER_PARAMS = dict(d_token=32, n_heads=4, n_layers=2, ff_mult=2, dropout=0.2,
                          lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=200,
                          patience=20, random_state=cfg.RANDOM_STATE)


def base_models():
    return {
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                 tree_method="hist", n_jobs=1, random_state=cfg.RANDOM_STATE),
        "SVM-RBF": SVC(kernel="rbf", probability=False, random_state=cfg.RANDOM_STATE),
        "Transformer": FTTransformerClassifier(**TRANSFORMER_PARAMS),
    }


def pick_candidate(results: pd.DataFrame, max_gap=cfg.MAX_TRAIN_VAL_GAP) -> int:
    """Index of the best-validating candidate whose train/val gap is acceptable."""
    gap = results["mean_train_score"] - results["mean_test_score"]
    ok = gap <= max_gap
    if ok.any():
        return int(results.loc[ok, "mean_test_score"].idxmax())
    return int(gap.idxmin())


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(np.clip(thr[np.argmax(j)], 0.01, 0.99))


# --------------------------------------------------------------------------- #
def tune(name, pipe, space, n_iter, X, y, cv):
    log.info("%s: RandomizedSearchCV with %d candidates x %d folds", name, n_iter, cfg.CV_FOLDS)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                n_jobs=2, refit=False, return_train_score=True,
                                random_state=cfg.RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    idx = pick_candidate(res)
    res["gap"] = res["mean_train_score"] - res["mean_test_score"]
    res["chosen"] = False
    res.loc[idx, "chosen"] = True
    params = res.loc[idx, "params"]
    log.info("%s: chosen CV AUC %.4f (train %.4f, gap %.3f), best overall %.4f",
             name, res.loc[idx, "mean_test_score"], res.loc[idx, "mean_train_score"],
             res.loc[idx, "gap"], res["mean_test_score"].max())
    keep = [c for c in res.columns if c.startswith("param_")] + [
        "mean_train_score", "mean_test_score", "std_test_score", "gap", "chosen"]
    res[keep].sort_values("mean_test_score", ascending=False).to_csv(
        cfg.TABLES_DIR / f"search_results_{name.lower().replace('-', '_')}.csv", index=False)
    return params, res


def plot_search(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, name in zip(axes[:2], ["XGBoost", "SVM-RBF"]):
        r = results[name]
        ax.scatter(r["mean_train_score"], r["mean_test_score"], s=26, color=cfg.MODEL_COLORS[name],
                   alpha=0.55, edgecolor=cfg.SURFACE, lw=1, label="Candidate")
        c = r[r.chosen]
        ax.scatter(c["mean_train_score"], c["mean_test_score"], s=110, marker="*",
                   color=cfg.TEXT_PRIMARY, label="Chosen", zorder=3)
        lo = min(r["mean_test_score"].min(), r["mean_train_score"].min())
        hi = max(r["mean_test_score"].max(), r["mean_train_score"].max())
        ax.plot([lo, hi], [lo, hi], color=cfg.NEUTRAL, lw=1, ls="--")
        ax.plot([lo, hi], [lo - cfg.MAX_TRAIN_VAL_GAP, hi - cfg.MAX_TRAIN_VAL_GAP],
                color="#eb6834", lw=1, ls=":")
        ax.set_xlabel("Mean CV train AUC"); ax.set_ylabel("Mean CV validation AUC")
        ax.set_title(f"{name}: RandomizedSearchCV candidates"); ax.legend(loc="lower right")
    ax = axes[2]
    r = results["SVM-RBF"]
    sc = ax.scatter(r["param_clf__C"].astype(float), r["param_clf__gamma"].astype(float),
                    c=r["mean_test_score"], cmap="Blues", s=60, edgecolor=cfg.NEUTRAL, lw=0.5)
    c = r[r.chosen]
    ax.scatter(c["param_clf__C"].astype(float), c["param_clf__gamma"].astype(float), s=160,
               marker="*", color="#eb6834", zorder=3, label="Chosen")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("C"); ax.set_ylabel("gamma")
    fig.colorbar(sc, ax=ax, label="CV validation AUC"); ax.legend(loc="lower left")
    ax.set_title("SVM-RBF: C x gamma landscape")
    fig.tight_layout()
    return save_fig(fig, "17_hyperparameter_search",
                    "Dashed = no overfitting; dotted orange = 0.03 AUC gap guard. Candidates below the dotted line are rejected")


def plot_transformer_history(model: FTTransformerClassifier):
    h = model.history_
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, h["train_loss"], color="#2a78d6", label="Train loss")
    a1.plot(ep, h["val_loss"], color="#eb6834", label="Validation loss")
    a1.axvline(model.best_epoch_ + 1, color=cfg.NEUTRAL, ls="--", lw=1)
    a1.text(model.best_epoch_ + 1, a1.get_ylim()[1], " best epoch", va="top", fontsize=7.5,
            color=cfg.TEXT_SECONDARY)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Binary cross-entropy"); a1.legend()
    a1.set_title("FT-Transformer training curve (early stopping)")
    a2.plot(ep, h["val_auc"], color="#1baf7a")
    a2.axvline(model.best_epoch_ + 1, color=cfg.NEUTRAL, ls="--", lw=1)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation ROC AUC")
    a2.set_title("Validation AUC by epoch")
    fig.tight_layout()
    return save_fig(fig, "18_transformer_training_curve")


# --------------------------------------------------------------------------- #
def run() -> dict:
    cfg.ensure_dirs()
    set_seed(cfg.RANDOM_STATE)
    sel = load_json(cfg.SELECTED_FEATURES_FILE)
    features = sel["selected_features"]
    num, cat = split_types(features)
    train = load_split("train")
    X, y = get_xy(train, features)
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    models = base_models()
    search_results, metadata, fitted = {}, {}, {}
    for name in ["XGBoost", "SVM-RBF", "Transformer"]:
        t0 = time.time()
        pipe = make_pipeline(models[name], num, cat)
        if name == "XGBoost":
            params, search_results[name] = tune(name, pipe, XGB_SPACE, cfg.N_ITER_XGB, X, y, cv)
        elif name == "SVM-RBF":
            params, search_results[name] = tune(name, pipe, SVM_SPACE, cfg.N_ITER_SVM, X, y, cv)
            params = {**params, "clf__probability": True}   # enable Platt-scaled probabilities
        else:
            params = {}
        final = clone(pipe).set_params(**params)

        # Out-of-fold probabilities on the training set -> threshold + honest CV AUC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof = cross_val_predict(final, X, y, cv=cv, method="predict_proba")[:, 1]
            final.fit(X, y)
        thr = youden_threshold(y, oof)
        joblib.dump(final, cfg.MODEL_FILES[name])
        fitted[name] = final
        metadata[name] = {
            "file": cfg.MODEL_FILES[name].name,
            "best_params": {k.replace("clf__", ""): v for k, v in params.items()}
            if params else {k: v for k, v in TRANSFORMER_PARAMS.items()},
            "oof_cv_auc": float(roc_auc_score(y, oof)),
            "decision_threshold": thr,
            "train_seconds": round(time.time() - t0, 1),
        }
        if name == "Transformer":
            metadata[name]["best_epoch"] = int(final.named_steps["clf"].best_epoch_) + 1
        log.info("%s: OOF AUC %.4f, Youden threshold %.3f, saved %s (%.0fs)", name,
                 metadata[name]["oof_cv_auc"], thr, cfg.MODEL_FILES[name].name,
                 metadata[name]["train_seconds"])

    plot_search(search_results)
    plot_transformer_history(fitted["Transformer"].named_steps["clf"])
    save_json({"features": features, "numeric": num, "categorical": cat,
               "positive_class": cfg.POSITIVE_CLASS, "models": metadata,
               "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "xgboost": xgboost.__version__, "torch": torch.__version__}},
              cfg.MODEL_METADATA_FILE)
    return {"models": fitted, "metadata": metadata, "search": search_results}


def load_models() -> dict:
    """Load the three saved pipelines plus their metadata."""
    meta = load_json(cfg.MODEL_METADATA_FILE)
    models = {n: joblib.load(p) for n, p in cfg.MODEL_FILES.items()}
    return models, meta


if __name__ == "__main__":
    run()
