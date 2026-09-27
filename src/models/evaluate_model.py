"""
Stage 5: evaluate and compare the three models on Train and Test.

Metrics (Pathogenic = positive class, each model at its own out-of-fold Youden
threshold):
    Accuracy     (TP + TN) / N
    Sensitivity  TP / (TP + FN)   recall for Pathogenic
    Specificity  TN / (TN + FP)   recall for Benign
    PPV          TP / (TP + FP)   precision for Pathogenic
    NPV          TN / (TN + FN)   precision for Benign
    AUC          area under the ROC curve (threshold-free)
    F1           harmonic mean of PPV and Sensitivity
plus Balanced accuracy, MCC and the Brier score (calibration).

Also produces ROC and PR curves, confusion matrices, calibration curves, a
metric comparison chart, XGBoost native importance and model-agnostic
permutation importance for all three models.

Run:  python -m src.models.evaluate_model
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src import config as cfg
from src.data.make_dataset import get_xy, load_split
from src.models.train_model import load_models
from src.utils import get_logger
from src.visualization.visualize import SEQ_CMAP, save_fig

log = get_logger(__name__)
MODEL_ORDER = ["XGBoost", "SVM-RBF", "Transformer"]
HEADLINE = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "F1"]


def compute_metrics(y, p, threshold) -> dict:
    """All classification metrics for probabilities `p` at a given threshold."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {
        "Accuracy": accuracy_score(y, pred),
        "Sensitivity": safe(tp, tp + fn),
        "Specificity": safe(tn, tn + fp),
        "PPV": safe(tp, tp + fp),
        "NPV": safe(tn, tn + fn),
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "Brier": brier_score_loss(y, p),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def predict_all(models, X) -> dict:
    return {n: models[n].predict_proba(X)[:, 1] for n in MODEL_ORDER}


def comparison_table(models, meta, sets) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        thr = meta["models"][name]["decision_threshold"]
        for split_name, (X, y) in sets.items():
            p = models[name].predict_proba(X)[:, 1]
            rows.append({"Model": name, "Set": split_name, "Threshold": thr,
                         **compute_metrics(y, p, thr)})
    return pd.DataFrame(rows)


def to_markdown(tab: pd.DataFrame) -> str:
    """Wide Train/Test comparison table in Markdown (used by the README)."""
    lines = ["| Metric | " + " | ".join(f"{m} Train | {m} Test" for m in MODEL_ORDER) + " |",
             "|---|" + "---:|" * (2 * len(MODEL_ORDER))]
    for met in HEADLINE:
        vals = []
        for m in MODEL_ORDER:
            for s in ["Train", "Test"]:
                vals.append(f"{tab[(tab.Model == m) & (tab.Set == s)][met].iloc[0]:.3f}")
        lines.append(f"| {met} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def plot_roc_pr(probs, ys):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, s in zip(axes[:2], ["Train", "Test"]):
        for m in MODEL_ORDER:
            fpr, tpr, _ = roc_curve(ys[s], probs[s][m])
            ax.plot(fpr, tpr, color=cfg.MODEL_COLORS[m],
                    label=f"{m} (AUC {roc_auc_score(ys[s], probs[s][m]):.3f})")
        ax.plot([0, 1], [0, 1], color=cfg.NEUTRAL, ls="--", lw=1)
        ax.set_xlabel("1 - Specificity (false positive rate)"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"ROC curve, {s} set"); ax.legend(loc="lower right")
    ax = axes[2]
    for m in MODEL_ORDER:
        pr, rc, _ = precision_recall_curve(ys["Test"], probs["Test"][m])
        ax.plot(rc, pr, color=cfg.MODEL_COLORS[m],
                label=f"{m} (AP {average_precision_score(ys['Test'], probs['Test'][m]):.3f})")
    ax.axhline(ys["Test"].mean(), color=cfg.NEUTRAL, ls="--", lw=1)
    ax.text(0.01, ys["Test"].mean() + 0.015, "prevalence", fontsize=7.5, color=cfg.TEXT_SECONDARY)
    ax.set_xlabel("Sensitivity (recall)"); ax.set_ylabel("PPV (precision)")
    ax.set_title("Precision-recall curve, Test set"); ax.legend(loc="lower left")
    fig.tight_layout()
    return save_fig(fig, "19_roc_pr_curves")


def plot_confusion(tab):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for j, m in enumerate(MODEL_ORDER):
        for i, s in enumerate(["Train", "Test"]):
            r = tab[(tab.Model == m) & (tab.Set == s)].iloc[0]
            cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
            ax = axes[i, j]
            ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQ_CMAP, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    share = cm[a, b] / cm[a].sum()
                    ax.text(b, a, f"{cm[a, b]}\n({share:.0%})", ha="center", va="center",
                            fontsize=10, color="white" if share > 0.55 else cfg.TEXT_PRIMARY)
            ax.set_xticks([0, 1], ["Benign", "Pathogenic"]); ax.set_yticks([0, 1], ["Benign", "Pathogenic"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
            ax.set_title(f"{m}, {s} (threshold {r.Threshold:.2f})", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "20_confusion_matrices", "Row percentages: sensitivity and specificity on the diagonal")


def plot_calibration(probs, ys):
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in MODEL_ORDER:
        fp, mp = calibration_curve(ys["Test"], probs["Test"][m], n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", ms=5, color=cfg.MODEL_COLORS[m],
                label=f"{m} (Brier {brier_score_loss(ys['Test'], probs['Test'][m]):.3f})")
    ax.plot([0, 1], [0, 1], color=cfg.NEUTRAL, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted P(Pathogenic)"); ax.set_ylabel("Observed Pathogenic fraction")
    ax.set_title("Calibration (reliability) curve, Test set"); ax.legend(loc="upper left")
    return save_fig(fig, "21_calibration_curves")


def plot_metric_comparison(tab):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(HEADLINE)); w = 0.26
    for i, m in enumerate(MODEL_ORDER):
        te = tab[(tab.Model == m) & (tab.Set == "Test")][HEADLINE].iloc[0].values
        tr = tab[(tab.Model == m) & (tab.Set == "Train")][HEADLINE].iloc[0].values
        ax.bar(x + (i - 1) * w, te, width=w - 0.03, color=cfg.MODEL_COLORS[m], label=f"{m} Test")
        ax.scatter(x + (i - 1) * w, tr, marker="_", s=260, color=cfg.TEXT_PRIMARY, lw=2,
                   label="Train value" if i == 0 else None, zorder=3)
    ax.set_xticks(x, HEADLINE); ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score")
    ax.grid(axis="x", visible=False); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Model comparison: Test bars with Train markers (close markers = little overfitting)")
    fig.tight_layout()
    return save_fig(fig, "22_metric_comparison")


def plot_xgb_importance(model):
    clf = model.named_steps["clf"]
    names = model.named_steps["prep"].get_feature_names_out()
    gain = pd.Series(clf.get_booster().get_score(importance_type="gain"))
    gain.index = [names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k for k in gain.index]
    gain = (gain / gain.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.barh(gain.index, gain.values, color="#2a78d6", height=0.65)
    ax.set_xlabel("Share of total gain"); ax.tick_params(axis="y", labelsize=7.8)
    ax.grid(axis="y", visible=False); ax.set_title("XGBoost native feature importance (gain)")
    fig.tight_layout()
    save_fig(fig, "23_xgboost_gain_importance")
    return gain


def permutation_table(models, X, y) -> pd.DataFrame:
    rows = []
    for m in MODEL_ORDER:
        r = permutation_importance(models[m], X, y, scoring="roc_auc", n_repeats=10,
                                   random_state=cfg.RANDOM_STATE, n_jobs=1)
        for f, mu, sd in zip(X.columns, r.importances_mean, r.importances_std):
            rows.append({"Model": m, "feature": f, "auc_drop_mean": mu, "auc_drop_sd": sd})
    return pd.DataFrame(rows)


def plot_permutation(perm: pd.DataFrame):
    order = perm.groupby("feature")["auc_drop_mean"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    yy = np.arange(len(order)); h = 0.26
    for i, m in enumerate(MODEL_ORDER):
        d = perm[perm.Model == m].set_index("feature").loc[order]
        ax.barh(yy + (1 - i) * h, d["auc_drop_mean"], xerr=d["auc_drop_sd"], height=h - 0.02,
                color=cfg.MODEL_COLORS[m], label=m, error_kw=dict(ecolor=cfg.NEUTRAL, lw=0.8))
    ax.set_yticks(yy, order, fontsize=8); ax.axvline(0, color=cfg.NEUTRAL, lw=1)
    ax.set_xlabel("Drop in Test ROC AUC when the feature is shuffled (10 repeats)")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
    ax.set_title("Permutation feature importance, all three models")
    fig.tight_layout()
    return save_fig(fig, "24_permutation_importance")


# --------------------------------------------------------------------------- #
def run() -> dict:
    cfg.ensure_dirs()
    models, meta = load_models()
    feats = meta["features"]
    Xtr, ytr = get_xy(load_split("train"), feats)
    Xte, yte = get_xy(load_split("test"), feats)
    sets = {"Train": (Xtr, ytr), "Test": (Xte, yte)}

    tab = comparison_table(models, meta, sets)
    tab.round(4).to_csv(cfg.TABLES_DIR / "model_comparison_train_test.csv", index=False)
    md = to_markdown(tab)
    (cfg.TABLES_DIR / "model_comparison_train_test.md").write_text(md + "\n")

    probs = {s: predict_all(models, X) for s, (X, _) in sets.items()}
    ys = {s: y.values for s, (_, y) in sets.items()}
    plot_roc_pr(probs, ys)
    plot_confusion(tab)
    plot_calibration(probs, ys)
    plot_metric_comparison(tab)
    gain = plot_xgb_importance(models["XGBoost"])
    gain.sort_values(ascending=False).to_csv(cfg.TABLES_DIR / "xgboost_gain_importance.csv",
                                             header=["gain_share"])
    perm = permutation_table(models, Xte, yte)
    perm.to_csv(cfg.TABLES_DIR / "permutation_importance.csv", index=False)
    plot_permutation(perm)
    log.info("Train/Test comparison:\n%s", md)
    return {"table": tab, "markdown": md, "permutation": perm, "xgb_gain": gain}


if __name__ == "__main__":
    run()
