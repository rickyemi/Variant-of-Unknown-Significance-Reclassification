"""
Stage 7: reclassify Variants of Uncertain Significance (VUS).

Each saved model gives P(Pathogenic) for every VUS. The outputs are:

* per-model probability and call (at that model's Youden threshold)
* ensemble probability = mean of the three probabilities
* final call = majority vote of the three model calls (2 of 3)
* agreement = how many models agree with the final call (3/3 or 2/3)
* confidence tier from the ensemble probability:
      High   P >= 0.80 or P <= 0.10
      Medium 0.60 <= P < 0.80 or 0.10 < P <= 0.25
      Low    anything in between (closest to the decision boundary; review first)

The same entry point scores any new CSV with the raw variant columns:
    python -m src.models.predict_model --input path/to/new_variants.csv --output out.csv

This is a research model. Calls are decision support for expert curation,
not a substitute for ACMG/AMP classification.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.models.evaluate_model import MODEL_ORDER
from src.models.train_model import load_models
from src.utils import get_logger
from src.visualization.visualize import save_fig

log = get_logger(__name__)


def confidence_tier(p: np.ndarray) -> np.ndarray:
    return np.select([(p >= 0.80) | (p <= 0.10), (p >= 0.60) | (p <= 0.25)],
                     ["High", "Medium"], "Low")


def reclassify(df: pd.DataFrame, models=None, meta=None) -> pd.DataFrame:
    """Return a table of probabilities and calls for every row of `df`."""
    if models is None:
        models, meta = load_models()
    X = df[meta["features"]]
    out = pd.DataFrame({cfg.ID_COL: df[cfg.ID_COL].values}) if cfg.ID_COL in df else pd.DataFrame(index=df.index)
    votes = np.zeros(len(df), dtype=int)
    probs = []
    for m in MODEL_ORDER:
        p = models[m].predict_proba(X)[:, 1]
        thr = meta["models"][m]["decision_threshold"]
        call = (p >= thr).astype(int)
        votes += call
        probs.append(p)
        key = m.replace("-", "_")
        out[f"P_Pathogenic_{key}"] = p.round(4)
        out[f"Call_{key}"] = np.where(call == 1, cfg.POSITIVE_CLASS, cfg.NEGATIVE_CLASS)
    ens = np.mean(probs, axis=0)
    final = votes >= 2
    out["P_Pathogenic_Ensemble"] = ens.round(4)
    out["P_Benign_Ensemble"] = (1 - ens).round(4)
    out["Reclassified_As"] = np.where(final, cfg.POSITIVE_CLASS, cfg.NEGATIVE_CLASS)
    out["Model_Agreement"] = np.where(final, votes, 3 - votes).astype(str) + "/3"
    out["Confidence"] = confidence_tier(ens)
    return out


def plot_vus(res: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    for m in MODEL_ORDER:
        ax.hist(res[f"P_Pathogenic_{m.replace('-', '_')}"], bins=bins, histtype="step", lw=1.8,
                color=cfg.MODEL_COLORS[m], label=m)
    ax.hist(res["P_Pathogenic_Ensemble"], bins=bins, color=cfg.NEUTRAL, alpha=0.25, label="Ensemble")
    ax.set_xlabel("P(Pathogenic)"); ax.set_ylabel("Number of VUS"); ax.legend()
    ax.set_title("VUS probability distributions")
    ax = axes[1]
    order = [cfg.NEGATIVE_CLASS, cfg.POSITIVE_CLASS]
    tab = pd.crosstab(res["Reclassified_As"], res["Confidence"]).reindex(order).fillna(0)
    tab = tab[[c for c in ["High", "Medium", "Low"] if c in tab.columns]]
    left = np.zeros(len(tab))
    shades = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    for tier in tab.columns:
        ax.barh(tab.index, tab[tier], left=left, height=0.55, alpha=shades[tier],
                color=[cfg.COLORS[c] for c in tab.index], edgecolor=cfg.SURFACE, lw=2,
                label=f"{tier} confidence")
        for i, (l, v) in enumerate(zip(left, tab[tier])):
            if v >= 12:
                ax.text(l + v / 2, i, f"{int(v)}", ha="center", va="center", fontsize=8.5,
                        color="white" if tier == "High" else cfg.TEXT_PRIMARY)
        left += tab[tier].values
    for i, total in enumerate(left):
        ax.text(total + 3, i, f"{int(total)} ({total / len(res):.0%})", va="center", fontsize=9)
    ax.set_xlim(0, left.max() * 1.25); ax.set_xlabel("Number of VUS")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cfg.NEUTRAL, alpha=shades[t], label=f"{t} confidence")
                       for t in tab.columns], loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, fontsize=7.5)
    ax.set_title("Reclassification result (majority vote)")
    ax = axes[2]
    agree = res["Model_Agreement"].value_counts().reindex(["3/3", "2/3"]).fillna(0)
    ax.bar(agree.index, agree.values, color="#3987e5", width=0.55)
    for i, v in enumerate(agree.values):
        ax.text(i, v + 3, f"{int(v)} ({v / len(res):.0%})", ha="center", fontsize=9)
    ax.set_ylim(0, agree.max() * 1.18)
    ax.set_xlabel("Models agreeing with the final call"); ax.set_ylabel("Number of VUS")
    ax.grid(axis="x", visible=False); ax.set_title("Model agreement")
    fig.tight_layout()
    return save_fig(fig, "32_vus_reclassification")


def plot_vus_pairwise(res: pd.DataFrame):
    keys = [m.replace("-", "_") for m in MODEL_ORDER]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (a, b) in zip(axes, pairs):
        x, y = res[f"P_Pathogenic_{keys[a]}"], res[f"P_Pathogenic_{keys[b]}"]
        col = np.where(res["Reclassified_As"] == cfg.POSITIVE_CLASS, cfg.COLORS["Pathogenic"],
                       cfg.COLORS["Benign"])
        ax.scatter(x, y, c=col, s=14, alpha=0.7, edgecolor="none")
        ax.plot([0, 1], [0, 1], color=cfg.NEUTRAL, ls="--", lw=1)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(f"{MODEL_ORDER[a]} P(Pathogenic)"); ax.set_ylabel(f"{MODEL_ORDER[b]} P(Pathogenic)")
        ax.set_title(f"{MODEL_ORDER[a]} vs {MODEL_ORDER[b]} (r = {r:.2f})", fontsize=10)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=cfg.COLORS[k], label=f"Final: {k}")
                            for k in ["Benign", "Pathogenic"]], loc="upper left")
    fig.suptitle("Do the three models agree on each VUS?", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "33_vus_model_concordance")


def main(input_path: Path = cfg.VUS_FILE, output_path: Path = cfg.VUS_PREDICTIONS_FILE):
    cfg.ensure_dirs()
    df = pd.read_csv(input_path, keep_default_na=False, na_values=[""])
    res = reclassify(df)
    res.to_csv(output_path, index=False)
    if Path(input_path) == cfg.VUS_FILE:
        plot_vus(res)
        plot_vus_pairwise(res)
        summary = (res.groupby(["Reclassified_As", "Confidence"]).size().unstack(fill_value=0))
        summary.to_csv(cfg.TABLES_DIR / "vus_reclassification_summary.csv")
    counts = res["Reclassified_As"].value_counts()
    log.info("Reclassified %d variants: %s -> %s", len(res), counts.to_dict(), output_path)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reclassify VUS as Pathogenic or Benign")
    p.add_argument("--input", default=str(cfg.VUS_FILE))
    p.add_argument("--output", default=str(cfg.VUS_PREDICTIONS_FILE))
    a = p.parse_args()
    main(Path(a.input), Path(a.output))
