"""
Stage 2a: biomarker and demographic data analysis.

Questions answered
------------------
* How are the classes distributed, and is the label split imbalanced?
* Do the tumour / clinical biomarkers differ between Pathogenic and Benign
  variants? (Mann-Whitney U with Benjamini-Hochberg FDR, rank-biserial effect
  size, single-biomarker ROC AUC.)
* Where do the VUS sit relative to the two labelled classes?
* Are demographic variables associated with pathogenicity? (chi-square,
  Cramer's V, age distribution.)

Outputs: reports/tables/biomarker_*.csv and reports/figures/01-05_*.png
Run:     python -m src.analysis.biomarker_analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.analysis.stats import categorical_tests, numeric_tests
from src.utils import get_logger
from src.visualization.visualize import save_fig

log = get_logger(__name__)
ORDER3 = [cfg.NEGATIVE_CLASS, cfg.POSITIVE_CLASS, cfg.UNLABELLED_CLASS]


def load_all() -> pd.DataFrame:
    return pd.read_csv(cfg.INTERIM_FILE, keep_default_na=False, na_values=[""])


def labelled(df):
    lab = df[df[cfg.TARGET_COL] != cfg.UNLABELLED_CLASS].reset_index(drop=True)
    y = (lab[cfg.TARGET_COL] == cfg.POSITIVE_CLASS).astype(int)
    return lab, y


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_class_distribution(df):
    counts = df[cfg.TARGET_COL].value_counts().reindex(ORDER3)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(counts.index, counts.values, width=0.6,
                  color=[cfg.COLORS[c] for c in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 25,
                f"{v:,}  ({v / counts.sum():.0%})", ha="center", fontsize=9)
    ax.set_ylabel("Number of variants")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Class distribution: 70% Benign, 20% Pathogenic, 10% VUS")
    return save_fig(fig, "01_class_distribution")


def plot_biomarker_distributions(df):
    cols = cfg.BIOMARKER_NUMERIC
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
    for ax, c in zip(axes.ravel(), cols):
        data = [df.loc[df[cfg.TARGET_COL] == k, c].dropna() for k in ORDER3]
        if c in cfg.LOG_FEATURES:
            data = [np.log10(d) for d in data]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for pc, k in zip(parts["bodies"], ORDER3):
            pc.set_facecolor(cfg.COLORS[k]); pc.set_alpha(0.35); pc.set_edgecolor("none")
        bp = ax.boxplot(data, widths=0.18, patch_artist=True, showfliers=False,
                        medianprops=dict(color=cfg.TEXT_PRIMARY, lw=1.4))
        for patch, k in zip(bp["boxes"], ORDER3):
            patch.set_facecolor(cfg.COLORS[k]); patch.set_edgecolor(cfg.COLORS[k])
        ax.set_xticks([1, 2, 3], ORDER3, fontsize=8)
        ax.set_title(("log10 " if c in cfg.LOG_FEATURES else "") + c, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Tumour and clinical biomarkers by class (VUS fall between the labelled classes)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "02_biomarker_distributions_by_class")


def plot_volcano(tests: pd.DataFrame):
    """Effect size vs significance for every numeric feature."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sig = tests["q_value"] < cfg.FDR_ALPHA
    y = -np.log10(tests["q_value"].clip(lower=1e-300))
    ax.scatter(tests.loc[~sig, "rank_biserial"], y[~sig], s=46, color=cfg.NEUTRAL,
               label=f"Not significant (q >= {cfg.FDR_ALPHA})", edgecolor=cfg.SURFACE, lw=1.5)
    ax.scatter(tests.loc[sig, "rank_biserial"], y[sig], s=46, color="#1c5cab",
               label=f"Significant (q < {cfg.FDR_ALPHA})", edgecolor=cfg.SURFACE, lw=1.5)
    # Label significant points, nudging labels apart so they do not collide.
    placed = []
    span = y.max() - y.min() + 1e-9
    for _, r in tests[sig].sort_values("q_value").iterrows():
        px, py = r["rank_biserial"], -np.log10(max(r["q_value"], 1e-300))
        ly = py
        while any(abs(ly - q) < 0.045 * span and abs(px - p) < 0.25 for p, q in placed):
            ly -= 0.045 * span
        placed.append((px, ly))
        ax.annotate(r["feature"], (px, py), xytext=(px + 0.02, ly), fontsize=7.2,
                    color=cfg.TEXT_SECONDARY, va="center",
                    arrowprops=dict(arrowstyle="-", color=cfg.GRID, lw=0.8) if ly != py else None)
    ns = ", ".join(tests.loc[~sig, "feature"])
    if ns:
        ax.text(0.02, 0.1, f"Not significant: {ns}", transform=ax.transAxes, fontsize=7.2,
                color=cfg.TEXT_SECONDARY)
    ax.axhline(-np.log10(cfg.FDR_ALPHA), color=cfg.NEUTRAL, lw=1, ls="--")
    ax.axvline(0, color=cfg.GRID, lw=1)
    ax.set_xlabel("Rank-biserial effect size (positive = higher in Pathogenic)")
    ax.set_ylabel("-log10 FDR q-value")
    ax.set_title("Volcano plot: numeric features, Pathogenic vs Benign")
    ax.legend(loc="upper center")
    return save_fig(fig, "03_biomarker_volcano_plot")


def plot_univariate_auc(tests: pd.DataFrame):
    t = tests.sort_values("univariate_auc")
    fig, ax = plt.subplots(figsize=(7, 5.4))
    colors = ["#1c5cab" if q < cfg.FDR_ALPHA else cfg.NEUTRAL for q in t["q_value"]]
    ax.barh(t["feature"], t["univariate_auc"] - 0.5, left=0.5, color=colors, height=0.65)
    for i, v in enumerate(t["univariate_auc"]):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=7.5, color=cfg.TEXT_SECONDARY)
    ax.axvline(0.5, color=cfg.NEUTRAL, lw=1)
    ax.set_xlim(0.45, max(1.0, t["univariate_auc"].max() + 0.05))
    ax.set_xlabel("Single-feature ROC AUC (0.5 = no signal)")
    ax.grid(axis="y", visible=False)
    ax.set_title("How well does each numeric feature separate the classes on its own?")
    fig.tight_layout()
    return save_fig(fig, "04_univariate_auc", "Dark blue = FDR q < 0.05; grey = not significant")


def plot_demography(df):
    lab = df.copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    ax = axes[0, 0]
    for k in ORDER3:
        ax.hist(lab.loc[lab[cfg.TARGET_COL] == k, "Age"].dropna(), bins=25, density=True,
                histtype="step", lw=2, color=cfg.COLORS[k], label=k)
    ax.set_title("Age at testing"); ax.set_xlabel("Years"); ax.legend()
    for ax, c in zip(axes.ravel()[1:], ["Sex", "Ethnicity", "Smoking_Status",
                                         "Family_History_Cancer", "Cancer_Type"]):
        tab = pd.crosstab(lab[c], lab[cfg.TARGET_COL], normalize="columns")[ORDER3]
        x = np.arange(len(tab)); w = 0.27
        for i, k in enumerate(ORDER3):
            ax.bar(x + (i - 1) * w, tab[k], width=w - 0.02, color=cfg.COLORS[k], label=k)
        ax.set_xticks(x, tab.index, rotation=25 if len(tab) > 3 else 0, fontsize=8)
        ax.set_ylabel("Share within class"); ax.set_title(c.replace("_", " "))
        ax.grid(axis="x", visible=False)
    axes[0, 1].legend()
    fig.suptitle("Demographic profile by class (only family history differs clearly)", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "05_demography_by_class")


# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame | None = None) -> dict:
    """Run the full biomarker/demography analysis and return the result tables."""
    cfg.ensure_dirs()
    df = load_all() if df is None else df
    lab, y = labelled(df)

    num_cols = cfg.BIOMARKER_NUMERIC + cfg.DEMOGRAPHIC_NUMERIC + cfg.GENOMIC_NUMERIC
    num_tests = numeric_tests(lab, num_cols, y)
    num_tests.insert(1, "group", num_tests["feature"].map(cfg.FEATURE_GROUP))
    demo_tests = categorical_tests(lab, cfg.DEMOGRAPHIC_CATEGORICAL, y)

    # Descriptive summary with VUS included, for context.
    summary = (df.groupby(cfg.TARGET_COL)[cfg.BIOMARKER_NUMERIC + cfg.DEMOGRAPHIC_NUMERIC]
               .median().T[ORDER3].round(3))
    summary.columns = [f"median_{c}" for c in summary.columns]

    num_tests.to_csv(cfg.TABLES_DIR / "biomarker_numeric_tests.csv", index=False)
    demo_tests.to_csv(cfg.TABLES_DIR / "biomarker_demography_tests.csv", index=False)
    summary.to_csv(cfg.TABLES_DIR / "biomarker_median_by_class.csv")

    plot_class_distribution(df)
    plot_biomarker_distributions(df)
    plot_volcano(num_tests)
    plot_univariate_auc(num_tests)
    plot_demography(df)
    log.info("Biomarker analysis: %d/%d numeric features significant at q < %.2f",
             (num_tests.q_value < cfg.FDR_ALPHA).sum(), len(num_tests), cfg.FDR_ALPHA)
    return {"numeric_tests": num_tests, "demography_tests": demo_tests, "summary": summary}


if __name__ == "__main__":
    run()
