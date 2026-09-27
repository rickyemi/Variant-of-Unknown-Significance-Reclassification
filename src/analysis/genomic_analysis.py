"""
Stage 2b: genomic (variant-level) data analysis.

Questions answered
------------------
* Which variant consequences (nonsense, frameshift, splice, missense,
  synonymous) are enriched for pathogenicity? (Fisher's exact odds ratios.)
* How does the pathogenic rate vary by gene, and by gene x consequence?
* Do the in-silico predictors (CADD, REVEL, SIFT, PolyPhen-2, SpliceAI) and
  conservation scores (GERP, PhyloP) separate the classes, and do they agree?
* Are pathogenic variants rarer in the population (gnomAD allele frequency),
  in line with the ACMG BA1 / BS1 / PM2 frequency criteria?

Outputs: reports/tables/genomic_*.csv and reports/figures/06-10_*.png
Run:     python -m src.analysis.genomic_analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.analysis.biomarker_analysis import ORDER3, labelled, load_all
from src.analysis.stats import level_enrichment, wilson_ci
from src.utils import get_logger
from src.visualization.visualize import SEQ_CMAP, save_fig

log = get_logger(__name__)

# Conventional "damaging" cut-offs used to count concordant predictor calls.
DAMAGING_RULES = {
    "CADD_Phred": lambda s: s >= 20,
    "REVEL_Score": lambda s: s >= 0.5,
    "SIFT_Score": lambda s: s < 0.05,
    "PolyPhen2_Score": lambda s: s >= 0.85,
    "SpliceAI_Score": lambda s: s >= 0.2,
    "GERP_RS": lambda s: s >= 2,
    "PhyloP_100way": lambda s: s >= 2,
}


def damaging_votes(df: pd.DataFrame) -> pd.Series:
    """Number of predictors (out of 7) that call the variant damaging.
    Missing scores do not vote, so this is a lower bound."""
    votes = sum(rule(df[c]).astype(int).where(df[c].notna(), 0)
                for c, rule in DAMAGING_RULES.items())
    return votes


# --------------------------------------------------------------------------- #
def plot_variant_type(df):
    tab = pd.crosstab(df["Variant_Type"], df[cfg.TARGET_COL], normalize="columns")[ORDER3]
    order = ["Nonsense", "Frameshift", "Splice_site", "Missense", "Synonymous"]
    tab = tab.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tab)); w = 0.27
    for i, k in enumerate(ORDER3):
        ax.bar(x + (i - 1) * w, tab[k], width=w - 0.02, color=cfg.COLORS[k], label=k)
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("Share of variants within class")
    ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("Variant consequence by class: loss-of-function types dominate Pathogenic")
    return save_fig(fig, "06_variant_type_by_class")


def plot_enrichment_forest(enr: pd.DataFrame):
    e = enr.sort_values("odds_ratio").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(e) + 1.2))
    y = np.arange(len(e))
    sig = e["fisher_q"] < cfg.FDR_ALPHA
    ax.hlines(y, e["or_ci_low"], e["or_ci_high"], color=np.where(sig, "#1c5cab", cfg.NEUTRAL), lw=2)
    ax.scatter(e["odds_ratio"], y, color=np.where(sig, "#1c5cab", cfg.NEUTRAL), s=40, zorder=3)
    ax.axvline(1, color=cfg.NEUTRAL, ls="--", lw=1)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y, [f"{f.replace('_', ' ')}: {l}" for f, l in zip(e["feature"], e["level"])],
                  fontsize=8)
    ax.set_xlabel("Odds ratio of being Pathogenic (log scale, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Genomic enrichment: which categories raise the odds of pathogenicity?")
    fig.tight_layout()
    return save_fig(fig, "07_genomic_enrichment_forest",
                    "Fisher's exact test, level vs rest; dark blue = FDR q < 0.05")


def plot_gene_landscape(lab, y):
    rows = []
    for g, sub in lab.groupby("Gene"):
        k = int((sub[cfg.TARGET_COL] == cfg.POSITIVE_CLASS).sum()); n = len(sub)
        lo, hi = wilson_ci(k, n)
        rows.append({"Gene": g, "rate": k / n, "lo": lo, "hi": hi, "n": n})
    gdf = pd.DataFrame(rows).sort_values("rate")
    heat = (lab.assign(y=y).pivot_table(index="Gene", columns="Variant_Type", values="y",
                                        aggfunc="mean")
            .reindex(gdf["Gene"][::-1])
            [["Nonsense", "Frameshift", "Splice_site", "Missense", "Synonymous"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    yy = np.arange(len(gdf))
    a1.hlines(yy, gdf["lo"], gdf["hi"], color="#1c5cab", lw=2)
    a1.scatter(gdf["rate"], yy, color="#1c5cab", s=40, zorder=3)
    a1.axvline(y.mean(), color=cfg.NEUTRAL, ls="--", lw=1)
    a1.text(y.mean(), -0.6, " overall rate", fontsize=7.5, color=cfg.TEXT_SECONDARY, va="top")
    a1.set_yticks(yy, gdf["Gene"]); a1.set_xlabel("Pathogenic rate (95% Wilson CI)")
    a1.grid(axis="y", visible=False)
    a1.set_title("Pathogenic rate by gene")
    im = a2.imshow(heat.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    a2.set_xticks(range(heat.shape[1]), heat.columns, rotation=20)
    a2.set_yticks(range(heat.shape[0]), heat.index)
    a2.grid(False)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                a2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else cfg.TEXT_PRIMARY)
    fig.colorbar(im, ax=a2, fraction=0.04, label="Pathogenic rate")
    a2.set_title("Pathogenic rate: gene x variant consequence")
    fig.tight_layout()
    return save_fig(fig, "08_gene_landscape"), gdf


def plot_insilico(df):
    cols = ["CADD_Phred", "REVEL_Score", "SIFT_Score", "PolyPhen2_Score",
            "SpliceAI_Score", "GERP_RS", "PhyloP_100way", "gnomAD_AF"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
    for ax, c in zip(axes.ravel(), cols):
        for k in ORDER3:
            v = df.loc[df[cfg.TARGET_COL] == k, c].dropna()
            if c == "gnomAD_AF":
                v = np.log10(v)
            ax.hist(v, bins=35, density=True, histtype="step", lw=1.8, color=cfg.COLORS[k], label=k)
        if c == "gnomAD_AF":
            for thr, lab_ in [(np.log10(0.05), "BA1"), (np.log10(0.01), "BS1"), (np.log10(1e-4), "PM2")]:
                ax.axvline(thr, color=cfg.NEUTRAL, ls=":", lw=1)
                ax.text(thr, ax.get_ylim()[1] * 0.92, lab_, fontsize=7, color=cfg.TEXT_SECONDARY,
                        ha="right", rotation=90)
            ax.set_title("log10 gnomAD allele frequency", fontsize=9.5)
        else:
            ax.set_title(c, fontsize=9.5)
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("In-silico pathogenicity, conservation and population-frequency scores by class",
                 x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "09_insilico_scores_by_class")


def plot_concordance(df):
    votes = damaging_votes(df)
    tab = pd.crosstab(votes, df[cfg.TARGET_COL], normalize="columns")[ORDER3]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = tab.index.values; w = 0.27
    for i, k in enumerate(ORDER3):
        ax.bar(x + (i - 1) * w, tab[k], width=w - 0.02, color=cfg.COLORS[k], label=k)
    ax.set_xticks(range(0, 8))
    ax.set_xlabel("Number of the 7 predictors calling the variant damaging")
    ax.set_ylabel("Share of variants within class")
    ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("In-silico concordance: Pathogenic variants collect more damaging calls")
    return save_fig(fig, "10_insilico_concordance")


# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame | None = None) -> dict:
    cfg.ensure_dirs()
    df = load_all() if df is None else df
    lab, y = labelled(df)

    enr = pd.concat([level_enrichment(lab, c, y)
                     for c in ["Variant_Type", "In_Functional_Domain", "Gene"]],
                    ignore_index=True)
    from src.analysis.stats import bh_fdr
    enr["fisher_q"] = bh_fdr(enr["fisher_p"])
    enr.to_csv(cfg.TABLES_DIR / "genomic_enrichment.csv", index=False)

    plot_variant_type(df)
    plot_enrichment_forest(enr[enr["feature"] != "Gene"])
    _, gdf = plot_gene_landscape(lab, y)
    gdf.to_csv(cfg.TABLES_DIR / "genomic_gene_pathogenic_rate.csv", index=False)
    plot_insilico(df)
    plot_concordance(df)

    conc = (df.assign(damaging_votes=damaging_votes(df))
              .groupby(cfg.TARGET_COL)["damaging_votes"].describe().round(2))
    conc.to_csv(cfg.TABLES_DIR / "genomic_insilico_concordance.csv")
    log.info("Genomic analysis done: %d enrichment tests, %d significant",
             len(enr), (enr.fisher_q < cfg.FDR_ALPHA).sum())
    return {"enrichment": enr, "gene_rates": gdf, "concordance": conc}


if __name__ == "__main__":
    run()
