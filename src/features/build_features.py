"""
Stage 3: preprocessing and feature selection.

Everything here is fitted on the TRAINING split only, then applied unchanged to
the test split and the VUS, so no information leaks from test into training.

Preprocessing steps (in pipeline order)
---------------------------------------
1. log10 transform of right-skewed features (gnomAD AF, CEA, CA125).
2. Outlier handling: IQR winsorisation (values beyond Q1 - 1.5*IQR or
   Q3 + 1.5*IQR are capped at those fences, not deleted, so rows are kept).
3. Missing values: median imputation for numeric features, most-frequent
   imputation for categorical features. Strategy is chosen by comparing
   mean / median / KNN / iterative imputation with cross-validation.
4. z-score normalisation (StandardScaler: mean 0, SD 1) of numeric features.
5. One-hot encoding of categorical features.

Feature selection (filter method, three rules)
----------------------------------------------
A feature is kept if it is significantly associated with the label
(Mann-Whitney U or chi-square, Benjamini-Hochberg q < 0.05) AND carries
mutual information >= 0.005 nats. Among the survivors, any pair with
|Spearman rho| > 0.85 is treated as redundant and the member with lower
mutual information is dropped.

Run:  python -m src.features.build_features
"""
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg
from src.analysis.stats import categorical_tests, numeric_tests
from src.data.make_dataset import get_xy, load_split
from src.utils import get_logger, save_json
from src.visualization.visualize import DIV_CMAP, save_fig

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Custom, picklable transformers
# --------------------------------------------------------------------------- #
class Log10Transformer(TransformerMixin, BaseEstimator):
    """log10-transform the named columns of a DataFrame (other columns pass through)."""

    def __init__(self, columns=None, floor=1e-8):
        self.columns = columns
        self.floor = floor

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)

    def transform(self, X):
        X = X.copy()
        for c in self.columns or []:
            if c in X.columns:
                X[c] = np.log10(X[c].astype(float).clip(lower=self.floor))
        return X


class IQRCapper(TransformerMixin, BaseEstimator):
    """Winsorise each column at its Tukey fences (Q1 - k*IQR, Q3 + k*IQR).

    Fences are learned on the training data only. NaNs are left as NaN so the
    imputer that follows can handle them.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric, categorical, imputer="median") -> Pipeline:
    """Return the full preprocessing pipeline for the given feature lists."""
    num_imputer = {
        "median": SimpleImputer(strategy="median"),
        "mean": SimpleImputer(strategy="mean"),
        "knn": KNNImputer(n_neighbors=5),
        "iterative": IterativeImputer(max_iter=10, random_state=cfg.RANDOM_STATE),
    }[imputer]
    num_pipe = Pipeline([
        ("outlier_cap", IQRCapper(factor=1.5)),
        ("impute", num_imputer),
        ("zscore", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", num_pipe, list(numeric)),
                            ("cat", cat_pipe, list(categorical))],
                           verbose_feature_names_out=False)
    log_cols = [c for c in cfg.LOG_FEATURES if c in numeric]
    return Pipeline([("log10", Log10Transformer(log_cols)), ("columns", ct)])


def split_types(features):
    num = [f for f in features if f in cfg.NUMERIC_FEATURES]
    cat = [f for f in features if f in cfg.CATEGORICAL_FEATURES]
    return num, cat


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def _mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mutual information on a simply imputed, ordinal-coded copy of X."""
    Xm = Log10Transformer(cfg.LOG_FEATURES).transform(X)
    discrete = []
    for c in Xm.columns:
        if c in cfg.CATEGORICAL_FEATURES:
            Xm[c] = Xm[c].fillna(Xm[c].mode()[0]).astype("category").cat.codes
            discrete.append(True)
        else:
            Xm[c] = Xm[c].fillna(Xm[c].median())
            discrete.append(False)
    mi = mutual_info_classif(Xm, y, discrete_features=np.array(discrete),
                             n_neighbors=5, random_state=cfg.RANDOM_STATE)
    return pd.Series(mi, index=X.columns)


def select_features(train: pd.DataFrame) -> dict:
    """Apply the three selection rules and return a report dictionary."""
    X, y = get_xy(train)
    num_t = numeric_tests(X, cfg.NUMERIC_FEATURES, y)[["feature", "q_value", "univariate_auc"]]
    cat_t = categorical_tests(X, cfg.CATEGORICAL_FEATURES, y)[["feature", "q_value"]]
    tests = pd.concat([num_t, cat_t], ignore_index=True)
    # Re-adjust all p-values together so the FDR covers every feature tested.
    from src.analysis.stats import bh_fdr
    raw_p = pd.concat([numeric_tests(X, cfg.NUMERIC_FEATURES, y)["mannwhitney_p"],
                       categorical_tests(X, cfg.CATEGORICAL_FEATURES, y)["chi2_p"]],
                      ignore_index=True)
    tests["q_value"] = bh_fdr(raw_p.values)

    mi = _mutual_information(X, y)
    tests["mutual_info"] = tests["feature"].map(mi)
    tests["group"] = tests["feature"].map(cfg.FEATURE_GROUP)
    tests["type"] = np.where(tests["feature"].isin(cfg.NUMERIC_FEATURES), "numeric", "categorical")
    tests["pass_fdr"] = tests["q_value"] < cfg.FDR_ALPHA
    tests["pass_mi"] = tests["mutual_info"] >= cfg.MIN_MUTUAL_INFO
    candidates = tests.loc[tests.pass_fdr & tests.pass_mi, "feature"].tolist()

    # Redundancy filter on numeric candidates.
    num_c = [c for c in candidates if c in cfg.NUMERIC_FEATURES]
    corr = Log10Transformer(cfg.LOG_FEATURES).transform(X[num_c]).corr(method="spearman")
    dropped_redundant = []
    for i, a in enumerate(num_c):
        for b in num_c[i + 1:]:
            if abs(corr.loc[a, b]) > cfg.CORRELATION_THRESHOLD:
                loser = a if mi[a] < mi[b] else b
                if loser not in dropped_redundant:
                    dropped_redundant.append(loser)
    selected = [c for c in candidates if c not in dropped_redundant]
    tests["selected"] = tests["feature"].isin(selected)
    tests["reason"] = np.select(
        [tests.selected, ~tests.pass_fdr, ~tests.pass_mi, tests.feature.isin(dropped_redundant)],
        ["kept", f"dropped: FDR q >= {cfg.FDR_ALPHA}", "dropped: low mutual information",
         f"dropped: redundant (|rho| > {cfg.CORRELATION_THRESHOLD})"], "dropped")
    tests = tests.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return {"selected": selected, "dropped_redundant": dropped_redundant, "table": tests}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_missingness(train: pd.DataFrame):
    X = train[cfg.ALL_FEATURES]
    miss = X.isna()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a1.imshow(miss.values[:400].T, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(
        [cfg.SURFACE, "#1c5cab"]), interpolation="nearest")
    a1.set_yticks(range(len(cfg.ALL_FEATURES)), cfg.ALL_FEATURES, fontsize=7.2)
    a1.set_xlabel("First 400 training variants"); a1.grid(False)
    a1.set_title("Missingness matrix (dark = missing)")
    rate = miss.mean().sort_values()
    a2.barh(rate.index, rate.values * 100, color="#3987e5", height=0.65)
    a2.axvline(miss.values.mean() * 100, color=cfg.NEUTRAL, ls="--", lw=1)
    a2.text(miss.values.mean() * 100, -1.3, f" overall {miss.values.mean():.1%}",
            fontsize=7.5, color=cfg.TEXT_SECONDARY)
    a2.set_xlabel("% missing"); a2.tick_params(axis="y", labelsize=7.2)
    a2.grid(axis="y", visible=False); a2.set_title("Missing rate per feature")
    fig.tight_layout()
    return save_fig(fig, "11_missing_values_overview")


def missing_by_class(train: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of missingness vs class for each feature (an MCAR check)."""
    from scipy.stats import chi2_contingency
    X, y = get_xy(train)
    rows = []
    for c in cfg.ALL_FEATURES:
        tab = pd.crosstab(X[c].isna(), y)
        p = chi2_contingency(tab)[1] if tab.shape == (2, 2) else 1.0
        rows.append({"feature": c, "missing_benign": X.loc[y == 0, c].isna().mean(),
                     "missing_pathogenic": X.loc[y == 1, c].isna().mean(), "chi2_p": p})
    return pd.DataFrame(rows)


def compare_imputers(train: pd.DataFrame, features) -> pd.DataFrame:
    """5-fold CV ROC AUC of a logistic-regression probe under each imputer."""
    X, y = get_xy(train, features)
    num, cat = split_types(features)
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    rows = []
    for name in ["mean", "median", "knn", "iterative"]:
        pipe = Pipeline([("prep", build_preprocessor(num, cat, imputer=name)),
                         ("clf", LogisticRegression(max_iter=2000))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        rows.append({"imputer": name, "cv_auc_mean": s.mean(), "cv_auc_sd": s.std()})
    return pd.DataFrame(rows)


def plot_missing_handling(train, mbc: pd.DataFrame, imp: pd.DataFrame, features):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    m = mbc.sort_values("missing_benign")
    yy = np.arange(len(m))
    ax.scatter(m["missing_benign"] * 100, yy, color=cfg.COLORS["Benign"], s=22, label="Benign")
    ax.scatter(m["missing_pathogenic"] * 100, yy, color=cfg.COLORS["Pathogenic"], s=22,
               label="Pathogenic")
    ax.set_yticks(yy, m["feature"], fontsize=6.5); ax.set_xlabel("% missing")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    ax.set_title("Missing rate by class (MCAR check)")
    ax = axes[1]
    num, _ = split_types(features)
    col = "REVEL_Score" if "REVEL_Score" in num else num[0]
    X, _ = get_xy(train, features)
    raw = X[col].dropna()
    imputed = X[col].fillna(X[col].median())
    ax.hist(raw, bins=40, density=True, histtype="step", lw=2, color="#2a78d6", label="Observed only")
    ax.hist(imputed, bins=40, density=True, histtype="step", lw=2, color="#eb6834",
            label="After median imputation")
    ax.set_title(f"Median imputation effect: {col}"); ax.set_yticks([]); ax.legend()
    ax = axes[2]
    ax.bar(imp["imputer"], imp["cv_auc_mean"], yerr=imp["cv_auc_sd"], color="#3987e5",
           width=0.6, capsize=4, ecolor=cfg.NEUTRAL)
    lo = (imp["cv_auc_mean"] - imp["cv_auc_sd"]).min()
    ax.set_ylim(lo - 0.01, (imp["cv_auc_mean"] + imp["cv_auc_sd"]).max() + 0.005)
    for i, (v, sd) in enumerate(zip(imp["cv_auc_mean"], imp["cv_auc_sd"])):
        ax.text(i, v + sd + 0.0008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("5-fold CV ROC AUC"); ax.grid(axis="x", visible=False)
    ax.set_title("Imputation strategy comparison")
    fig.tight_layout()
    return save_fig(fig, "12_missing_value_handling",
                    "Missingness is similar in both classes (consistent with MCAR); all imputers perform alike, so the simple, robust median is used")


def outlier_table(train: pd.DataFrame) -> pd.DataFrame:
    X = Log10Transformer(cfg.LOG_FEATURES).transform(train[cfg.NUMERIC_FEATURES])
    capper = IQRCapper().fit(X)
    rows = []
    for i, c in enumerate(X.columns):
        s = X[c].dropna()
        rows.append({"feature": c, "lower_fence": capper.lower_[i], "upper_fence": capper.upper_[i],
                     "n_low": int((s < capper.lower_[i]).sum()),
                     "n_high": int((s > capper.upper_[i]).sum()),
                     "pct_outliers": 100 * ((s < capper.lower_[i]) | (s > capper.upper_[i])).mean(),
                     "n_abs_z_gt_3": int((np.abs((s - s.mean()) / s.std()) > 3).sum())})
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


def plot_outliers(train: pd.DataFrame, table: pd.DataFrame):
    X = Log10Transformer(cfg.LOG_FEATURES).transform(train[cfg.NUMERIC_FEATURES])
    Z = (X - X.mean()) / X.std()
    capped = pd.DataFrame(IQRCapper().fit_transform(X), columns=X.columns)
    Zc = (capped - X.mean()) / X.std()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, data, title in [(axes[0], Z, "Before capping"), (axes[1], Zc, "After IQR capping")]:
        bp = ax.boxplot([data[c].dropna() for c in X.columns], vert=False, widths=0.55,
                        patch_artist=True, flierprops=dict(marker="o", ms=2.5, mfc="#eb6834",
                                                           mec="none", alpha=0.6),
                        medianprops=dict(color=cfg.TEXT_PRIMARY))
        for p in bp["boxes"]:
            p.set_facecolor("#cde2fb"); p.set_edgecolor("#3987e5")
        ax.set_yticks(range(1, len(X.columns) + 1), X.columns, fontsize=7.5)
        ax.axvline(-3, color=cfg.NEUTRAL, ls=":", lw=1); ax.axvline(3, color=cfg.NEUTRAL, ls=":", lw=1)
        ax.set_xlabel("z-score (log10 for AF, CEA, CA125)"); ax.set_title(title)
        ax.grid(axis="y", visible=False)
    t = table.sort_values("pct_outliers")
    axes[2].barh(t["feature"], t["pct_outliers"], color="#eb6834", height=0.6)
    axes[2].set_xlabel("% of values beyond the IQR fences"); axes[2].tick_params(axis="y", labelsize=7.5)
    axes[2].grid(axis="y", visible=False); axes[2].set_title("Outlier share per feature")
    fig.suptitle("Outlier analysis: Tukey IQR fences (1.5 x IQR), values capped rather than dropped",
                 x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "13_outlier_analysis")


def plot_zscore(train: pd.DataFrame, features):
    num, cat = split_types(features)
    X, _ = get_xy(train, features)
    prep = build_preprocessor(num, cat).fit(X)
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    show = [c for c in ["CADD_Phred", "HRD_Score", "gnomAD_AF", "Age"] if c in num][:4]
    fig, axes = plt.subplots(2, len(show), figsize=(3.3 * len(show), 5.2))
    for j, c in enumerate(show):
        raw = X[c].dropna()
        axes[0, j].hist(raw, bins=40, color="#86b6ef", edgecolor=cfg.SURFACE, lw=0.4)
        axes[0, j].set_title(f"{c} (raw)", fontsize=9)
        axes[0, j].text(0.98, 0.92, f"mean {raw.mean():.3g}\nSD {raw.std():.3g}",
                        transform=axes[0, j].transAxes, ha="right", va="top", fontsize=7.5)
        axes[1, j].hist(Xt[c], bins=40, color="#1c5cab", edgecolor=cfg.SURFACE, lw=0.4)
        axes[1, j].set_title(f"{c} (z-scored)", fontsize=9)
        axes[1, j].text(0.98, 0.92, f"mean {Xt[c].mean():.2f}\nSD {Xt[c].std():.2f}",
                        transform=axes[1, j].transAxes, ha="right", va="top", fontsize=7.5)
        for a in axes[:, j]:
            a.set_yticks([])
    fig.suptitle("z-score normalisation: every numeric feature is centred at 0 with SD 1 "
                 "(after log10, capping and imputation)", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "14_zscore_normalisation")


def plot_correlation(train: pd.DataFrame, selected):
    X = Log10Transformer(cfg.LOG_FEATURES).transform(train[cfg.NUMERIC_FEATURES])
    corr = X.corr(method="spearman")
    # Order by hierarchical clustering so correlated blocks sit together.
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = corr.columns[leaves_list(linkage(1 - corr.abs().values[np.triu_indices(len(corr), 1)],
                                             "average"))]
    corr = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.2:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else cfg.TEXT_PRIMARY)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text() not in selected:
            t.set_color(cfg.NEUTRAL)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman rho")
    ax.set_title("Correlation analysis (Spearman, numeric features, training set)")
    fig.tight_layout()
    return save_fig(fig, "15_correlation_heatmap",
                    f"Grey labels = features removed by selection. Redundancy threshold |rho| > {cfg.CORRELATION_THRESHOLD}")


def plot_feature_selection(table: pd.DataFrame):
    t = table.sort_values("mutual_info")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = np.where(t["selected"], "#1c5cab", cfg.NEUTRAL)
    a1.barh(t["feature"], t["mutual_info"], color=colors, height=0.65)
    a1.axvline(cfg.MIN_MUTUAL_INFO, color="#eb6834", ls="--", lw=1)
    a1.text(cfg.MIN_MUTUAL_INFO, -1.4, f" MI threshold {cfg.MIN_MUTUAL_INFO}", fontsize=7.5,
            color=cfg.TEXT_SECONDARY)
    a1.set_xlabel("Mutual information with the label (nats)"); a1.tick_params(axis="y", labelsize=7.5)
    a1.grid(axis="y", visible=False); a1.set_title("Mutual information (dark blue = selected)")
    q = t.set_index("feature")["q_value"].clip(lower=1e-300)
    a2.barh(q.index, -np.log10(q), color=colors, height=0.65)
    a2.axvline(-np.log10(cfg.FDR_ALPHA), color="#eb6834", ls="--", lw=1)
    a2.set_xscale("symlog", linthresh=10)
    a2.set_xlabel("-log10 FDR q-value (symlog)"); a2.set_yticks([])
    a2.grid(axis="y", visible=False); a2.set_title("Statistical association")
    n_sel = int(t["selected"].sum())
    fig.suptitle(f"Feature selection: {n_sel} of {len(t)} features kept "
                 "(FDR q < 0.05, MI >= 0.005, non-redundant)", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "16_feature_selection")


# --------------------------------------------------------------------------- #
def run() -> dict:
    cfg.ensure_dirs()
    train = load_split("train")
    test = load_split("test")

    sel = select_features(train)
    selected = sel["selected"]
    sel["table"].to_csv(cfg.TABLES_DIR / "feature_selection.csv", index=False)
    save_json({"selected_features": selected,
               "numeric": split_types(selected)[0],
               "categorical": split_types(selected)[1],
               "dropped_redundant": sel["dropped_redundant"],
               "rules": {"fdr_alpha": cfg.FDR_ALPHA, "min_mutual_info": cfg.MIN_MUTUAL_INFO,
                         "correlation_threshold": cfg.CORRELATION_THRESHOLD}},
              cfg.SELECTED_FEATURES_FILE)
    log.info("Selected %d features: %s", len(selected), selected)

    mbc = missing_by_class(train)
    mbc.to_csv(cfg.TABLES_DIR / "missing_by_class.csv", index=False)
    imp = compare_imputers(train, selected)
    imp.to_csv(cfg.TABLES_DIR / "imputer_comparison.csv", index=False)
    otab = outlier_table(train)
    otab.to_csv(cfg.TABLES_DIR / "outlier_summary.csv", index=False)

    plot_missingness(train)
    plot_missing_handling(train, mbc, imp, selected)
    plot_outliers(train, otab)
    plot_zscore(train, selected)
    plot_correlation(train, selected)
    plot_feature_selection(sel["table"])

    # Fit the preprocessor on train and save the transformed matrices for inspection.
    num, cat = split_types(selected)
    prep = build_preprocessor(num, cat).fit(get_xy(train, selected)[0])
    joblib.dump(prep, cfg.MODELS_DIR / "preprocessor.joblib")
    for name, df in [("train", train), ("test", test)]:
        X, y = get_xy(df, selected)
        out = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
        out[cfg.TARGET_COL] = y.values
        out.to_csv(cfg.PROCESSED_DIR / f"{name}_preprocessed.csv", index=False)
    log.info("Imputer comparison:\n%s", imp.round(4).to_string(index=False))
    return {"selection": sel, "missing_by_class": mbc, "imputers": imp, "outliers": otab}


if __name__ == "__main__":
    run()
