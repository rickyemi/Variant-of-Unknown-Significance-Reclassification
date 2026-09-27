"""
Stage 6: stress tests - how robust, stable and trustworthy are the models?

Test                         What it checks                              Pass rule
---------------------------  ------------------------------------------  ---------------------------------
1  Bootstrap CI              uncertainty of Test metrics (1000 draws)    AUC 95% CI lower bound >= 0.90
2  Missing-data injection    extra MCAR missingness at inference         AUC drop at +30% missing <= 0.05
3  Gaussian noise injection  measurement noise on numeric inputs         AUC drop at 0.25 SD noise <= 0.05
4  Prevalence shift          PPV / NPV when Pathogenic prevalence moves  reported (no pass rule)
5  Subgroup performance      AUC by sex, ethnicity, age band, cancer     no subgroup CI entirely below overall AUC
6  Label-noise robustness    retrain with 5 / 10 / 20% flipped labels    AUC drop at 10% flips <= 0.03
7  Permutation test          refit on shuffled labels (null AUC)         null ~0.5, real >= max null + 0.30
8  Learning curve            Train/Test AUC vs training-set size         final Train-Test AUC gap <= 0.05
9  CV stability              5-fold CV AUC spread                        fold AUC SD <= 0.02
10 Feature knockout          top-3 permutation features removed         reported (no pass rule)
11 Input edge cases          all-missing rows, extreme values, unseen    probabilities finite, in [0, 1],
                             categories, repeat calls                    deterministic

Outputs: reports/tables/stress_*.csv and reports/figures/25-34_*.png
Run:     python -m src.models.stress_test          (about 5 minutes on 2 CPUs)
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as cfg
from src.data.make_dataset import get_xy, load_split
from src.models.evaluate_model import HEADLINE, MODEL_ORDER, compute_metrics
from src.models.train_model import load_models
from src.utils import get_logger, set_seed
from src.visualization.visualize import save_fig

log = get_logger(__name__)
RNG = np.random.default_rng(cfg.RANDOM_STATE)


def _auc(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def _refit(model, X, y):
    m = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


# --------------------------------------------------------------------------- #
# 1. Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(models, meta, X, y, n_boot=1000):
    y = y.values
    rows = []
    for m in MODEL_ORDER:
        p = models[m].predict_proba(X)[:, 1]
        thr = meta["models"][m]["decision_threshold"]
        draws = []
        for _ in range(n_boot):
            idx = RNG.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                continue
            draws.append({k: v for k, v in compute_metrics(y[idx], p[idx], thr).items() if k in HEADLINE})
        d = pd.DataFrame(draws)
        point = compute_metrics(y, p, thr)
        for met in HEADLINE:
            rows.append({"Model": m, "Metric": met, "estimate": point[met],
                         "ci_low": d[met].quantile(0.025), "ci_high": d[met].quantile(0.975)})
    return pd.DataFrame(rows)


def plot_bootstrap(ci):
    fig, ax = plt.subplots(figsize=(10, 5))
    yy = np.arange(len(HEADLINE)); h = 0.25
    for i, m in enumerate(MODEL_ORDER):
        d = ci[ci.Model == m].set_index("Metric").loc[HEADLINE]
        pos = yy + (1 - i) * h
        ax.hlines(pos, d.ci_low, d.ci_high, color=cfg.MODEL_COLORS[m], lw=2.2)
        ax.scatter(d.estimate, pos, color=cfg.MODEL_COLORS[m], s=36, zorder=3, label=m)
    ax.set_yticks(yy, HEADLINE); ax.invert_yaxis()
    ax.set_xlabel("Test-set value with 95% bootstrap CI (1000 resamples)")
    ax.grid(axis="y", visible=False); ax.legend(loc="lower left")
    ax.set_title("Stress test 1: bootstrap confidence intervals")
    fig.tight_layout()
    return save_fig(fig, "25_stress_bootstrap_ci")


# --------------------------------------------------------------------------- #
# 2-3. Missing-data and noise injection at inference time
# --------------------------------------------------------------------------- #
def missing_injection(models, X, y, rates=(0, .1, .2, .3, .4, .5), reps=5):
    rows = []
    for r in rates:
        for rep in range(reps):
            Xm = X.copy()
            mask = RNG.random(X.shape) < r
            Xm = Xm.mask(mask)
            for m in MODEL_ORDER:
                rows.append({"Model": m, "extra_missing": r, "rep": rep, "AUC": _auc(models[m], Xm, y)})
    return pd.DataFrame(rows)


def noise_injection(models, X, y, Xtrain, levels=(0, .1, .25, .5, .75, 1.0), reps=5):
    num = [c for c in X.columns if c in cfg.NUMERIC_FEATURES]
    sd = Xtrain[num].std()
    rows = []
    for lvl in levels:
        for rep in range(reps):
            Xn = X.copy()
            noise = RNG.normal(0, 1, (len(X), len(num))) * sd.values * lvl
            Xn[num] = Xn[num] + noise
            # keep bounded scores physically valid
            for c in ["REVEL_Score", "SIFT_Score", "PolyPhen2_Score", "SpliceAI_Score",
                      "Variant_Allele_Freq", "gnomAD_AF"]:
                if c in Xn:
                    Xn[c] = Xn[c].clip(1e-8, 1)
            for m in MODEL_ORDER:
                rows.append({"Model": m, "noise_sd": lvl, "rep": rep, "AUC": _auc(models[m], Xn, y)})
    return pd.DataFrame(rows)


def plot_degradation(miss, noise):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, d, col, xl, t in [
            (a1, miss, "extra_missing", "Extra share of cells set to missing", "Missing-data injection"),
            (a2, noise, "noise_sd", "Added Gaussian noise (x training SD)", "Measurement-noise injection")]:
        for m in MODEL_ORDER:
            g = d[d.Model == m].groupby(col)["AUC"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=m)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=cfg.MODEL_COLORS[m], alpha=0.15, lw=0)
        ax.set_xlabel(xl); ax.set_ylabel("Test ROC AUC"); ax.legend()
        ax.set_title(t)
    fig.suptitle("Stress tests 2-3: graceful degradation under corrupted inputs", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "26_stress_missing_and_noise")


# --------------------------------------------------------------------------- #
# 4. Prevalence shift
# --------------------------------------------------------------------------- #
def prevalence_shift(models, meta, X, y, prevalences=(.05, .1, .2, .3, .4, .5), reps=200, n=600):
    y = y.values
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    probs = {m: models[m].predict_proba(X)[:, 1] for m in MODEL_ORDER}
    for prev in prevalences:
        k = int(round(prev * n))
        for _ in range(reps):
            idx = np.concatenate([RNG.choice(pos, k, replace=True), RNG.choice(neg, n - k, replace=True)])
            for m in MODEL_ORDER:
                met = compute_metrics(y[idx], probs[m][idx], meta["models"][m]["decision_threshold"])
                rows.append({"Model": m, "prevalence": prev, "PPV": met["PPV"], "NPV": met["NPV"],
                             "Sensitivity": met["Sensitivity"], "Specificity": met["Specificity"]})
    return pd.DataFrame(rows).groupby(["Model", "prevalence"]).mean().reset_index()


def plot_prevalence(prev):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met in [(a1, "PPV"), (a2, "NPV")]:
        for m in MODEL_ORDER:
            d = prev[prev.Model == m]
            ax.plot(d.prevalence, d[met], marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=m)
        ax.axvline(0.222, color=cfg.NEUTRAL, ls="--", lw=1)
        ax.text(0.222, ax.get_ylim()[0], " study prevalence", fontsize=7.5, color=cfg.TEXT_SECONDARY,
                va="bottom")
        ax.set_xlabel("Pathogenic prevalence in the tested population"); ax.set_ylabel(met)
        ax.set_title(f"{met} vs prevalence"); ax.legend()
    fig.suptitle("Stress test 4: prevalence shift (sensitivity and specificity stay fixed; PPV and NPV move)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "27_stress_prevalence_shift")


# --------------------------------------------------------------------------- #
# 5. Subgroup performance
# --------------------------------------------------------------------------- #
def subgroup_performance(models, meta, test, feats):
    X, y = get_xy(test, feats)
    df = test.copy()
    df["Age_band"] = pd.cut(df["Age"], [0, 45, 65, 120], labels=["<45", "45-65", ">65"]).astype(object)
    rows = []
    for m in MODEL_ORDER:
        p = models[m].predict_proba(X)[:, 1]
        thr = meta["models"][m]["decision_threshold"]
        for col in ["Sex", "Ethnicity", "Age_band", "Cancer_Type"]:
            for g, idx in df.groupby(col).groups.items():
                idx = np.asarray(idx)
                if y.iloc[idx].nunique() < 2 or len(idx) < 30:
                    continue
                met = compute_metrics(y.iloc[idx], p[idx], thr)
                yy_, pp_ = y.values[idx], p[idx]
                boots = []
                for _ in range(500):
                    b = RNG.integers(0, len(idx), len(idx))
                    if yy_[b].min() != yy_[b].max():
                        boots.append(roc_auc_score(yy_[b], pp_[b]))
                rows.append({"Model": m, "attribute": col, "group": g, "n": len(idx),
                             "AUC": met["AUC"], "AUC_ci_low": np.quantile(boots, .025),
                             "AUC_ci_high": np.quantile(boots, .975),
                             "overall_AUC": roc_auc_score(y, p),
                             "Sensitivity": met["Sensitivity"],
                             "Specificity": met["Specificity"]})
    return pd.DataFrame(rows)


def plot_subgroups(sub):
    groups = sub[sub.Model == MODEL_ORDER[0]][["attribute", "group", "n"]].reset_index(drop=True)
    labels = [f"{a.replace('_', ' ')}: {g} (n={n})" for a, g, n in groups.values]
    fig, ax = plt.subplots(figsize=(9, 0.33 * len(labels) + 1.5))
    yy = np.arange(len(labels)); h = 0.26
    for i, m in enumerate(MODEL_ORDER):
        d = sub[sub.Model == m].reset_index(drop=True)
        ax.hlines(yy + (1 - i) * h, d.AUC_ci_low, d.AUC_ci_high, color=cfg.MODEL_COLORS[m], lw=1.4)
        ax.scatter(d.AUC, yy + (1 - i) * h, color=cfg.MODEL_COLORS[m], s=30, label=m, zorder=3)
    ax.set_yticks(yy, labels, fontsize=7.8); ax.invert_yaxis()
    ax.axvline(sub.overall_AUC.mean(), color=cfg.NEUTRAL, ls="--", lw=1)
    ax.set_xlabel("Test ROC AUC within subgroup (95% bootstrap CI); dashed = overall Test AUC"); ax.legend(loc="lower left")
    ax.set_title("Stress test 5: subgroup performance (demographic fairness check)")
    fig.tight_layout()
    return save_fig(fig, "28_stress_subgroup_performance")


# --------------------------------------------------------------------------- #
# 6-9. Retraining-based tests (fixed tuned hyperparameters, no re-search)
# --------------------------------------------------------------------------- #
def label_noise(models, Xtr, ytr, Xte, yte, rates=(0, .05, .10, .20)):
    rows = []
    for r in rates:
        yn = ytr.copy()
        flip = RNG.random(len(yn)) < r
        yn[flip] = 1 - yn[flip]
        for m in MODEL_ORDER:
            fitted = models[m] if r == 0 else _refit(models[m], Xtr, yn)
            rows.append({"Model": m, "flip_rate": r, "AUC": _auc(fitted, Xte, yte)})
    return pd.DataFrame(rows)


def y_scramble(models, Xtr, ytr, Xte, yte, n_perm=5):
    """Permutation (y-randomisation) test.

    For each permutation the TRAIN labels are shuffled, the model is refitted
    and scored against independently shuffled TEST labels. This builds the
    null distribution of Test AUC when there is no real signal; the real Test
    AUC should sit far above it.
    """
    rows = []
    for m in MODEL_ORDER:
        real = _auc(models[m], Xte, yte)
        for k in range(n_perm):
            ys = pd.Series(RNG.permutation(ytr.values), index=ytr.index)
            yt = pd.Series(RNG.permutation(yte.values), index=yte.index)
            rows.append({"Model": m, "perm": k, "AUC_real": real,
                         "AUC_null": _auc(_refit(models[m], Xtr, ys), Xte, yt)})
    return pd.DataFrame(rows)


def learning_curve(models, Xtr, ytr, Xte, yte, fracs=(.1, .25, .5, .75, 1.0)):
    rows = []
    for f in fracs:
        if f < 1:
            idx = (pd.DataFrame({"y": ytr}).groupby("y", group_keys=False)
                   .apply(lambda g: g.sample(frac=f, random_state=cfg.RANDOM_STATE)).index)
        else:
            idx = ytr.index
        for m in MODEL_ORDER:
            fitted = models[m] if f == 1 else _refit(models[m], Xtr.loc[idx], ytr.loc[idx])
            rows.append({"Model": m, "train_fraction": f, "n_train": len(idx),
                         "Train_AUC": _auc(fitted, Xtr.loc[idx], ytr.loc[idx]),
                         "Test_AUC": _auc(fitted, Xte, yte)})
    return pd.DataFrame(rows)


def cv_stability(models, Xtr, ytr):
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE + 1)
    rows = []
    for k, (a, b) in enumerate(cv.split(Xtr, ytr)):
        for m in MODEL_ORDER:
            fitted = _refit(models[m], Xtr.iloc[a], ytr.iloc[a])
            rows.append({"Model": m, "fold": k + 1, "AUC": _auc(fitted, Xtr.iloc[b], ytr.iloc[b])})
    return pd.DataFrame(rows)


def plot_retraining(ln, ys, lc, cvs):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    ax = axes[0]
    for m in MODEL_ORDER:
        d = ln[ln.Model == m]
        ax.plot(d.flip_rate * 100, d.AUC, marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=m)
    ax.set_xlabel("% training labels flipped"); ax.set_ylabel("Test ROC AUC")
    ax.set_title("6. Label-noise robustness"); ax.legend()
    ax = axes[1]
    x = np.arange(len(MODEL_ORDER))
    real = ys.groupby("Model").AUC_real.first().reindex(MODEL_ORDER)
    ax.bar(x, real, 0.55, color=[cfg.MODEL_COLORS[m] for m in MODEL_ORDER], label="Real labels")
    for i, m in enumerate(MODEL_ORDER):
        nul = ys[ys.Model == m].AUC_null
        ax.scatter(np.full(len(nul), i), nul, color=cfg.TEXT_PRIMARY, s=16, zorder=3,
                   label="Null (shuffled)" if i == 0 else None)
    ax.axhline(0.5, color=cfg.TEXT_SECONDARY, ls="--", lw=1)
    ax.set_xticks(x, MODEL_ORDER); ax.set_ylim(0.3, 1.12); ax.set_ylabel("Test ROC AUC")
    ax.grid(axis="x", visible=False); ax.legend(loc="upper center", ncol=2, fontsize=7.5)
    ax.set_title("7. Permutation test (signal is real)")
    ax = axes[2]
    for m in MODEL_ORDER:
        d = lc[lc.Model == m]
        ax.plot(d.n_train, d.Test_AUC, marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=f"{m} Test")
        ax.plot(d.n_train, d.Train_AUC, ls="--", lw=1.4, color=cfg.MODEL_COLORS[m])
    ax.set_xlabel("Training variants"); ax.set_ylabel("ROC AUC (dashed = Train)")
    ax.set_title("8. Learning curve"); ax.legend(fontsize=7.5)
    ax = axes[3]
    data = [cvs[cvs.Model == m].AUC for m in MODEL_ORDER]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color=cfg.TEXT_PRIMARY))
    for p, m in zip(bp["boxes"], MODEL_ORDER):
        p.set_facecolor(cfg.MODEL_COLORS[m]); p.set_alpha(0.6)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1), d, color=cfg.TEXT_PRIMARY, s=12, zorder=3)
    ax.set_xticks([1, 2, 3], MODEL_ORDER); ax.set_ylabel("Fold ROC AUC")
    ax.grid(axis="x", visible=False); ax.set_title("9. 5-fold CV stability")
    fig.suptitle("Stress tests 6-9: retraining-based robustness checks", x=0.01, ha="left")
    fig.tight_layout()
    return save_fig(fig, "29_stress_retraining_tests")


# --------------------------------------------------------------------------- #
# 10-11. Feature knockout and input edge cases
# --------------------------------------------------------------------------- #
def feature_knockout(models, Xte, yte):
    perm = pd.read_csv(cfg.TABLES_DIR / "permutation_importance.csv")
    top = perm.groupby("feature")["auc_drop_mean"].mean().sort_values(ascending=False).index[:3].tolist()
    rows = []
    for k in range(0, 4):
        Xk = Xte.copy()
        for f in top[:k]:
            Xk[f] = np.nan
        for m in MODEL_ORDER:
            rows.append({"Model": m, "features_removed": ", ".join(top[:k]) or "none",
                         "n_removed": k, "AUC": _auc(models[m], Xk, yte)})
    return pd.DataFrame(rows)


def plot_knockout(ko):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in MODEL_ORDER:
        d = ko[ko.Model == m]
        ax.plot(d.n_removed, d.AUC, marker="o", ms=5, color=cfg.MODEL_COLORS[m], label=m)
    labels = ko[ko.Model == MODEL_ORDER[0]].features_removed.tolist()
    ax.set_xticks(range(4), ["none"] + [f"+ {s.split(', ')[-1]}" for s in labels[1:]])
    ax.set_xlabel("Cumulatively removed (set to missing) at inference")
    ax.set_ylabel("Test ROC AUC"); ax.legend()
    ax.set_title("Stress test 10: knocking out the most important features")
    fig.tight_layout()
    return save_fig(fig, "30_stress_feature_knockout")


def edge_cases(models, Xte):
    cats = [c for c in Xte.columns if c in cfg.CATEGORICAL_FEATURES]
    nums = [c for c in Xte.columns if c in cfg.NUMERIC_FEATURES]
    cases = {}
    cases["all_missing_row"] = pd.DataFrame([{c: np.nan for c in Xte.columns}]).astype(Xte.dtypes.to_dict())
    ext = Xte.head(20).copy()
    ext[nums] = ext[nums] * 50
    cases["extreme_values_x50"] = ext
    neg = Xte.head(20).copy()
    neg[nums] = -neg[nums].abs() * 50
    for c in ["REVEL_Score", "SIFT_Score", "PolyPhen2_Score", "SpliceAI_Score", "gnomAD_AF",
              "Variant_Allele_Freq"]:
        if c in neg:
            neg[c] = 1e-9
    cases["extreme_negative_values"] = neg
    unseen = Xte.head(20).copy()
    for c in cats:
        unseen[c] = "UNSEEN_CATEGORY"
    cases["unseen_categories"] = unseen
    cases["single_row"] = Xte.head(1)
    rows = []
    for m in MODEL_ORDER:
        for name, Xc in cases.items():
            try:
                p = models[m].predict_proba(Xc)[:, 1]
                ok = bool(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)))
                msg = f"P range {p.min():.3f}-{p.max():.3f}"
            except Exception as e:  # noqa: BLE001 - any crash is a failed case
                ok, msg = False, f"error: {e}"
            rows.append({"Model": m, "case": name, "passed": ok, "detail": msg})
        p1 = models[m].predict_proba(Xte)[:, 1]
        p2 = models[m].predict_proba(Xte)[:, 1]
        rows.append({"Model": m, "case": "deterministic_repeat_call",
                     "passed": bool(np.allclose(p1, p2)), "detail": f"max diff {np.abs(p1 - p2).max():.2e}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def summarise(ci, miss, noise, sub, ln, ys, lc, cvs, edge) -> pd.DataFrame:
    rows = []
    for m in MODEL_ORDER:
        def add(test, value, passed, rule):
            rows.append({"Model": m, "Test": test, "Value": value, "Rule": rule,
                         "Result": "PASS" if passed else "FAIL"})
        lo = ci[(ci.Model == m) & (ci.Metric == "AUC")].ci_low.iloc[0]
        add("1 Bootstrap AUC CI lower bound", f"{lo:.3f}", lo >= 0.90, ">= 0.90")
        g = miss[miss.Model == m].groupby("extra_missing").AUC.mean()
        add("2 AUC drop at +30% missing", f"{g[0] - g[0.3]:.3f}", g[0] - g[0.3] <= 0.05, "<= 0.05")
        g = noise[noise.Model == m].groupby("noise_sd").AUC.mean()
        add("3 AUC drop at 0.25 SD noise", f"{g[0] - g[0.25]:.3f}", g[0] - g[0.25] <= 0.05, "<= 0.05")
        s = sub[sub.Model == m]
        # Fairness concern = a subgroup doing significantly WORSE than overall.
        k = int((s.AUC_ci_high >= s.overall_AUC).sum())
        add("5 No subgroup significantly below overall AUC", f"{k}/{len(s)}", k == len(s),
            "every subgroup 95% CI upper bound >= overall AUC")
        d = ln[ln.Model == m].set_index("flip_rate").AUC
        add("6 AUC drop at 10% label noise", f"{d[0] - d[0.10]:.3f}", d[0] - d[0.10] <= 0.03, "<= 0.03")
        d7 = ys[ys.Model == m]
        real, nmax, nmean = d7.AUC_real.iloc[0], d7.AUC_null.max(), d7.AUC_null.mean()
        add("7 Permutation test: real vs null AUC", f"{real:.3f} vs {nmean:.3f}",
            abs(nmean - 0.5) <= 0.05 and real - nmax >= 0.30, "null ~0.5; real >= max null + 0.30")
        r = lc[(lc.Model == m) & (lc.train_fraction == 1.0)].iloc[0]
        add("8 Final Train-Test AUC gap", f"{r.Train_AUC - r.Test_AUC:.3f}",
            abs(r.Train_AUC - r.Test_AUC) <= 0.05, "|gap| <= 0.05")
        sd = cvs[cvs.Model == m].AUC.std()
        add("9 CV fold AUC SD", f"{sd:.3f}", sd <= 0.02, "<= 0.02")
        e = edge[edge.Model == m]
        add("11 Input edge cases", f"{e.passed.sum()}/{len(e)} passed", e.passed.all(), "all pass")
    return pd.DataFrame(rows)


def plot_summary(summary):
    tests = summary.Test.unique()
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(tests) + 1.4))
    ax.set_xlim(-0.5, len(MODEL_ORDER) + 0.5); ax.set_ylim(len(tests) - 0.5, -0.5)
    for i, t in enumerate(tests):
        for j, m in enumerate(MODEL_ORDER):
            r = summary[(summary.Test == t) & (summary.Model == m)].iloc[0]
            ok = r.Result == "PASS"
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                                       color="#d6ecdf" if ok else "#f7d4d4", lw=0))
            ax.text(j, i, f"{'PASS' if ok else 'FAIL'}  {r.Value}", ha="center", va="center",
                    fontsize=8, color="#0b5d36" if ok else "#8f1d1d", fontweight="bold")
        ax.text(len(MODEL_ORDER) - 0.4, i, summary[summary.Test == t].Rule.iloc[0], va="center",
                fontsize=7.5, color=cfg.TEXT_SECONDARY)
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER); ax.xaxis.tick_top()
    ax.set_yticks(range(len(tests)), tests, fontsize=8.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Stress-test scorecard (rule shown on the right)", pad=26)
    fig.tight_layout()
    return save_fig(fig, "31_stress_test_scorecard")


def run() -> dict:
    cfg.ensure_dirs()
    set_seed(cfg.RANDOM_STATE)
    models, meta = load_models()
    feats = meta["features"]
    train, test = load_split("train"), load_split("test")
    Xtr, ytr = get_xy(train, feats)
    Xte, yte = get_xy(test, feats)

    log.info("1/11 bootstrap CIs"); ci = bootstrap_ci(models, meta, Xte, yte)
    log.info("2/11 missing injection"); miss = missing_injection(models, Xte, yte)
    log.info("3/11 noise injection"); noise = noise_injection(models, Xte, yte, Xtr)
    log.info("4/11 prevalence shift"); prev = prevalence_shift(models, meta, Xte, yte)
    log.info("5/11 subgroups"); sub = subgroup_performance(models, meta, test, feats)
    log.info("6/11 label noise (retraining)"); ln = label_noise(models, Xtr, ytr, Xte, yte)
    log.info("7/11 y-scrambling (retraining)"); ys = y_scramble(models, Xtr, ytr, Xte, yte)
    log.info("8/11 learning curve (retraining)"); lc = learning_curve(models, Xtr, ytr, Xte, yte)
    log.info("9/11 CV stability (retraining)"); cvs = cv_stability(models, Xtr, ytr)
    log.info("10/11 feature knockout"); ko = feature_knockout(models, Xte, yte)
    log.info("11/11 edge cases"); edge = edge_cases(models, Xte)

    for name, d in [("bootstrap_ci", ci), ("missing_injection", miss), ("noise_injection", noise),
                    ("prevalence_shift", prev), ("subgroups", sub), ("label_noise", ln),
                    ("y_scramble", ys), ("learning_curve", lc), ("cv_stability", cvs),
                    ("feature_knockout", ko), ("edge_cases", edge)]:
        d.to_csv(cfg.TABLES_DIR / f"stress_{name}.csv", index=False)

    plot_bootstrap(ci); plot_degradation(miss, noise); plot_prevalence(prev)
    plot_subgroups(sub); plot_retraining(ln, ys, lc, cvs); plot_knockout(ko)
    summary = summarise(ci, miss, noise, sub, ln, ys, lc, cvs, edge)
    summary.to_csv(cfg.TABLES_DIR / "stress_test_summary.csv", index=False)
    plot_summary(summary)
    log.info("Stress-test scorecard:\n%s", summary.pivot(index="Test", columns="Model",
                                                           values="Result").to_string())
    return {"summary": summary, "bootstrap": ci, "missing": miss, "noise": noise,
            "prevalence": prev, "subgroups": sub, "label_noise": ln, "y_scramble": ys,
            "learning_curve": lc, "cv": cvs, "knockout": ko, "edge": edge}


if __name__ == "__main__":
    run()
