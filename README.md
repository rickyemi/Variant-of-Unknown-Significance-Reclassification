[README.md](https://github.com/user-attachments/files/32691508/README.md)
# Reclassifying Clinical Variants of Uncertain Significance (VUS) with Machine Learning

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20SVM--RBF%20%7C%20FT--Transformer-orange)

 **Status:** research prototype on synthetic data

## Goal
About 10% of clinically reported cancer variants are **VUS**, which can't guide treatment or cascade testing. This project builds a reproducible, production-ready pipeline that learns from confidently labelled variants and reclassifies each VUS as **Pathogenic** or **Benign**, with a probability, a confidence tier and model agreement. The target was ≥ 88–92% accuracy on both Train and Test without overfitting.

## Data
| Item | Value |
|---|---|
| Source | Synthetic cohort (`src/data/generate_synthetic_data.py`, seed 42); no real patients |
| Size | 3,410 variants × 30 columns; **13% of cells missing** at random |
| Features | 8 demographic, 12 genomic (variant consequence, gene, CADD, REVEL, SIFT, PolyPhen-2, SpliceAI, GERP, PhyloP, gnomAD AF), 8 tumour/clinical biomarkers (VAF, TMB, Ki-67, PD-L1, CEA, CA125, HRD, read depth) |
| Target | Benign 2,387 (70%) · Pathogenic 682 (20%) · VUS 341 (10%) |
| Split | Labelled rows split 80/20, stratified: **Train 2,455 / Test 614**; the 341 VUS are held out for reclassification |

## Methodology
| Stage | What was done |
|---|---|
| **Biomarker & genomic analysis** | Mann-Whitney U tests with Benjamini-Hochberg FDR, rank-biserial effect sizes and single-feature AUC; chi-square with Cramér's V for demographics; Fisher's exact enrichment (odds ratios) for variant consequence; gene × consequence pathogenic rates; in-silico predictor concordance; allele frequency against the ACMG BA1/BS1/PM2 thresholds |
| **Preprocessing** (fitted on Train only) | log10 of skewed features → **outliers** capped at Tukey IQR fences (1.5 × IQR) → **missing values** filled with the median (numeric) or the most frequent level (categorical), chosen after comparing mean, median, KNN and iterative imputers by CV AUC (MCAR check passed) → **z-score** scaling → one-hot encoding |
| **Feature selection** | Keep if FDR q < 0.05 **and** mutual information ≥ 0.005, then drop redundant pairs (abs Spearman rho > 0.85). **16 of 28 features kept**; all 10 designed-noise columns plus CEA and CA125 removed |
| **Model development** | **XGBoost** (RandomizedSearchCV, 40 candidates × 5 folds, 9 hyperparameters) · **RBF-SVM** (RandomizedSearchCV, 30 × 5; C, gamma, class weight; Platt probabilities) · **FT-Transformer** (PyTorch feature-token self-attention, 2 layers × 4 heads, early stopping). Overfitting guard: best CV AUC among candidates with train-to-validation gap ≤ 0.03. Decision threshold: Youden's J on out-of-fold *training* predictions |
| **Evaluation** | Accuracy, sensitivity, specificity, PPV, NPV, AUC, F1 (plus MCC, Brier) on Train and Test; ROC/PR curves, confusion matrices, calibration, XGBoost gain importance and permutation importance for all 3 models |
| **Stress tests** | Bootstrap CIs, missing-data and noise injection, prevalence shift, subgroup fairness, label noise, permutation test, learning curve, CV stability, feature knockout, input edge cases |

## Results
**Train vs Test performance** (Pathogenic = positive class)

| Metric | XGBoost Train | XGBoost Test | SVM-RBF Train | SVM-RBF Test | Transformer Train | Transformer Test |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.913 | 0.907 | 0.903 | 0.912 | 0.886 | 0.896 |
| Sensitivity | 0.940 | 0.912 | 0.912 | 0.912 | 0.930 | 0.934 |
| Specificity | 0.906 | 0.906 | 0.901 | 0.912 | 0.874 | 0.885 |
| PPV | 0.740 | 0.734 | 0.725 | 0.747 | 0.678 | 0.698 |
| NPV | 0.981 | 0.973 | 0.973 | 0.973 | 0.978 | 0.979 |
| AUC | 0.953 | 0.947 | 0.934 | 0.950 | 0.929 | 0.952 |
| F1 | 0.828 | 0.813 | 0.808 | 0.821 | 0.785 | 0.799 |

* Train and Test agree within 0.03 for every metric, so there is **no overfitting**. Accuracy is 0.89–0.91 and AUC 0.93–0.95.
* **SVM-RBF is the most balanced model** (best Test accuracy, specificity and PPV). XGBoost is a close second, and the Transformer has the highest sensitivity.
* PPV (0.70–0.75) and F1 (0.80–0.83) are lower because only 22% of labelled variants are Pathogenic. PPV rises to about 0.91 at 50% prevalence (stress test 4).
* **Top predictors (all 3 models):** variant consequence, REVEL, PolyPhen-2, functional-domain location, CADD, HRD score, conservation and gnomAD AF.

**Stress tests: all 11 passed by all 3 models.** Test AUC 95% CI lower bound ≥ 0.924; AUC drop ≤ 0.02 with 30% extra missing data or 0.25 SD noise; no subgroup (sex, ethnicity, age, cancer type) significantly below the overall AUC; ≤ 0.006 AUC lost with 10% label noise; permutation test AUC ≈ 0.95 vs ≈ 0.50 null; fold AUC SD ≤ 0.02; safe outputs on all-missing, extreme and unseen-category inputs.

**VUS reclassification** (majority vote of 3 models; ensemble probability = mean)

| Reclassified as | High confidence | Medium | Low | Total |
|---|---:|---:|---:|---:|
| Benign | 152 | 38 | 1 | **191 (56%)** |
| Pathogenic | 11 | 54 | 85 | **150 (44%)** |

All three models agree on **300 of 341 VUS (88%)**. The 41 split votes and the Low-confidence calls are the priority list for expert review. Per-variant probabilities are in `reports/vus_reclassification.csv`.

> **Caveat:** the data are synthetic, so these results show that the pipeline works, not that it is clinically valid. Retrain on curated real data (e.g. ClinVar ≥ 2-star) before use. Outputs are decision support, not ACMG/AMP classification.

---

### Run it
```bash
make install        # pinned dependencies (CPU is enough)
make all            # data → analysis → features → train → evaluate → stress → predict (~6 min on 2 CPU cores)
make test           # 16 unit/integration tests
make notebooks      # execute the 5 notebooks in place
python -m src.models.predict_model --input new_variants.csv --output predictions.csv
docker build -t vus-reclassification . && docker run --rm -v "$PWD":/app vus-reclassification make all
```

### Repository layout
```
├── LICENSE · Makefile · README.md · requirements.txt · pyproject.toml
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
├── data/        raw/ (synthetic CSV) · interim/ · processed/ · external/
├── docs/        VUS_Reclassification_OnePager.docx
├── models/      xgboost_model.joblib · svm_rbf_model.joblib · transformer_model.joblib
│                preprocessor.joblib · model_metadata.json · MODEL_CARD.md
├── notebooks/   01 biomarker & genomic analysis · 02 preprocessing & feature selection
│                03 training & evaluation · 04 stress tests · 05 VUS reclassification
├── references/  data_dictionary.md · methods_references.md
├── reports/     figures/ (33 PNGs) · tables/ (CSV results) · vus_reclassification.csv
├── src/         config.py · pipeline.py · utils.py
│   ├── data/          make_dataset.py · generate_synthetic_data.py
│   ├── analysis/      biomarker_analysis.py · genomic_analysis.py · stats.py
│   ├── features/      build_features.py
│   ├── models/        train_model.py · transformer_model.py · evaluate_model.py
│   │                  stress_test.py · predict_model.py
│   └── visualization/ visualize.py
└── tests/       test_data.py · test_features.py · test_models.py
```
