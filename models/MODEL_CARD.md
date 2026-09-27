# Model card: VUS reclassification models

| Item | Detail |
|---|---|
| Files | `xgboost_model.joblib`, `svm_rbf_model.joblib`, `transformer_model.joblib` (each a full scikit-learn Pipeline: preprocessing + classifier); `preprocessor.joblib` (standalone preprocessing, for inspection); `model_metadata.json` (features, tuned hyperparameters, thresholds, library versions) |
| Task | Binary classification of germline/somatic cancer variants: **Pathogenic (1)** vs **Benign (0)**, applied to reclassify VUS |
| Input | The 16 selected raw columns listed in `model_metadata.json` (missing values allowed) |
| Output | `predict_proba(X)[:, 1]` = P(Pathogenic); call = P ≥ the model's `decision_threshold` |
| Training data | 2,455 labelled synthetic variants (80% stratified split); 614 held out for testing |
| Thresholds | Youden's J on out-of-fold training predictions (not tuned on test) |
| Versions | Python 3.11, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.14.0 (load with the pinned `requirements.txt`) |

## Loading
```python
import joblib, json, pandas as pd
model = joblib.load("models/xgboost_model.joblib")
meta = json.load(open("models/model_metadata.json"))
df = pd.read_csv("new_variants.csv", keep_default_na=False, na_values=[""])
p = model.predict_proba(df[meta["features"]])[:, 1]
call = p >= meta["models"]["XGBoost"]["decision_threshold"]
```
Loading `transformer_model.joblib` requires the project package on the path (`pip install -e .`), because it contains the custom `FTTransformerClassifier` class.

## Intended use
Research and decision support for prioritising VUS for expert curation. **Not** a medical device, and not a replacement for ACMG/AMP classification by a qualified variant scientist.

## Limitations
* Trained on **synthetic** data. The performance figures show the pipeline works; they are not clinical validity. Retrain and validate on real, curated data (e.g. ClinVar with review status ≥ 2 stars) before any real-world use.
* PPV depends on the prevalence of pathogenic variants in the population scored (see stress test 4).
* Probabilities from the Transformer are less calibrated than those from the SVM (see the calibration figure), so use its calls, or the ensemble, rather than its raw probability.
* Performance was checked across sex, ethnicity, age band and cancer type. No subgroup was significantly worse, but small subgroups (n < 50) have wide confidence intervals.
