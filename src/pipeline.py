"""
End-to-end pipeline runner (the same stages the Makefile calls).

    python -m src.pipeline                 # all stages
    python -m src.pipeline --skip-stress   # everything except the ~4 min stress suite
    python -m src.pipeline --stages data features train

Stages, in order:
    data      validate raw CSV, split train / test / VUS
    analysis  biomarker + genomic analysis (tables and figures 01-10)
    features  missing values, outliers, z-score, correlation, feature selection (11-16)
    train     XGBoost + RBF-SVM (RandomizedSearchCV) and FT-Transformer -> models/*.joblib
    evaluate  Train/Test metric table, ROC/PR, confusion, calibration, importance (19-24)
    stress    11 stress tests and the scorecard (25-31)
    predict   reclassify the VUS -> reports/vus_reclassification.csv (32-33)
"""
import argparse
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")   # headless: save figures, never open windows

from src.utils import get_logger  # noqa: E402

log = get_logger("pipeline")
STAGES = ["data", "analysis", "features", "train", "evaluate", "stress", "predict"]


def run_stage(name: str):
    if name == "data":
        from src.data.make_dataset import main
        main()
    elif name == "analysis":
        from src.analysis import biomarker_analysis, genomic_analysis
        biomarker_analysis.run()
        genomic_analysis.run()
    elif name == "features":
        from src.features.build_features import run
        run()
    elif name == "train":
        from src.models.train_model import run
        run()
    elif name == "evaluate":
        from src.models.evaluate_model import run
        run()
    elif name == "stress":
        from src.models.stress_test import run
        run()
    elif name == "predict":
        from src.models.predict_model import main
        main()


def main():
    p = argparse.ArgumentParser(description="VUS reclassification pipeline")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--skip-stress", action="store_true")
    a = p.parse_args()
    stages = [s for s in a.stages if not (a.skip_stress and s == "stress")]
    t0 = time.time()
    for s in stages:
        t = time.time()
        log.info("==== stage: %s ====", s)
        run_stage(s)
        log.info("stage %s finished in %.0fs", s, time.time() - t)
    log.info("Pipeline complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
