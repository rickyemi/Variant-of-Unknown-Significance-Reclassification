"""
Central configuration for the VUS reclassification project.

Every path, column group, random seed and plotting colour lives here so that
scripts, notebooks and tests all agree on the same settings. Paths are built
from this file's location, so the project works from any working directory
(repo root, notebooks/, Docker container, CI runner).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

RAW_FILE = RAW_DIR / "VUS_biomarker_cancer_mutation.csv"
INTERIM_FILE = INTERIM_DIR / "variants_validated.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
VUS_FILE = PROCESSED_DIR / "vus.csv"
SELECTED_FEATURES_FILE = PROCESSED_DIR / "selected_features.json"

MODEL_FILES = {
    "XGBoost": MODELS_DIR / "xgboost_model.joblib",
    "SVM-RBF": MODELS_DIR / "svm_rbf_model.joblib",
    "Transformer": MODELS_DIR / "transformer_model.joblib",
}
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
VUS_PREDICTIONS_FILE = REPORTS_DIR / "vus_reclassification.csv"

# --------------------------------------------------------------------------- #
# Reproducibility and split settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 / 20 stratified split of the labelled variants
CV_FOLDS = 5              # folds for RandomizedSearchCV and out-of-fold thresholds
N_ITER_XGB = 40           # RandomizedSearch candidates for XGBoost
N_ITER_SVM = 30           # RandomizedSearch candidates for the RBF-SVM
MAX_TRAIN_VAL_GAP = 0.03  # overfitting guard used when picking the best candidate

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ID_COL = "Variant_ID"
TARGET_COL = "Classification"
POSITIVE_CLASS = "Pathogenic"          # coded as 1
NEGATIVE_CLASS = "Benign"              # coded as 0
UNLABELLED_CLASS = "VUS"               # reclassified at the end
TARGET_LEVELS = [NEGATIVE_CLASS, POSITIVE_CLASS, UNLABELLED_CLASS]

DEMOGRAPHIC_NUMERIC = ["Age", "BMI"]
DEMOGRAPHIC_CATEGORICAL = [
    "Sex", "Ethnicity", "Smoking_Status", "Alcohol_Use",
    "Family_History_Cancer", "Cancer_Type",
]
GENOMIC_NUMERIC = [
    "Exon_Number", "CADD_Phred", "REVEL_Score", "SIFT_Score", "PolyPhen2_Score",
    "SpliceAI_Score", "GERP_RS", "PhyloP_100way", "gnomAD_AF",
]
GENOMIC_CATEGORICAL = ["Gene", "Variant_Type", "In_Functional_Domain"]
BIOMARKER_NUMERIC = [
    "Variant_Allele_Freq", "Read_Depth", "Tumor_Mutational_Burden", "Ki67_Index",
    "PD_L1_TPS", "CEA_ng_mL", "CA125_U_mL", "HRD_Score",
]

NUMERIC_FEATURES = DEMOGRAPHIC_NUMERIC + GENOMIC_NUMERIC + BIOMARKER_NUMERIC
CATEGORICAL_FEATURES = DEMOGRAPHIC_CATEGORICAL + GENOMIC_CATEGORICAL
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Right-skewed features that are log10-transformed before scaling.
LOG_FEATURES = ["gnomAD_AF", "CEA_ng_mL", "CA125_U_mL"]

FEATURE_GROUP = {
    **{c: "Demography" for c in DEMOGRAPHIC_NUMERIC + DEMOGRAPHIC_CATEGORICAL},
    **{c: "Genomic" for c in GENOMIC_NUMERIC + GENOMIC_CATEGORICAL},
    **{c: "Biomarker" for c in BIOMARKER_NUMERIC},
}

# --------------------------------------------------------------------------- #
# Feature-selection thresholds
# --------------------------------------------------------------------------- #
FDR_ALPHA = 0.05              # Benjamini-Hochberg q-value cut-off
MIN_MUTUAL_INFO = 0.005       # features must carry at least this much information
CORRELATION_THRESHOLD = 0.85  # |Spearman rho| above this marks a redundant pair

# --------------------------------------------------------------------------- #
# Plot style (validated colour-blind-safe categorical palette, fixed order)
# --------------------------------------------------------------------------- #
COLORS = {
    "Benign": "#2a78d6",       # blue
    "Pathogenic": "#eb6834",   # orange
    "VUS": "#1baf7a",          # aqua
}
MODEL_COLORS = {
    "XGBoost": "#2a78d6",
    "SVM-RBF": "#eb6834",
    "Transformer": "#1baf7a",
}
NEUTRAL = "#8a8985"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL_CMAP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING_CMAP = ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b52f2f"]


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
              MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
