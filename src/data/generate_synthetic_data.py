"""
Synthetic data generator for the VUS reclassification project.

Creates 3,410 variants x 30 columns (demography, variant annotation, in-silico
pathogenicity scores and tumour/clinical biomarkers) with a 3-level target:
Benign 70%, Pathogenic 20%, VUS 10%. Exactly 13% of all cells are blank
(missing completely at random, never in the ID or target columns).

How the signal is built
-----------------------
* Every variant has a hidden "latent pathogenicity" (1 = Pathogenic, 0 = Benign).
  About 35% of VUS are hidden-Pathogenic, and all VUS get intermediate values.
* Informative features shift their mean with the latent value; BMI, sex,
  ethnicity, smoking, alcohol, cancer type, gene, exon, read depth and PD-L1
  are pure noise on purpose.
* NOISE scales the measurement noise; FLIP is the share of labelled rows whose
  biology follows the opposite class (mimicking historic misclassification).
  NOISE = 3.5 and FLIP = 0.065 put Train and Test accuracy near 90%.

Usage:  python src/data/generate_synthetic_data.py [NOISE] [FLIP] [SEED] [OUTPUT.csv]
        make synthetic
Note: pandas reads the Alcohol_Use level "None" as NaN by default; the project
loader uses keep_default_na=False, na_values=[""] to keep it.
"""
import sys

import numpy as np
import pandas as pd

NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5   # multiplier on feature noise SD
FLIP  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.065 # share of labelled P/B rows whose features follow the opposite class
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 42
OUT   = sys.argv[4] if len(sys.argv) > 4 else "VUS_biomarker_cancer_mutation.csv"
rng = np.random.default_rng(SEED)
N = 3410

# ---- Target: 70% Benign, 20% Pathogenic, 10% VUS (exact counts) ----
n_ben, n_path = round(N * 0.70), round(N * 0.20)
n_vus = N - n_ben - n_path
target = np.array(["Benign"] * n_ben + ["Pathogenic"] * n_path + ["VUS"] * n_vus)
rng.shuffle(target)

# Latent pathogenicity: 1 = pathogenic, 0 = benign.
# VUS get an intermediate latent value (hidden truth is mixed and blurred).
latent = np.where(target == "Pathogenic", 1.0, 0.0)
vus_mask = target == "VUS"
hidden = rng.random(vus_mask.sum()) < 0.35          # ~35% of VUS truly pathogenic
latent[vus_mask] = np.where(hidden, 0.75, 0.25) + rng.normal(0, 0.08, vus_mask.sum())
latent = np.clip(latent, 0, 1)
# label noise: some labelled Pathogenic/Benign rows carry the other class's biology
# (mimics historic misclassification in ClinVar-style labels)
pb = np.where(~vus_mask)[0]
flip_idx = rng.choice(pb, round(FLIP * len(pb)), replace=False)
latent[flip_idx] = 1 - latent[flip_idx]

def sig(mu0, mu1, sd, size=N):
    """Continuous feature whose mean shifts with latent pathogenicity."""
    return mu0 + (mu1 - mu0) * latent + rng.normal(0, sd * NOISE, size)

df = pd.DataFrame()
df["Variant_ID"] = [f"VAR{str(i).zfill(5)}" for i in range(1, N + 1)]

# ---------------- Demographic features ----------------
df["Age"] = np.clip(rng.normal(56, 13, N) - 4 * latent, 18, 90).round().astype(int)
df["Sex"] = rng.choice(["Female", "Male"], N, p=[0.58, 0.42])
df["Ethnicity"] = rng.choice(
    ["White", "Black", "Hispanic", "Asian", "Other"], N, p=[0.55, 0.15, 0.15, 0.10, 0.05])
df["BMI"] = np.clip(rng.normal(27.5, 5, N), 16, 50).round(1)
df["Smoking_Status"] = rng.choice(["Never", "Former", "Current"], N, p=[0.55, 0.30, 0.15])
df["Alcohol_Use"] = rng.choice(["None", "Moderate", "Heavy"], N, p=[0.40, 0.48, 0.12])
p_fh = 0.20 + 0.55 * latent
df["Family_History_Cancer"] = np.where(rng.random(N) < p_fh, "Yes", "No")
df["Cancer_Type"] = rng.choice(
    ["Breast", "Ovarian", "Colorectal", "Lung", "Prostate", "Pancreatic"], N,
    p=[0.30, 0.12, 0.18, 0.18, 0.14, 0.08])

# ---------------- Variant / gene annotation ----------------
genes = ["BRCA1", "BRCA2", "TP53", "PALB2", "ATM", "CHEK2", "MLH1", "MSH2", "APC", "PTEN"]
df["Gene"] = rng.choice(genes, N)

vt_path = [0.35, 0.22, 0.25, 0.15, 0.03]   # Missense, Nonsense, Frameshift, Splice, Synonymous
vt_ben  = [0.62, 0.02, 0.03, 0.05, 0.28]
vtypes = ["Missense", "Nonsense", "Frameshift", "Splice_site", "Synonymous"]
df["Variant_Type"] = [
    rng.choice(vtypes, p=np.array(vt_path) * l + np.array(vt_ben) * (1 - l)) for l in latent]

df["Exon_Number"] = rng.integers(1, 28, N)
df["In_Functional_Domain"] = np.where(rng.random(N) < 0.25 + 0.60 * latent, "Yes", "No")

# ---------------- In-silico pathogenicity predictors (strong signal) ----------------
df["CADD_Phred"]     = np.clip(sig(8, 28, 4.5), 0, 50).round(2)
df["REVEL_Score"]    = np.clip(sig(0.18, 0.78, 0.13), 0, 1).round(3)
df["SIFT_Score"]     = np.clip(sig(0.55, 0.04, 0.14), 0, 1).round(3)     # low = damaging
df["PolyPhen2_Score"]= np.clip(sig(0.20, 0.90, 0.15), 0, 1).round(3)
df["SpliceAI_Score"] = np.clip(sig(0.04, 0.45, 0.12), 0, 1).round(3)
df["GERP_RS"]        = np.clip(sig(0.5, 4.8, 1.3), -12, 6.2).round(2)
df["PhyloP_100way"]  = np.clip(sig(0.8, 6.5, 1.8), -20, 10).round(2)
# Population frequency: pathogenic variants are rare
log_af = sig(-2.6, -5.6, 0.7)
df["gnomAD_AF"] = np.clip(10 ** log_af, 1e-7, 0.5).round(8)

# ---------------- Tumour / clinical biomarkers (moderate/weak signal) ----------------
df["Variant_Allele_Freq"]   = np.clip(sig(0.35, 0.52, 0.12), 0.02, 1).round(3)
df["Read_Depth"]            = np.clip(rng.normal(320, 110, N), 20, 1000).round().astype(int)
df["Tumor_Mutational_Burden"] = np.clip(sig(6, 13, 4), 0, 60).round(2)
df["Ki67_Index"]            = np.clip(sig(18, 38, 12), 0, 100).round(1)
df["PD_L1_TPS"]             = np.clip(rng.normal(22, 18, N), 0, 100).round(1)
df["CEA_ng_mL"]             = np.clip(np.exp(sig(0.9, 1.6, 0.6)), 0.1, 200).round(2)
df["CA125_U_mL"]            = np.clip(np.exp(sig(2.9, 3.7, 0.7)), 1, 2000).round(1)
df["HRD_Score"]             = np.clip(sig(18, 52, 11), 0, 100).round(1)

df["Classification"] = target

# ---------------- 13% missing completely at random ----------------
feature_cols = [c for c in df.columns if c not in ("Variant_ID", "Classification")]
feat = df[feature_cols].astype(object)
# exactly 13% of ALL cells in the file are blanked, only in feature columns (ID and target stay complete)
n_missing = round(0.13 * N * df.shape[1])
flat = rng.choice(feat.size, n_missing, replace=False)
mask = np.zeros(feat.size, bool); mask[flat] = True
feat = feat.mask(mask.reshape(feat.shape))
df[feature_cols] = feat
for c in feature_cols:
    try:
        df[c] = pd.to_numeric(df[c])
    except (ValueError, TypeError):   # categorical column: keep as text
        pass

df.to_csv(OUT, index=False)

# ---------------- Report ----------------
print("shape:", df.shape)
print(df["Classification"].value_counts(normalize=True).round(3).to_dict())
print("missing % (features):", round(df[feature_cols].isna().mean().mean() * 100, 2))
print("missing % (whole file):", round(df.isna().mean().mean() * 100, 2))
