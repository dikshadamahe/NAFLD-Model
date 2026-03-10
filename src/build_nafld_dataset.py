"""
=============================================================================
Build NAFLD Final Dataset for ML Training
=============================================================================
Loads 6 NHANES 2017–2018 .xpt datasets, merges them on SEQN,
selects clinically relevant features, handles missing values,
derives the NAFLD proxy target label, and saves the final
training-ready dataset.

Input datasets (data/):
  - DEMO_J.xpt     : Demographics (Age, Gender, Ethnicity)
  - BMX_J.xpt      : Body Measurements (BMI, Waist Circumference)
  - TRIGLY_J.xpt    : Triglycerides & LDL Cholesterol
  - HDL_J.xpt       : HDL Cholesterol
  - GLU_J.xpt       : Fasting Glucose
  - BIOPRO_J.xpt    : Standard Biochemistry Profile (ALT, AST, Total Cholesterol)

Output:
  - data/nafld_final_dataset.csv
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────────────

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "nafld_final_dataset.csv")

# NHANES variable name -> friendly name mapping per dataset
DATASETS = {
    "DEMO_J.xpt": {
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender",
        "RIDRETH3": "Ethnicity",
    },
    "BMX_J.xpt": {
        "BMXBMI": "BMI",
        "BMXWAIST": "Waist_Circumference",
    },
    "TRIGLY_J.xpt": {
        "LBXTR": "Triglycerides",
        "LBDLDL": "LDL",
    },
    "HDL_J.xpt": {
        "LBDHDD": "HDL",
    },
    "GLU_J.xpt": {
        "LBXGLU": "Glucose",
    },
    "BIOPRO_J.xpt": {
        "LBXSATSI": "AST",
        "LBXSAL": "ALT",
        "LBXSCH": "Total_Cholesterol",
    },
}

# Features to retain in the final dataset
FEATURE_COLUMNS = [
    "Age", "Gender", "Ethnicity",
    "BMI", "Waist_Circumference",
    "Total_Cholesterol", "LDL", "HDL", "Triglycerides",
    "ALT", "AST",
    "Glucose",
]

TARGET = "NAFLD"


# ── Step 1: Load datasets ──────────────────────────────────────────────────

def load_dataset(filename, columns_map):
    """Load a single .xpt file and extract SEQN + relevant columns."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [ERROR] File not found: {filepath}")
        return None

    df = pd.read_sas(filepath, format="xport", encoding="utf-8")
    print(f"  Loaded {filename:<16s} → {df.shape[0]:,} rows, {df.shape[1]} cols")

    keep = ["SEQN"]
    rename = {}
    for nhanes_col, name in columns_map.items():
        if nhanes_col in df.columns:
            keep.append(nhanes_col)
            rename[nhanes_col] = name
        else:
            print(f"    [WARNING] Column '{nhanes_col}' not in {filename}")

    df = df[keep].copy()
    df.rename(columns=rename, inplace=True)
    return df


def load_all():
    """Load all 6 NHANES datasets."""
    print("=" * 65)
    print("  STEP 1: Loading NHANES .xpt datasets")
    print("=" * 65)

    frames = []
    for fname, cmap in DATASETS.items():
        df = load_dataset(fname, cmap)
        if df is not None:
            frames.append(df)

    print(f"\n  Loaded {len(frames)}/{len(DATASETS)} datasets.\n")
    return frames


# ── Step 2: Merge on SEQN ──────────────────────────────────────────────────

def merge(frames):
    """Inner-join all datasets on the shared participant ID SEQN."""
    print("=" * 65)
    print("  STEP 2: Merging datasets on SEQN (inner join)")
    print("=" * 65)

    if not frames:
        sys.exit("  No datasets loaded. Place .xpt files in data/ and retry.")

    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on="SEQN", how="inner")

    print(f"  Merged shape: {merged.shape}")
    print(f"  Participants: {len(merged):,}\n")
    return merged


# ── Step 3: Select features ────────────────────────────────────────────────

def select_features(df):
    """Keep only the clinically relevant columns."""
    print("=" * 65)
    print("  STEP 3: Selecting relevant health features")
    print("=" * 65)

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [INFO] Unavailable features (dataset missing): {missing}")

    df = df[["SEQN"] + available].copy()
    print(f"  Kept {len(available)} features: {available}\n")
    return df


# ── Step 4: Handle missing values ──────────────────────────────────────────

def handle_missing(df):
    """Drop rows with excessive missingness; median-impute the rest."""
    print("=" * 65)
    print("  STEP 4: Handling missing values")
    print("=" * 65)

    miss = df.isnull().sum()
    total = miss.sum()
    print(f"  Missing values before cleaning: {total}")

    if total > 0:
        for col in df.columns:
            n = miss[col]
            if n > 0:
                print(f"    {col}: {n} ({100*n/len(df):.1f}%)")

    # Drop rows with >50% feature missingness
    feat_cols = [c for c in df.columns if c != "SEQN"]
    thresh = int(len(feat_cols) * 0.5) + 1
    before = len(df)
    df = df.dropna(thresh=thresh, subset=feat_cols).copy()
    dropped = before - len(df)
    if dropped:
        print(f"\n  Dropped {dropped} rows (>50% features missing).")

    # Median imputation for remaining NaNs
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "SEQN"]
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"  Missing after cleaning: {df.isnull().sum().sum()}")
    print(f"  Rows retained: {len(df):,}\n")
    return df


# ── Step 5: Derive NAFLD target label ──────────────────────────────────────

def derive_target(df):
    """
    Create a proxy NAFLD binary label using clinical risk factors.

    Scoring logic (replace with actual clinical labels for publication):
      - Age >= 45          → +0.35
      - Male (Gender == 1) → +0.15
      - BMI >= 30          → +0.20
      - Glucose >= 126     → +0.15
      - ALT >= 40          → +0.15
      + random noise [0, 0.2)
    Top 25% of scores → NAFLD = 1, rest → 0.
    """
    print("=" * 65)
    print("  STEP 5: Deriving NAFLD target label")
    print("=" * 65)

    np.random.seed(SEED)
    score = np.zeros(len(df))

    if "Age" in df.columns:
        score += (df["Age"] >= 45).astype(float) * 0.35
    if "Gender" in df.columns:
        score += (df["Gender"] == 1).astype(float) * 0.15
    if "BMI" in df.columns:
        score += (df["BMI"] >= 30).astype(float) * 0.20
    if "Glucose" in df.columns:
        score += (df["Glucose"] >= 126).astype(float) * 0.15
    if "ALT" in df.columns:
        score += (df["ALT"] >= 40).astype(float) * 0.15

    score += np.random.uniform(0, 0.2, len(df))
    threshold = np.percentile(score, 75)
    df[TARGET] = (score >= threshold).astype(int)

    n_pos = df[TARGET].sum()
    n_neg = len(df) - n_pos
    print(f"  Target column: '{TARGET}'")
    print(f"  Class distribution: 0={n_neg:,}  1={n_pos:,}  ({100*n_pos/len(df):.1f}% positive)\n")
    return df


# ── Step 6: Save final dataset ─────────────────────────────────────────────

def save(df):
    """Drop SEQN and save the training-ready dataset."""
    print("=" * 65)
    print("  STEP 6: Saving final dataset")
    print("=" * 65)

    df = df.drop(columns=["SEQN"]).copy()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  Shape:    {df.shape}  ({len(df.columns)-1} features + 1 target)\n")
    return df


# ── Step 7: Summary ────────────────────────────────────────────────────────

def summary(df):
    """Print final dataset summary."""
    print("=" * 65)
    print("  STEP 7: Dataset Summary")
    print("=" * 65)

    print(f"\n  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        uniq = df[col].nunique()
        miss = df[col].isnull().sum()
        print(f"    {col:<25s}  dtype={str(dtype):<10s}  unique={uniq:<6d}  missing={miss}")

    total_cells = df.shape[0] * df.shape[1]
    total_miss = df.isnull().sum().sum()
    print(f"\n  Total cells:   {total_cells:,}")
    print(f"  Total missing: {total_miss} ({100*total_miss/total_cells:.2f}%)")

    print(f"\n  Descriptive statistics:")
    print(df.describe().round(2).to_string())

    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  BUILD NAFLD FINAL DATASET FOR ML TRAINING")
    print("=" * 65 + "\n")

    frames = load_all()
    merged = merge(frames)
    merged = select_features(merged)
    merged = handle_missing(merged)
    merged = derive_target(merged)
    final = save(merged)
    summary(final)

    print("=" * 65)
    print("  Done! data/nafld_final_dataset.csv is ready for training.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
