"""
=============================================================================
NHANES Dataset Merger for NAFLD Prediction
=============================================================================
Loads multiple NHANES .xpt datasets, merges them on SEQN,
selects relevant health features, handles missing values,
and saves the final merged dataset.

Datasets:
  - DEMO_J.xpt     : Demographics (Age, Gender, Ethnicity)
  - BMX_J.xpt      : Body Measurements (BMI, Waist Circumference)
  - TRIGLY_J.xpt    : Triglycerides & LDL Cholesterol
  - HDL_J.xpt       : HDL Cholesterol
  - GLU_J.xpt       : Fasting Glucose
  - BIOPRO_J.xpt    : Standard Biochemistry Profile (ALT, AST, Total Cholesterol)
=============================================================================
"""

import os
import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "merged_nhanes_dataset.csv")

# Each entry: (filename, list of columns to keep besides SEQN)
# NHANES variable name -> descriptive rename
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
        "LBDLDL": "LDL",            # LDL Cholesterol (calculated)
    },
    "HDL_J.xpt": {
        "LBDHDD": "HDL",
    },
    "GLU_J.xpt": {
        "LBXGLU": "Glucose",
    },
    "BIOPRO_J.xpt": {               # Standard Biochemistry Profile (includes liver enzymes)
        "LBXSATSI": "AST",
        "LBXSAL": "ALT",
        "LBXSCH": "Total_Cholesterol",
    },
}


# ── Step 1: Load individual datasets ───────────────────────────────────────

def load_dataset(filename, columns_map):
    """Load a single .xpt dataset and keep only SEQN + relevant columns."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found: {filepath} — skipping.")
        return None

    df = pd.read_sas(filepath, format="xport", encoding="utf-8")
    print(f"  Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

    # Keep SEQN + the requested columns that actually exist in the file
    keep_cols = ["SEQN"]
    rename_map = {}
    for nhanes_col, friendly_name in columns_map.items():
        if nhanes_col in df.columns:
            keep_cols.append(nhanes_col)
            rename_map[nhanes_col] = friendly_name
        else:
            print(f"    [WARNING] Column '{nhanes_col}' not found in {filename}")

    df = df[keep_cols].copy()
    df.rename(columns=rename_map, inplace=True)
    return df


def load_all_datasets():
    """Load all configured NHANES datasets."""
    print("=" * 60)
    print("Step 1: Loading NHANES datasets")
    print("=" * 60)

    loaded = []
    for filename, columns_map in DATASETS.items():
        df = load_dataset(filename, columns_map)
        if df is not None:
            loaded.append(df)

    print(f"\n  Successfully loaded {len(loaded)} / {len(DATASETS)} datasets.\n")
    return loaded


# ── Step 2: Merge datasets on SEQN ─────────────────────────────────────────

def merge_datasets(dataframes):
    """Merge all datasets on SEQN using inner join to keep complete records."""
    print("=" * 60)
    print("Step 2: Merging datasets on SEQN")
    print("=" * 60)

    if not dataframes:
        raise ValueError("No datasets loaded — nothing to merge.")

    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = pd.merge(merged, df, on="SEQN", how="inner")

    print(f"  Merged dataset shape: {merged.shape}")
    print(f"  Participants with records in all datasets: {len(merged)}\n")
    return merged


# ── Step 3: Select relevant features ───────────────────────────────────────

def select_features(df):
    """Keep only the health features needed for NAFLD prediction."""
    print("=" * 60)
    print("Step 3: Selecting relevant health features")
    print("=" * 60)

    # All possible feature columns (friendly names from DATASETS config)
    desired_features = [
        "Age", "Gender", "Ethnicity",
        "BMI", "Waist_Circumference",
        "Total_Cholesterol", "LDL", "HDL", "Triglycerides",
        "ALT", "AST",
        "Glucose",
    ]

    available = [col for col in desired_features if col in df.columns]
    missing = [col for col in desired_features if col not in df.columns]

    if missing:
        print(f"  [INFO] Features not available (dataset missing): {missing}")

    # Keep SEQN as identifier + available features
    df = df[["SEQN"] + available].copy()
    print(f"  Selected {len(available)} features: {available}\n")
    return df


# ── Step 4: Handle missing values ──────────────────────────────────────────

def handle_missing_values(df):
    """Handle missing values with appropriate strategies per column type."""
    print("=" * 60)
    print("Step 4: Handling missing values")
    print("=" * 60)

    # Report missing values before cleaning
    missing_before = df.isnull().sum()
    total_missing = missing_before.sum()
    print(f"  Total missing values before cleaning: {total_missing}")

    if total_missing > 0:
        print("\n  Missing values per column (before):")
        for col in df.columns:
            n_miss = missing_before[col]
            if n_miss > 0:
                pct = 100.0 * n_miss / len(df)
                print(f"    {col}: {n_miss} ({pct:.1f}%)")

    # Drop rows where critical identifiers or too many features are missing
    # First, drop rows missing more than 50% of feature columns
    feature_cols = [c for c in df.columns if c != "SEQN"]
    threshold = len(feature_cols) * 0.5
    before_count = len(df)
    df = df.dropna(thresh=int(threshold) + 1, subset=feature_cols).copy()
    dropped = before_count - len(df)
    if dropped > 0:
        print(f"\n  Dropped {dropped} rows with >50% missing features.")

    # Impute remaining missing values
    # Numeric columns: fill with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "SEQN"]

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                print(f"  [WARNING] Column '{col}' is entirely NaN. Dropping column.")
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].fillna(median_val)

    missing_after = df.isnull().sum().sum()
    print(f"\n  Total missing values after cleaning: {missing_after}")
    print(f"  Final dataset size: {len(df)} rows\n")
    return df


# ── Step 5: Save merged dataset ────────────────────────────────────────────

def save_dataset(df):
    """Save the final merged and cleaned dataset to CSV."""
    print("=" * 60)
    print("Step 5: Saving merged dataset")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to: {OUTPUT_PATH}\n")


# ── Step 6: Print summary ──────────────────────────────────────────────────

def print_summary(df):
    """Print dataset shape, column names, and missing value summary."""
    print("=" * 60)
    print("Step 6: Dataset Summary")
    print("=" * 60)

    print(f"\n  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_miss = df[col].isnull().sum()
        print(f"    - {col:<25s} dtype={str(dtype):<10s}  unique={n_unique:<6d}  missing={n_miss}")

    print(f"\n  Missing value summary:")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    print(f"    Total cells:   {total_cells}")
    print(f"    Total missing: {total_missing} ({100.0 * total_missing / total_cells:.2f}%)")

    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print()

    print(f"\n  Descriptive statistics:")
    print(df.describe().round(2).to_string())
    print()


# ── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  NHANES Data Merger for NAFLD Prediction")
    print("=" * 60 + "\n")

    # Step 1: Load all datasets
    dataframes = load_all_datasets()

    # Step 2: Merge on SEQN
    merged = merge_datasets(dataframes)

    # Step 3: Select relevant features
    merged = select_features(merged)

    # Step 4: Handle missing values
    merged = handle_missing_values(merged)

    # Step 5: Save to CSV
    save_dataset(merged)

    # Step 6: Print summary
    print_summary(merged)

    print("=" * 60)
    print("  Done! Merged dataset ready for NAFLD model training.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
