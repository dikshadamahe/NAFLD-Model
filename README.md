# Machine Learning-Based Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD)

## Using Clinical and Lifestyle Data

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements a comprehensive machine learning pipeline for predicting NAFLD using 24 classification algorithms. The pipeline follows a rigorous, reproducible methodology suitable for IEEE/Springer publication.

### Pipeline Architecture

```
Raw Dataset (DEMO_J.xpt)
    ↓
Data Cleaning (median/mode imputation)
    ↓
Feature Encoding (OneHotEncoder)
    ↓
Scaling (StandardScaler)
    ↓
Stratified Train-Test Split (70–30)
    ↓
SMOTE (on training set only)
    ↓
5-Fold Cross Validation (training set only)
    ↓
Train 24 ML Models
    ↓
Performance Comparison (ROC-AUC ranking)
    ↓
Best Model Selection → saved as .pkl
    ↓
Feature Importance + ROC Curves
```

---

## 24 Models Trained

| # | Model | # | Model |
|---|-------|---|-------|
| 1 | Logistic Regression | 13 | KNN |
| 2 | Ridge Classifier | 14 | Gaussian Naive Bayes |
| 3 | Lasso Logistic Regression | 15 | AdaBoost |
| 4 | Decision Tree | 16 | Bagging Classifier |
| 5 | Random Forest | 17 | SGD Classifier |
| 6 | Extra Trees | 18 | Perceptron |
| 7 | Gradient Boosting | 19 | Passive Aggressive |
| 8 | XGBoost | 20 | Quadratic Discriminant Analysis |
| 9 | LightGBM | 21 | Linear Discriminant Analysis |
| 10 | CatBoost | 22 | MLP Classifier |
| 11 | SVM (Linear) | 23 | Histogram Gradient Boosting |
| 12 | SVM (RBF) | 24 | Voting Classifier (top 3) |

---

## Project Structure

```
NAFLD-Model/
├── nafld_pipeline.py          # Main ML pipeline (all 24 models)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── push_changes.sh            # Safe git push script
├── git_workflow.md            # Git branching & tagging reference
├── .gitignore                 # Research-safe gitignore
├── data/
│   └── DEMO_J.xpt            # Raw NHANES dataset (excluded from git)
├── models/
│   └── best_nafld_model.pkl   # Serialized best model (excluded from git)
├── figures/
│   ├── roc_curves_top5.png    # ROC curves for top 5 models
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_xgboost.png
│   ├── feature_importance_lightgbm.png
│   └── feature_importance_catboost.png
└── results/
    └── model_comparison.csv   # Ranked model comparison table
```

---

## Reproducibility Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dikshadamahe/NAFLD-Model.git
cd NAFLD-Model
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Dataset

Place `DEMO_J.xpt` inside the `data/` directory:
```bash
cp /path/to/DEMO_J.xpt data/
```

### 4. Run the Pipeline

```bash
python3 nafld_pipeline.py
```

### 5. Outputs

| Output | Location |
|--------|----------|
| Ranked comparison table | `results/model_comparison.csv` |
| ROC curves (top 5) | `figures/roc_curves_top5.png` |
| Feature importance plots | `figures/feature_importance_*.png` |
| Best model (serialized) | `models/best_nafld_model.pkl` |

---

## Key Design Decisions

- **No data leakage**: SMOTE applied only on training set after split.
- **ColumnTransformer**: Separate pipelines for numerical (median impute → scale) and categorical (mode impute → one-hot encode) features.
- **Class imbalance**: Dual strategy — SMOTE + `class_weight='balanced'` where supported.
- **Reproducibility**: `random_state=42` used consistently across all models and splits.
- **Automatic column detection**: No hardcoded feature names except the target column.

---

## Git Workflow

### Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable research version |
| `dev` | Active development |
| `experiment` | Model experimentation |

### Tags (Research Milestones)

| Tag | Milestone |
|-----|-----------|
| `v1.0-preprocessing` | Preprocessing pipeline complete |
| `v2.0-model-comparison` | 24 models trained and compared |
| `v3.0-final-model` | Best model selected and saved |

---

## Requirements

- Python ≥ 3.10
- See [requirements.txt](requirements.txt) for full dependency list.

---

## Citation

If you use this code in your research, please cite:

```
@misc{nafld-model-2026,
  author = {Diksha Damahe},
  title  = {Machine Learning-Based Prediction of NAFLD Using Clinical and Lifestyle Data},
  year   = {2026},
  url    = {https://github.com/dikshadamahe/NAFLD-Model}
}
```

---

## License

This project is for academic research purposes.
