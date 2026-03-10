# Machine Learning-Based Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD)

## Using Clinical and Lifestyle Data from NHANES 2017–2018

**Author:** Diksha Damahe  
**Date:** March 2026  
**Repository:** [github.com/dikshadamahe/NAFLD-Model](https://github.com/dikshadamahe/NAFLD-Model)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Sources](#2-data-sources)
3. [Methodology](#3-methodology)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Models Trained](#5-models-trained)
6. [Results](#6-results)
   - [Full Model Comparison](#61-full-model-comparison-ranked-by-test-roc-auc)
   - [Top 5 Models](#62-top-5-models)
   - [Best Model Classification Report](#63-best-model-classification-report)
   - [Interpretable Model Comparison](#64-interpretable-model-comparison)
   - [Statistical Comparison — McNemar's Test](#65-statistical-comparison--mcnemars-test)
7. [Visualizations](#7-visualizations)
8. [Project Structure](#8-project-structure)
9. [Reproducibility Instructions](#9-reproducibility-instructions)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Dependencies](#11-dependencies)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Project Overview

This project implements a comprehensive machine learning pipeline for predicting **Non-Alcoholic Fatty Liver Disease (NAFLD)** using **24 classification algorithms**. The pipeline follows a rigorous, reproducible methodology suitable for IEEE/Springer publication.

NAFLD is the most common chronic liver disease worldwide, affecting approximately 25% of the global population. Early detection through clinical and lifestyle markers can significantly improve patient outcomes. This project leverages data from the **National Health and Nutrition Examination Survey (NHANES) 2017–2018** to build and compare predictive models.

---

## 2. Data Sources

The project uses **6 NHANES 2017–2018 datasets**, merged on the common participant identifier `SEQN`:

| Dataset | Description | Features Extracted |
|---------|-------------|--------------------|
| `DEMO_J.xpt` | Demographics | Age, Gender, Ethnicity |
| `BMX_J.xpt` | Body Measurements | BMI, Waist Circumference |
| `TRIGLY_J.xpt` | Triglycerides | Triglycerides, LDL Cholesterol |
| `HDL_J.xpt` | HDL Cholesterol | HDL Cholesterol |
| `GLU_J.xpt` | Fasting Glucose | Glucose |
| `BIOPRO_J.xpt` | Standard Biochemistry Profile | ALT, AST, Total Cholesterol |

### 12 Clinical Features Used

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | Age | Numerical | DEMO_J |
| 2 | Gender | Categorical | DEMO_J |
| 3 | Ethnicity | Categorical | DEMO_J |
| 4 | BMI | Numerical | BMX_J |
| 5 | Waist Circumference | Numerical | BMX_J |
| 6 | Total Cholesterol | Numerical | BIOPRO_J |
| 7 | LDL Cholesterol | Numerical | TRIGLY_J |
| 8 | HDL Cholesterol | Numerical | HDL_J |
| 9 | Triglycerides | Numerical | TRIGLY_J |
| 10 | ALT | Numerical | BIOPRO_J |
| 11 | AST | Numerical | BIOPRO_J |
| 12 | Glucose | Numerical | GLU_J |

**Target Variable:** `NAFLD` (binary: 0 = No NAFLD, 1 = NAFLD)

---

## 3. Methodology

### Data Preprocessing
- **Missing value handling:** Rows with >50% missing features are dropped; remaining values are imputed with median (numerical) or mode (categorical).
- **Feature encoding:** Categorical features are one-hot encoded via `OneHotEncoder` with first-category drop.
- **Feature scaling:** Numerical features are standardized using `StandardScaler`.
- **ColumnTransformer:** Separate pipelines for numerical (median impute → scale) and categorical (mode impute → one-hot encode) features.

### Train-Test Split
- **Stratified 70/30 split** preserving class distribution.
- `random_state=42` for reproducibility.

### Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied **only on the training set** after the split — no data leakage.
- `class_weight='balanced'` used where supported by the classifier.

### Cross-Validation
- **5-Fold Stratified Cross-Validation** on the training set only.
- Metrics tracked: Accuracy, ROC-AUC.

### Model Evaluation
- Test set evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Feature importance analysis for tree-based models.
- SHAP explainability for the best model.
- McNemar's test for statistical comparison between top models.

---

## 4. Pipeline Architecture

```
6 NHANES .xpt Datasets
    ↓
merge_nhanes_datasets.py / src/build_nafld_dataset.py
    ↓  (merge on SEQN, select 12 health features, handle missing values)
Merged Dataset (data/nafld_final_dataset.csv)
    ↓
nafld_pipeline.py
    ↓
Feature Encoding (OneHotEncoder) + Scaling (StandardScaler)
    ↓
Stratified Train-Test Split (70–30)
    ↓
SMOTE (on training set only)
    ↓
5-Fold Stratified Cross-Validation (training set only)
    ↓
Train 24 ML Models
    ↓
Performance Comparison (ROC-AUC ranking)
    ↓
Best Model Selection → saved as .pkl
    ↓
nafld_research_analysis.py
    ↓
9 Research Analyses:
  1. Model Ranking & Comparison Table
  2. ROC Curve Visualization
  3. Confusion Matrix + Sensitivity/Specificity
  4. Feature Importance (4 tree-based models + comparison)
  5. SHAP Explainability
  6. Interpretable Simple Model (5-feature Logistic Regression)
  7. External Validation Stub
  8. McNemar's Test
  9. Save All Results
```

---

## 5. Models Trained

24 classifiers spanning linear, tree-based, ensemble, kernel-based, and neural network families:

| # | Model | Family | # | Model | Family |
|---|-------|--------|---|-------|--------|
| 1 | Logistic Regression | Linear | 13 | KNN | Instance-based |
| 2 | Ridge Classifier | Linear | 14 | Gaussian Naive Bayes | Probabilistic |
| 3 | Lasso Logistic Regression | Linear | 15 | AdaBoost | Ensemble (Boosting) |
| 4 | Decision Tree | Tree | 16 | Bagging Classifier | Ensemble (Bagging) |
| 5 | Random Forest | Ensemble (Bagging) | 17 | SGD Classifier | Linear |
| 6 | Extra Trees | Ensemble (Bagging) | 18 | Perceptron | Linear |
| 7 | Gradient Boosting | Ensemble (Boosting) | 19 | Passive Aggressive | Linear |
| 8 | XGBoost | Ensemble (Boosting) | 20 | QDA | Discriminant |
| 9 | LightGBM | Ensemble (Boosting) | 21 | LDA | Discriminant |
| 10 | CatBoost | Ensemble (Boosting) | 22 | MLP Classifier | Neural Network |
| 11 | SVM (Linear) | Kernel | 23 | Hist Gradient Boosting | Ensemble (Boosting) |
| 12 | SVM (RBF) | Kernel | 24 | Voting Classifier (top 3) | Meta-Ensemble |

---

## 6. Results

### 6.1 Full Model Comparison (Ranked by Test ROC-AUC)

| Rank | Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Precision | Recall | F1-score | Test ROC-AUC |
|------|-------|-------------|------------|---------------|-----------|--------|----------|--------------|
| 1 | Random Forest | 0.9337 | 0.9887 | 0.8921 | 0.7553 | 0.8404 | 0.7956 | **0.9644** |
| 2 | Gradient Boosting | 0.9239 | 0.9867 | 0.8769 | 0.7160 | 0.8404 | 0.7732 | 0.9638 |
| 3 | CatBoost | 0.9373 | 0.9885 | 0.8839 | 0.7262 | 0.8592 | 0.7871 | 0.9632 |
| 4 | XGBoost | 0.9440 | 0.9895 | 0.8980 | 0.7890 | 0.8075 | 0.7981 | 0.9631 |
| 5 | Hist Gradient Boosting | 0.9414 | 0.9907 | 0.8898 | 0.7742 | 0.7887 | 0.7814 | 0.9630 |
| 6 | Bagging Classifier | 0.9323 | 0.9871 | 0.8863 | 0.7500 | 0.8169 | 0.7820 | 0.9629 |
| 7 | LightGBM | 0.9420 | 0.9906 | 0.8945 | 0.7808 | 0.8028 | 0.7917 | 0.9612 |
| 8 | Voting Classifier | 0.9437 | 0.9921 | 0.8957 | 0.7870 | 0.7981 | 0.7925 | 0.9583 |
| 9 | AdaBoost | 0.9199 | 0.9816 | 0.8769 | 0.7077 | 0.8638 | 0.7780 | 0.9574 |
| 10 | Extra Trees | 0.9450 | 0.9906 | 0.8722 | 0.7301 | 0.7746 | 0.7517 | 0.9462 |
| 11 | SVM (RBF) | 0.8961 | 0.9585 | 0.8441 | 0.6399 | 0.8592 | 0.7335 | 0.9321 |
| 12 | MLP Classifier | 0.8710 | 0.9498 | 0.8499 | 0.6545 | 0.8451 | 0.7377 | 0.9213 |
| 13 | Logistic Regression | 0.8361 | 0.9199 | 0.8242 | 0.6047 | 0.8545 | 0.7082 | 0.9107 |
| 14 | Lasso Logistic Regression | 0.8351 | 0.9198 | 0.8206 | 0.6000 | 0.8451 | 0.7018 | 0.9102 |
| 15 | SVM (Linear) | 0.8351 | 0.9195 | 0.8183 | 0.5967 | 0.8404 | 0.6979 | 0.9099 |
| 16 | LDA | 0.8351 | 0.9184 | 0.8113 | 0.5788 | 0.8967 | 0.7035 | 0.9070 |
| 17 | Ridge Classifier | 0.8338 | 0.9184 | 0.8206 | 0.6000 | 0.8451 | 0.7018 | 0.9070 |
| 18 | QDA | 0.8388 | 0.9150 | 0.8113 | 0.5813 | 0.8732 | 0.6979 | 0.9023 |
| 19 | Passive Aggressive | 0.8174 | 0.9059 | 0.8019 | 0.5710 | 0.8310 | 0.6769 | 0.9022 |
| 20 | Perceptron | 0.8164 | 0.8973 | 0.8089 | 0.5828 | 0.8263 | 0.6835 | 0.8973 |
| 21 | SGD Classifier | 0.8170 | 0.9021 | 0.8019 | 0.5688 | 0.8545 | 0.6829 | 0.8973 |
| 22 | Gaussian Naive Bayes | 0.8318 | 0.9141 | 0.8101 | 0.5889 | 0.7934 | 0.6760 | 0.8898 |
| 23 | KNN | 0.8703 | 0.9498 | 0.7902 | 0.5541 | 0.8169 | 0.6603 | 0.8675 |
| 24 | Decision Tree | 0.9145 | 0.9146 | 0.8757 | 0.7238 | 0.8122 | 0.7655 | 0.8545 |

### 6.2 Top 5 Models

| Rank | Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Precision | Recall | F1-score | Test ROC-AUC |
|------|-------|-------------|------------|---------------|-----------|--------|----------|--------------|
| 1 | AdaBoost | 0.9050 | 0.9701 | 0.8819 | 0.7473 | 0.7968 | 0.7713 | **0.9445** |
| 2 | Gradient Boosting | 0.9113 | 0.9735 | 0.8783 | 0.7342 | 0.8040 | 0.7675 | 0.9427 |
| 3 | CatBoost | 0.9118 | 0.9756 | 0.8783 | 0.7452 | 0.7795 | 0.7620 | 0.9422 |
| 4 | Voting Classifier | 0.9131 | 0.9772 | 0.8776 | 0.7479 | 0.7695 | 0.7585 | 0.9415 |
| 5 | Hist Gradient Boosting | 0.9102 | 0.9764 | 0.8711 | 0.7386 | 0.7493 | 0.7439 | 0.9394 |

### 6.3 Best Model Classification Report

The **Random Forest** classifier achieved the highest Test ROC-AUC score:

```
Classification Report — Random Forest
==================================================
              precision    recall  f1-score   support

    No NAFLD       0.94      0.91      0.93       640
       NAFLD       0.76      0.84      0.80       213

    accuracy                           0.89       853
   macro avg       0.85      0.87      0.86       853
weighted avg       0.90      0.89      0.89       853
```

**Key Observations:**
- **Overall accuracy:** 89%
- **NAFLD class recall:** 84% — the model correctly identifies 84% of actual NAFLD cases
- **NAFLD class precision:** 76% — of predicted NAFLD cases, 76% are true positives
- **No NAFLD class:** 94% precision, 91% recall — excellent performance on the majority class
- **Macro average F1-score:** 0.86 — balanced performance across both classes

### 6.4 Interpretable Model Comparison

Comparison of a simple 5-feature Logistic Regression against the full-feature AdaBoost model:

| Model | Accuracy | ROC-AUC | Features |
|-------|----------|---------|----------|
| Logistic Regression (5 features) | 0.8318 | 0.9127 | RIDAGEYR, RIAGENDR_2.0, DMDHRAGZ_2.0, INDFMPIR, DMDEDUC2_4.0 |
| AdaBoost (all features) | 0.8819 | 0.9445 | all |

The interpretable model achieves **83.2% accuracy** and **0.913 ROC-AUC** with only 5 features, demonstrating that a simple, clinically explainable model can still provide strong predictive performance. The full AdaBoost model gains an additional ~5% accuracy and ~0.03 ROC-AUC using all features.

### 6.5 Statistical Comparison — McNemar's Test

McNemar's test was used to compare the predictions of the **top 2 models** (AdaBoost vs. Gradient Boosting):

| Comparison | Chi² | p-value | Significant? | b (M1✓ M2✗) | c (M1✗ M2✓) |
|------------|------|---------|--------------|--------------|--------------|
| AdaBoost vs. Gradient Boosting | 1.397 | 0.237 | **No** | 34 | 24 |

**Interpretation:** The p-value of 0.237 (> 0.05) indicates **no statistically significant difference** between the predictions of AdaBoost and Gradient Boosting. Both models perform comparably on this dataset. The choice between them can be based on other factors such as interpretability, training speed, or deployment considerations.

---

## 7. Visualizations

The following publication-quality figures (300 dpi) are saved in the `figures/` directory:

| Figure | File | Description |
|--------|------|-------------|
| ROC Curves (Top 5) | `roc_curves_top5.png` | Overlaid ROC curves for the top 5 models |
| Confusion Matrix (Best) | `confusion_matrix_best.png` | Confusion matrix for the best model |
| Confusion Matrices (Top 5) | `confusion_matrices_top5.png` | Side-by-side confusion matrices for top 5 |
| Feature Importance — Random Forest | `feature_importance_random_forest.png` | Feature importance bar chart |
| Feature Importance — XGBoost | `feature_importance_xgboost.png` | Feature importance bar chart |
| Feature Importance — LightGBM | `feature_importance_lightgbm.png` | Feature importance bar chart |
| Feature Importance — CatBoost | `feature_importance_catboost.png` | Feature importance bar chart |
| Feature Importance — Gradient Boosting | `feature_importance_gradient_boosting.png` | Feature importance bar chart |
| Feature Importance — Comparison | `feature_importance_comparison.png` | Cross-model feature importance comparison |
| Model Comparison Chart | `model_comparison_chart.png` | Bar chart comparing all 24 models |
| SHAP Summary Plot | `shap_summary.png` / `shap_summary_plot.png` | SHAP beeswarm summary |
| SHAP Bar Plot | `shap_bar.png` / `shap_bar_plot.png` | SHAP mean absolute value bar chart |

---

## 8. Project Structure

```
NAFLD-Model/
├── merge_nhanes_datasets.py       # Dataset merger (6 NHANES → 1 CSV)
├── nafld_pipeline.py              # Main ML pipeline (24 models)
├── nafld_research_analysis.py     # Extended 9-analysis research suite
├── requirements.txt               # Python dependencies
├── README.md                      # Project README
├── NAFLD_Project_Report.md        # This comprehensive report
├── push_changes.sh                # Git push script
├── git_workflow.md                # Git branching & tagging reference
│
├── data/
│   ├── DEMO_J.xpt                # NHANES Demographics
│   ├── BMX_J.xpt                 # NHANES Body Measurements
│   ├── TRIGLY_J.xpt              # NHANES Triglycerides
│   ├── HDL_J.xpt                 # NHANES HDL Cholesterol
│   ├── GLU_J.xpt                 # NHANES Fasting Glucose
│   ├── BIOPRO_J.xpt              # NHANES Biochemistry Profile
│   ├── merged_nhanes_dataset.csv  # Merged dataset
│   └── nafld_final_dataset.csv    # Final training-ready dataset
│
├── src/
│   └── build_nafld_dataset.py     # Alternative dataset builder with NAFLD label
│
├── models/
│   └── best_nafld_model.pkl       # Serialized best model
│
├── figures/                       # Publication-quality plots (300 dpi)
│   ├── roc_curves_top5.png
│   ├── confusion_matrix_best.png
│   ├── confusion_matrices_top5.png
│   ├── feature_importance_*.png
│   ├── feature_importance_comparison.png
│   ├── model_comparison_chart.png
│   ├── shap_summary.png
│   └── shap_bar.png
│
├── results/                       # Quantitative results (CSV/TXT)
│   ├── model_comparison.csv       # All 24 models ranked
│   ├── top5_models.csv            # Top 5 models detail
│   ├── classification_report.txt  # Best model classification report
│   ├── interpretable_model_comparison.csv
│   └── mcnemar_test.csv           # Statistical test results
│
└── catboost_info/                 # CatBoost training logs
```

---

## 9. Reproducibility Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/dikshadamahe/NAFLD-Model.git
cd NAFLD-Model
```

### Step 2: Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Place NHANES Datasets

Place all 6 NHANES `.xpt` files inside the `data/` directory.

### Step 4: Build the Dataset

```bash
python3 src/build_nafld_dataset.py
```

Or use the alternative merger:

```bash
python3 merge_nhanes_datasets.py
```

### Step 5: Run the ML Pipeline

```bash
python3 nafld_pipeline.py
```

### Step 6: Run Research Analysis (optional)

```bash
python3 nafld_research_analysis.py
```

### Outputs

| Output | Location |
|--------|----------|
| Merged dataset | `data/merged_nhanes_dataset.csv` |
| Final dataset | `data/nafld_final_dataset.csv` |
| Ranked model comparison | `results/model_comparison.csv` |
| Top 5 models | `results/top5_models.csv` |
| Classification report | `results/classification_report.txt` |
| Interpretable comparison | `results/interpretable_model_comparison.csv` |
| McNemar's test | `results/mcnemar_test.csv` |
| ROC curves | `figures/roc_curves_top5.png` |
| Feature importance plots | `figures/feature_importance_*.png` |
| SHAP plots | `figures/shap_*.png` |
| Best model (serialized) | `models/best_nafld_model.pkl` |

---

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-dataset integration** | 6 NHANES datasets merged on `SEQN` with inner join (complete records only) |
| **12 clinically relevant features** | Age, Gender, Ethnicity, BMI, Waist Circumference, Total Cholesterol, LDL, HDL, Triglycerides, ALT, AST, Glucose |
| **No data leakage** | SMOTE applied only on training set after the split |
| **Class imbalance** | Dual strategy — SMOTE + `class_weight='balanced'` where supported |
| **Reproducibility** | `random_state=42` used consistently across all models and splits |
| **Automatic column detection** | No hardcoded feature names except the target column |
| **24 diverse classifiers** | Cover linear, tree, ensemble, kernel, probabilistic, and neural network families |
| **CalibratedClassifierCV** | Wraps classifiers lacking `predict_proba` for fair ROC-AUC comparison |
| **Voting Classifier** | Soft voting ensemble of top 3 models by CV ROC-AUC |
| **Publication-quality figures** | 300 dpi, serif fonts, sized for IEEE/Springer column widths |

---

## 11. Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | ≥ 1.24.0 | Numerical computing |
| pandas | ≥ 2.0.0 | Data manipulation |
| scikit-learn | ≥ 1.3.0 | ML models, preprocessing, metrics |
| xgboost | ≥ 2.0.0 | XGBoost classifier |
| lightgbm | ≥ 4.0.0 | LightGBM classifier |
| catboost | ≥ 1.2.0 | CatBoost classifier |
| imbalanced-learn | ≥ 0.11.0 | SMOTE oversampling |
| matplotlib | ≥ 3.7.0 | Plotting |
| seaborn | ≥ 0.12.0 | Statistical visualizations |
| joblib | ≥ 1.3.0 | Model serialization |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Citation

```bibtex
@misc{nafld-model-2026,
  author = {Diksha Damahe},
  title  = {Machine Learning-Based Prediction of NAFLD Using Clinical and Lifestyle Data},
  year   = {2026},
  url    = {https://github.com/dikshadamahe/NAFLD-Model}
}
```

---

## 13. License

This project is for academic research purposes.
