# NAFLD Prediction — Final Results Summary

**Generated**: March 2026
**Total models trained**: 24
**Best model**: Random Forest

---

## 1. Best Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.8921 |
| Precision | 0.7553 |
| Recall (Sensitivity) | 0.8404 |
| F1-score | 0.7956 |
| ROC-AUC | 0.9644 |
| CV Accuracy | 0.9337 |
| CV ROC-AUC | 0.9887 |

---

## 2. Top 5 Models

| Rank | Model | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|-------|---------------|-----------|--------|----------|---------|
| 1 | Random Forest | 0.8921 | 0.7553 | 0.8404 | 0.7956 | 0.9644 |
| 2 | Gradient Boosting | 0.8769 | 0.7160 | 0.8404 | 0.7732 | 0.9638 |
| 3 | CatBoost | 0.8839 | 0.7262 | 0.8592 | 0.7871 | 0.9632 |
| 4 | XGBoost | 0.8980 | 0.7890 | 0.8075 | 0.7981 | 0.9631 |
| 5 | Hist Gradient Boosting | 0.8898 | 0.7742 | 0.7887 | 0.7814 | 0.9630 |

---

## 3. Model Performance Explanation

The **Random Forest** achieved the highest test ROC-AUC of **0.9644**
across 24 classifiers evaluated. The top 5 models are all ensemble-based
methods, indicating that combining multiple decision trees yields the most robust
predictions for NAFLD risk.

Key observations:
- All top 5 models achieved ROC-AUC > 0.96, indicating excellent discriminative ability.
- The best model's recall of 0.8404 means it correctly identifies
  ~84% of NAFLD cases, which is critical for clinical screening.
- Cross-validation AUC (0.9887) closely matches test AUC (0.9644),
  suggesting minimal overfitting.

---

## 4. Interpretable Model Comparison

| Model | Accuracy | ROC-AUC | Features |
|-------|----------|---------|----------|
| Logistic Regression (5 features) | 0.8218 | 0.9096 | Age, Glucose, Waist_Circumference, BMI, Gender_2.0 |
| Random Forest | 0.8921 | 0.9644 | all |

A simple Logistic Regression using only 5 features achieves a competitive
ROC-AUC, validating that core clinical features (Age, Glucose,
Waist Circumference, BMI, Gender) carry the majority of predictive signal.

---

## 5. Statistical Comparison (McNemar's Test)

- **Models compared**: Random Forest vs Gradient Boosting
- **Chi² statistic**: 3.2000
- **p-value**: 0.073638
- **Significant (α=0.05)**: False

---

## 6. Classification Report

```
Classification Report — Random Forest
==================================================
              precision    recall  f1-score   support

    No NAFLD       0.94      0.91      0.93       640
       NAFLD       0.76      0.84      0.80       213

    accuracy                           0.89       853
   macro avg       0.85      0.87      0.86       853
weighted avg       0.90      0.89      0.89       853


Sensitivity (Recall) : 0.8404
Specificity          : 0.9094
Precision (PPV)      : 0.7553
NPV                  : 0.9448
F1-score             : 0.7956
Accuracy             : 0.8921
```

---

## 7. Generated Figures

- `model_comparison_chart.png`
- `roc_curves.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `shap_summary.png`

All figures are saved at 300 DPI in `results/figures/`.

---

## 8. File Cleanup Recommendations

The following files in `figures/` are now **superseded** by the organized
outputs in `results/figures/` and can be safely removed:

- `figures/confusion_matrices_top5.png` (replaced by `results/figures/confusion_matrix.png`)
- `figures/confusion_matrix_best.png` (replaced)
- `figures/feature_importance_*.png` (consolidated into `results/figures/feature_importance.png`)
- `figures/model_comparison_chart.png` (replaced)
- `figures/roc_curves_top5.png` (replaced by `results/figures/roc_curves.png`)
- `figures/shap_bar.png`, `figures/shap_bar_plot.png` (superseded)
- `figures/shap_summary.png`, `figures/shap_summary_plot.png` (replaced)
- `catboost_info/` directory (training artifacts, not needed for final results)

---

## 9. Final Results Directory Structure

```
results/
    figures/
        model_comparison_chart.png
        roc_curves.png
        confusion_matrix.png
        feature_importance.png
        shap_summary.png
    model_comparison.csv
    top5_models.csv
    interpretable_model_comparison.csv
    mcnemar_test.csv
    classification_report.txt
    final_results_summary.md
```
