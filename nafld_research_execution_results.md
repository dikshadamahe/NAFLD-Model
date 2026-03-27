# NAFLD Research Analysis - Extended Evaluation Suite

Publication-ready execution report for IEEE/Springer documentation.

## Execution Metadata

- Date: March 13, 2026
- Status: Completed
- Analyses executed: 9/9
- Best model: Random Forest
- Best ROC-AUC: 0.9644

## Environment and Data Loading

- Xtr_smote shape: (2984, 16)
- Xte shape: (853, 16)
- Total features: 16

## Model Training Summary (24 Models)

| # | Model | Test ROC-AUC |
|---|---|---:|
| 1 | Logistic Regression | 0.9107 |
| 2 | Ridge Classifier | 0.9070 |
| 3 | Lasso Logistic Regression | 0.9102 |
| 4 | Decision Tree | 0.8545 |
| 5 | Random Forest | 0.9644 |
| 6 | Extra Trees | 0.9462 |
| 7 | Gradient Boosting | 0.9638 |
| 8 | XGBoost | 0.9631 |
| 9 | LightGBM | 0.9612 |
| 10 | CatBoost | 0.9632 |
| 11 | SVM (Linear) | 0.9099 |
| 12 | SVM (RBF) | 0.9321 |
| 13 | KNN | 0.8675 |
| 14 | Gaussian Naive Bayes | 0.8898 |
| 15 | AdaBoost | 0.9574 |
| 16 | Bagging Classifier | 0.9629 |
| 17 | SGD Classifier | 0.8973 |
| 18 | Perceptron | 0.8973 |
| 19 | Passive Aggressive | 0.9111 |
| 20 | QDA | 0.9023 |
| 21 | LDA | 0.9070 |
| 22 | MLP Classifier | 0.9213 |
| 23 | Hist Gradient Boosting | 0.9630 |
| 24 | Voting Classifier | 0.9583 |

## Analysis 1: Model Ranking (Top 10)

| Rank | Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Precision | Recall | F1-score | Test ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Random Forest | 0.9337 | 0.9887 | 0.8921 | 0.7553 | 0.8404 | 0.7956 | 0.9644 |
| 2 | Gradient Boosting | 0.9239 | 0.9867 | 0.8769 | 0.7160 | 0.8404 | 0.7732 | 0.9638 |
| 3 | CatBoost | 0.9373 | 0.9885 | 0.8839 | 0.7262 | 0.8592 | 0.7871 | 0.9632 |
| 4 | XGBoost | 0.9440 | 0.9895 | 0.8980 | 0.7890 | 0.8075 | 0.7981 | 0.9631 |
| 5 | Hist Gradient Boosting | 0.9414 | 0.9907 | 0.8898 | 0.7742 | 0.7887 | 0.7814 | 0.9630 |
| 6 | Bagging Classifier | 0.9323 | 0.9871 | 0.8863 | 0.7500 | 0.8169 | 0.7820 | 0.9629 |
| 7 | LightGBM | 0.9420 | 0.9906 | 0.8945 | 0.7808 | 0.8028 | 0.7917 | 0.9612 |
| 8 | Voting Classifier | 0.9437 | 0.9921 | 0.8957 | 0.7870 | 0.7981 | 0.7925 | 0.9583 |
| 9 | AdaBoost | 0.9199 | 0.9816 | 0.8769 | 0.7077 | 0.8638 | 0.7780 | 0.9574 |
| 10 | Extra Trees | 0.9450 | 0.9906 | 0.8722 | 0.7301 | 0.7746 | 0.7517 | 0.9462 |

Top 5 by Test ROC-AUC:
1. Random Forest (0.9644)
2. Gradient Boosting (0.9638)
3. CatBoost (0.9632)
4. XGBoost (0.9631)
5. Hist Gradient Boosting (0.9630)

Saved outputs:
- results/model_comparison.csv
- results/top5_models.csv

## Analysis 2: ROC Curves (Top 5)

Saved output:
- figures/roc_curves_top5.png

## Analysis 3: Confusion Matrix (Best Model)

Best model: Random Forest

- Accuracy: 0.8921
- Sensitivity (Recall): 0.8404
- Specificity: 0.9094
- Precision (PPV): 0.7553
- NPV: 0.9448
- F1-score: 0.7956
- Confusion matrix values: TP=179, FP=58, FN=34, TN=582

Saved outputs:
- figures/confusion_matrix_best.png
- results/classification_report.txt

## Analysis 4: Feature Importance

Saved outputs:
- figures/feature_importance_random_forest.png
- figures/feature_importance_xgboost.png
- figures/feature_importance_gradient_boosting.png
- figures/feature_importance_lightgbm.png
- figures/feature_importance_comparison.png

## Analysis 5: SHAP Explainability

Model used: XGBoost

Saved outputs:
- figures/shap_summary_plot.png
- figures/shap_bar_plot.png

## Analysis 6: Interpretable Simple Model (Top 5 Features)

Top 5 features:
- Age
- Glucose
- Waist_Circumference
- BMI
- Gender_2.0

Comparison:
- Logistic Regression (5 features): Accuracy = 0.8218, ROC-AUC = 0.9096
- Random Forest: Accuracy = 0.8921, ROC-AUC = 0.9644
- AUC drop from best model: 0.0548

Logistic Regression coefficients:
- Gender_2.0: -1.9173 (decrease in risk)
- Age: +1.8663 (increase in risk)
- Glucose: +0.7694 (increase in risk)
- Waist_Circumference: +0.7361 (increase in risk)
- BMI: +0.5348 (increase in risk)

Saved output:
- results/interpretable_model_comparison.csv

## Analysis 7: External Validation

External dataset not found at data/external_validation.csv.

Required format for external validation file:
- Include target column: NAFLD
- Numerical columns expected: Age, BMI, Waist_Circumference, Total_Cholesterol, LDL, HDL, Triglycerides, ALT, AST, Glucose
- Categorical columns expected: Gender, Ethnicity

## Analysis 8: Statistical Comparison (McNemar's Test)

Compared models:
- Model 1: Random Forest
- Model 2: Gradient Boosting

Contingency table:
- Both correct: 732
- Model 1 correct only: 29
- Model 2 correct only: 16
- Both wrong: 76

Statistics:
- Chi-square: 3.2000
- p-value: 0.073638
- alpha: 0.05
- Significant difference: No

Interpretation: No statistically significant difference between the two models.

Saved output:
- results/mcnemar_test.csv

## Analysis 9: Save All Results

Saved model:
- models/best_nafld_model.pkl

Primary generated artifacts:
- data/merged_nhanes_dataset.csv
- data/nafld_final_dataset.csv
- figures/confusion_matrix_best.png
- figures/feature_importance_comparison.png
- figures/feature_importance_gradient_boosting.png
- figures/feature_importance_lightgbm.png
- figures/feature_importance_random_forest.png
- figures/feature_importance_xgboost.png
- figures/roc_curves_top5.png
- figures/shap_bar_plot.png
- figures/shap_summary_plot.png
- results/classification_report.txt
- results/interpretable_model_comparison.csv
- results/mcnemar_test.csv
- results/model_comparison.csv
- results/top5_models.csv
- models/best_nafld_model.pkl

## Warning Notes During Execution

scikit-learn warning observed for AdaBoost:
- FutureWarning: The SAMME.R algorithm is deprecated and will be removed in scikit-learn 1.6.
- Suggested action from warning: use SAMME explicitly to avoid this deprecation warning.

## Final Outcome

- Research analysis complete
- Best model: Random Forest
- Best ROC-AUC: 0.9644
- Figures directory: figures/
- Results directory: results/
- Serialized model: models/best_nafld_model.pkl
