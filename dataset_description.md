Dataset: NHANES DEMO_J.xpt

Samples: ~4000+

Target variable:
disease
0 = No NAFLD
1 = NAFLD

Feature types:
- Demographic
- Clinical
- Lifestyle
- Biomarkers

Preprocessing:
- Median imputation for numeric features
- Mode imputation for categorical features
- OneHotEncoder for categorical features
- StandardScaler for numeric features

Split:
70% train
30% test
Stratified

Class imbalance handled with SMOTE.