"""
=============================================================================
Machine Learning-Based Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD)
Using Clinical and Lifestyle Data

Publication-quality ML pipeline for IEEE/Springer submission.

Data Source: NHANES DEMO_J.xpt (demographics) — raw SAS transport file.
The pipeline automatically cleans NHANES data, drops survey-design
columns, and prepares features for modelling.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FOLDS = 5
TARGET_COL = "disease"
DATA_PATH = os.path.join("data", "DEMO_J.xpt")
MODEL_SAVE_PATH = os.path.join("models", "best_nafld_model.pkl")
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

# Columns to drop from raw NHANES DEMO_J (IDs, survey design, weights)
DROP_COLS = [
    "SEQN",        # Respondent sequence number (ID)
    "SDDSRVYR",    # Data release cycle
    "RIDSTATR",    # Interview/examination status
    "WTINT2YR",    # Full sample 2-year interview weight
    "WTMEC2YR",    # Full sample 2-year MEC exam weight
    "SDMVPSU",     # Masked variance pseudo-PSU
    "SDMVSTRA",    # Masked variance pseudo-stratum
    "RIDAGEMN",    # Age in months (redundant with RIDAGEYR)
    "RIDEXAGM",    # Age in months at exam (redundant)
    "RIDEXMON",    # Six-month time period (exam scheduling)
    "SIALANG",     # Language of SP interview
    "SIAPROXY",    # Proxy used in SP interview
    "SIAINTRP",    # Interpreter used in SP interview
    "FIALANG",     # Language of family interview
    "FIAPROXY",    # Proxy used in family interview
    "FIAINTRP",    # Interpreter used in family interview
    "MIALANG",     # Language of MEC interview
    "MIAPROXY",    # Proxy used in MEC interview
    "MIAINTRP",    # Interpreter used in MEC interview
    "AIALANGA",    # Language of ACASI interview
]

np.random.seed(RANDOM_STATE)

for d in [FIGURES_DIR, RESULTS_DIR, "models"]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# STEP 1: DATA LOADING & PREPARATION
# ============================================================================
def load_data(path: str, target: str):
    """
    Load dataset from .xpt (SAS) file.
    If the target column 'disease' does not exist (raw NHANES DEMO_J),
    the pipeline automatically:
      1. Drops survey-design and ID columns.
      2. Derives a binary proxy target from available clinical indicators.
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)

    df = pd.read_sas(path, format="xport", encoding="utf-8")

    print(f"Raw dataset shape : {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head())
    print()

    # --- Drop survey-design / ID columns ---
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} survey-design / ID columns.")
    print(f"Remaining columns ({len(df.columns)}): {list(df.columns)}")

    # --- Handle target column ---
    if target in df.columns:
        print(f"\nTarget column '{target}' found in dataset.")
    else:
        print(f"\n⚠  Target column '{target}' NOT found in raw NHANES DEMO_J.")
        print("   Deriving binary proxy target from demographic risk factors.")
        print("   (Replace this with your actual NAFLD labels for publication.)\n")
        # Derive a proxy NAFLD-risk target using known demographic risk factors:
        #   Age ≥ 50, BMI proxy via household size indicators, diabetes indicator
        #   This is a DEMONSTRATION proxy — replace with clinical NAFLD diagnosis.
        np.random.seed(RANDOM_STATE)
        risk_score = np.zeros(len(df))
        # Age risk (RIDAGEYR ≥ 45)
        if "RIDAGEYR" in df.columns:
            risk_score += (df["RIDAGEYR"].fillna(0) >= 45).astype(float) * 0.35
        # Gender risk (male = 1, RIAGENDR == 1)
        if "RIAGENDR" in df.columns:
            risk_score += (df["RIAGENDR"].fillna(0) == 1).astype(float) * 0.15
        # Low income risk (INDFMPIR < 1.5)
        if "INDFMPIR" in df.columns:
            risk_score += (df["INDFMPIR"].fillna(5) < 1.5).astype(float) * 0.15
        # Low education risk
        if "DMDEDUC2" in df.columns:
            risk_score += (df["DMDEDUC2"].fillna(3) <= 2).astype(float) * 0.10
        # Add calibrated noise for realistic class balance (~25% positive)
        noise = np.random.uniform(0, 0.4, size=len(df))
        risk_score += noise
        threshold = np.percentile(risk_score, 75)
        df[target] = (risk_score >= threshold).astype(int)
        print(f"   Proxy target created with threshold={threshold:.3f}")

    # Drop rows where target is NaN
    df.dropna(subset=[target], inplace=True)

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features shape     : {X.shape}")
    print(f"Target shape       : {y.shape}")
    return X, y, df


# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
def identify_column_types(X: pd.DataFrame):
    """Automatically detect numerical and categorical columns."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Treat low-cardinality integer columns as categorical
    for col in num_cols.copy():
        if X[col].nunique() <= 10:
            cat_cols.append(col)
            num_cols.remove(col)

    print(f"Numerical columns  ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list):
    """Build a ColumnTransformer with imputation + encoding + scaling."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def print_class_distribution(y, label="Full"):
    """Print class distribution."""
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"\nClass distribution ({label}):")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt} ({cnt / total * 100:.1f}%)")
    ratio = counts.min() / counts.max()
    print(f"  Imbalance ratio: {ratio:.3f}")
    return ratio


# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
def split_data(X, y):
    """Stratified 70-30 train-test split."""
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN-TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set : {X_train.shape[0]} samples")
    print(f"Test set     : {X_test.shape[0]} samples")
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_test, "Test")

    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 4 & 5: MODEL DEFINITIONS + CROSS VALIDATION + EVALUATION
# ============================================================================
def get_base_models():
    """Return dictionary of 23 base models (Voting added later)."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Ridge Classifier": CalibratedClassifierCV(
            RidgeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            cv=3, method="sigmoid"
        ),
        "Lasso Logistic Regression": LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced",
            verbose=-1, force_col_wise=True
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200, random_state=RANDOM_STATE, verbose=0,
            auto_class_weights="Balanced"
        ),
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"),
            cv=3, method="sigmoid"
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "Bagging Classifier": BaggingClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SGD Classifier": CalibratedClassifierCV(
            SGDClassifier(
                max_iter=5000, random_state=RANDOM_STATE,
                class_weight="balanced", loss="modified_huber"
            ),
            cv=3, method="sigmoid"
        ),
        "Perceptron": CalibratedClassifierCV(
            Perceptron(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"),
            cv=3, method="sigmoid"
        ),
        "Passive Aggressive": CalibratedClassifierCV(
            PassiveAggressiveClassifier(
                max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"
            ),
            cv=3, method="sigmoid"
        ),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1000,
            random_state=RANDOM_STATE, early_stopping=True
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }
    return models


def cross_validate_model(model, X_train, y_train, cv):
    """Run stratified k-fold CV and return mean accuracy and ROC-AUC."""
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc"}
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
    )
    return (
        cv_results["test_accuracy"].mean(),
        cv_results["test_roc_auc"].mean(),
    )


def evaluate_on_test(model, X_test, y_test):
    """Evaluate fitted model on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test, preprocessor):
    """Train 24 models, cross-validate, evaluate, return results DataFrame."""
    print("\n" + "=" * 70)
    print("STEP 4 & 5: CROSS VALIDATION + MODEL TRAINING")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Pre-process once for efficiency
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
    print(f"\nAfter SMOTE — Training set: {X_train_res.shape[0]} samples")
    print_class_distribution(y_train_res, "Train-SMOTE")

    base_models = get_base_models()
    results = []
    fitted_models = {}

    for i, (name, model) in enumerate(base_models.items(), 1):
        print(f"\n[{i:2d}/24] Training: {name} ...", end=" ", flush=True)
        try:
            cv_acc, cv_auc = cross_validate_model(model, X_train_res, y_train_res, cv)
            model.fit(X_train_res, y_train_res)
            test_metrics = evaluate_on_test(model, X_test_processed, y_test)

            results.append({
                "Model": name,
                "CV Accuracy": cv_acc,
                "CV ROC-AUC": cv_auc,
                **test_metrics,
            })
            fitted_models[name] = model
            print(f"ROC-AUC={test_metrics['Test ROC-AUC']:.4f} ✓")
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({
                "Model": name,
                "CV Accuracy": np.nan,
                "CV ROC-AUC": np.nan,
                "Test Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1-score": np.nan,
                "Test ROC-AUC": np.nan,
            })

    # --- Model 24: Voting Classifier (top 3 by CV ROC-AUC) ---
    print("\n[24/24] Building Voting Classifier from top 3 CV models ...", end=" ", flush=True)
    results_df_temp = pd.DataFrame(results).dropna(subset=["CV ROC-AUC"])
    top3 = results_df_temp.nlargest(3, "CV ROC-AUC")["Model"].tolist()
    print(f"({', '.join(top3)})")

    voting_estimators = [(n, fitted_models[n]) for n in top3 if n in fitted_models]
    voting = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)

    try:
        cv_acc, cv_auc = cross_validate_model(voting, X_train_res, y_train_res, cv)
        voting.fit(X_train_res, y_train_res)
        test_metrics = evaluate_on_test(voting, X_test_processed, y_test)

        results.append({
            "Model": "Voting Classifier",
            "CV Accuracy": cv_acc,
            "CV ROC-AUC": cv_auc,
            **test_metrics,
        })
        fitted_models["Voting Classifier"] = voting
        print(f"  ROC-AUC={test_metrics['Test ROC-AUC']:.4f} ✓")
    except Exception as e:
        print(f"  FAILED ({e})")

    results_df = pd.DataFrame(results)
    return results_df, fitted_models, X_train_res, y_train_res, X_test_processed


# ============================================================================
# STEP 6: PERFORMANCE COMPARISON
# ============================================================================
def performance_comparison(results_df: pd.DataFrame):
    """Rank models and display comparison table."""
    print("\n" + "=" * 70)
    print("STEP 6: PERFORMANCE COMPARISON")
    print("=" * 70)

    ranked = results_df.sort_values("Test ROC-AUC", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n", ranked.to_string())

    print("\n\n--- TOP 5 MODELS ---")
    for i, row in ranked.head(5).iterrows():
        print(f"  {i}. {row['Model']} — Test ROC-AUC: {row['Test ROC-AUC']:.4f}")

    # Save to CSV
    ranked.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    print(f"\nComparison table saved → {RESULTS_DIR}/model_comparison.csv")

    return ranked


# ============================================================================
# STEP 7: ROC CURVE PLOTTING
# ============================================================================
def plot_roc_curves(ranked_df, fitted_models, X_test, y_test):
    """Plot ROC curves for top 5 models."""
    print("\n" + "=" * 70)
    print("STEP 7: ROC CURVE PLOTTING")
    print("=" * 70)

    top5 = ranked_df.head(5)["Model"].tolist()

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(top5))

    for name, color in zip(top5, colors):
        model = fitted_models[name]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curves — Top 5 Models", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "roc_curves_top5.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved → {save_path}")


# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(fitted_models, feature_names):
    """Plot top 15 feature importances for tree-based models."""
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE IMPORTANCE")
    print("=" * 70)

    tree_models = {
        "Random Forest": fitted_models.get("Random Forest"),
        "XGBoost": fitted_models.get("XGBoost"),
        "LightGBM": fitted_models.get("LightGBM"),
        "CatBoost": fitted_models.get("CatBoost"),
    }

    for name, model in tree_models.items():
        if model is None:
            continue

        importances = model.feature_importances_
        n_features = min(len(feature_names), len(importances))
        feat_names = feature_names[:n_features]
        imp = importances[:n_features]

        indices = np.argsort(imp)[::-1][:15]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=imp[indices],
            y=[feat_names[i] for i in indices],
            palette="viridis",
        )
        plt.title(f"Top 15 Feature Importance — {name}", fontsize=14, fontweight="bold")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(FIGURES_DIR, f"feature_importance_{name.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {name} → {save_path}")


# ============================================================================
# STEP 9: BEST MODEL SELECTION & SAVING
# ============================================================================
def select_and_save_best(ranked_df, fitted_models, X_train, y_train):
    """Select best model, retrain on full training data, and save."""
    print("\n" + "=" * 70)
    print("STEP 9: BEST MODEL SELECTION")
    print("=" * 70)

    best_row = ranked_df.iloc[0]
    best_name = best_row["Model"]
    best_auc = best_row["Test ROC-AUC"]

    print(f"\n  🏆 Best Model  : {best_name}")
    print(f"  🎯 Test ROC-AUC: {best_auc:.4f}")

    best_model = fitted_models[best_name]
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"  💾 Model saved → {MODEL_SAVE_PATH}")

    return best_name, best_auc


# ============================================================================
# STEP 10: GET FEATURE NAMES AFTER PREPROCESSING
# ============================================================================
def get_feature_names(preprocessor, num_cols, cat_cols):
    """Extract feature names from fitted ColumnTransformer."""
    feature_names = list(num_cols)
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names += cat_feature_names
        except Exception:
            feature_names += [f"cat_{i}" for i in range(len(cat_cols))]
    return np.array(feature_names)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  NAFLD Prediction Pipeline — 24 ML Models                          ║")
    print("║  Machine Learning-Based Prediction of NAFLD                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    # STEP 1: Load
    X, y, df = load_data(DATA_PATH, TARGET_COL)

    # STEP 2: Preprocess
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    print_class_distribution(y, "Full Dataset")
    num_cols, cat_cols = identify_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # STEP 3: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # STEPS 4-5: CV + Training
    results_df, fitted_models, X_train_res, y_train_res, X_test_proc = (
        train_and_evaluate_all(X_train, X_test, y_train, y_test, preprocessor)
    )

    # STEP 6: Comparison
    ranked = performance_comparison(results_df)

    # STEP 7: ROC Curves
    plot_roc_curves(ranked, fitted_models, X_test_proc, y_test)

    # STEP 8: Feature Importance
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    plot_feature_importance(fitted_models, feature_names)

    # STEP 9: Save best model
    best_name, best_auc = select_and_save_best(ranked, fitted_models, X_train_res, y_train_res)

    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Best Model  : {best_name}")
    print(f"  ROC-AUC     : {best_auc:.4f}")
    print(f"  Saved at    : {MODEL_SAVE_PATH}")
    print(f"  Figures     : {FIGURES_DIR}/")
    print(f"  Results     : {RESULTS_DIR}/model_comparison.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
