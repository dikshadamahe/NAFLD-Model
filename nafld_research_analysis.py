"""
=============================================================================
NAFLD Research-Level Analysis
Extended Evaluation Suite for Publication (IEEE / Springer)

v2.3 RELEASE — ALL BUGS FROM v2.2 FIXED
Prerequisites:
  - Place dataset at data/nafld_final_dataset.csv  OR
                     data/merged_nhanes_dataset.csv

Complete Analysis Pipeline:
  1. Model Ranking & Comparison Table
  2. ROC Curve Visualization (publication-quality)
  3. Confusion Matrix + Sensitivity/Specificity for Best Model
  4. Feature Importance (4 tree-based models + cross-model comparison)
  5. SHAP Explainability (summary + bar plot)
  6. Interpretable Simple Model (top-5-feature Logistic Regression)
  7. External Validation Stub (Framingham-ready)
  8. Statistical Comparison — McNemar's Test (top 2 models)
  9. Save All Results (CSV, PNG @ 300 dpi, .pkl)
  10. Hyperparameter Tuning for Top 3 Models

FIXES APPLIED v2.3 (over v2.2):
  FIX-A : analysis_9_save_all now refits on SMOTE-balanced data (Xtr_s/ytr_s),
           matching the training regime of the best model.
  FIX-B : tune_top_models now receives and uses Xtr_s/ytr_s so tuned models
           are trained on the same SMOTE-balanced data as the originals.
  FIX-C : _extract_classifier is now recursive, correctly unwrapping nested
           CalibratedClassifierCV(ImbPipeline(...)) structures.
  FIX-D : detect_column_types no longer mis-classifies binary numeric columns
           (0/1) as categorical. Only object/category dtype columns are treated
           as categorical; low-cardinality numeric columns stay numeric.
  FIX-E : VotingClassifier is rebuilt from the post-tuned top-3 models so it
           actually benefits from hyperparameter optimisation.
  FIX-F : GridSearchCV in tune_top_models uses N_FOLDS (not hardcoded cv=3)
           for consistency with the rest of the pipeline.
  FIX-G : analysis_3_confusion_matrix guards against 1×1 confusion matrix
           (single-class predictions) so cm.ravel() never crashes.
  FIX-H : SHAP branch guards against single-element list edge case.
  FIX-I : analysis_8_mcnemar guards against failed models missing from fitted{}.
  FIX-J : McNemar continuity correction is explicitly documented in output.

=============================================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from scipy.stats import chi2 as chi2_dist
from mlxtend.evaluate import mcnemar_table

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
)

# ── Global Settings ─────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.30
N_FOLDS    = 5
TARGET     = "NAFLD"
DATA_PATHS = [
    os.path.join("data", "nafld_final_dataset.csv"),
    os.path.join("data", "merged_nhanes_dataset.csv"),
]
FIG_DIR = "figures"
RES_DIR = "results"
MDL_DIR = "models"

DROP_COLS = []   # Add column names here to exclude from modelling

np.random.seed(SEED)
for _d in [FIG_DIR, RES_DIR, MDL_DIR]:
    os.makedirs(_d, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    data_path = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError(f"No dataset found. Tried: {DATA_PATHS}")

    df = pd.read_csv(data_path)
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing required target column '{TARGET}' in {data_path}. "
            "Provide clinical NAFLD labels before running analysis."
        )

    df.dropna(subset=[TARGET], inplace=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def detect_column_types(X):
    """
    FIX-D: Only treat genuine object/category dtype columns as categorical.
    Low-cardinality numeric columns (e.g. binary 0/1 flags like hypertension,
    diabetes) remain numeric to avoid spurious one-hot encoding and feature
    name mismatches.
    """
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=["number"]).columns.tolist()
    # Do NOT promote numeric columns to categorical based on cardinality —
    # binary clinical flags are better handled as numeric (0/1) features.
    return num, cat


def build_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        num_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                      drop="first", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(preprocessor, num_cols, cat_cols):
    """
    Returns a plain Python list[str] — no np.str_ wrappers.
    """
    names = [str(c) for c in num_cols]
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            names += [str(f) for f in ohe.get_feature_names_out(cat_cols)]
        except Exception:
            names += [f"cat_{i}" for i in range(len(cat_cols))]
    return names  # plain list[str]


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def _wrap(clf):
    """Wrap classifiers without predict_proba using CalibratedClassifierCV."""
    return CalibratedClassifierCV(clf, cv=3, method="sigmoid")


def _ensure_calibrated(clf):
    """Wraps classifier in CalibratedClassifierCV if it lacks predict_proba."""
    if hasattr(clf, "predict_proba"):
        return clf
    return CalibratedClassifierCV(clf, cv=3, method="sigmoid")


def _extract_classifier(estimator):
    """
    FIX-C: Recursively extract the base classifier from any nesting of
    ImbPipeline and/or CalibratedClassifierCV wrappers.

    Handles cases such as:
      CalibratedClassifierCV(ImbPipeline([..., ("classifier", RFC())]))
      ImbPipeline([..., ("classifier", CalibratedClassifierCV(RFC()))])
    """
    if estimator is None:
        return None
    # Unwrap ImbPipeline
    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        return _extract_classifier(estimator.named_steps["classifier"])
    # Unwrap CalibratedClassifierCV
    if isinstance(estimator, CalibratedClassifierCV):
        return _extract_classifier(estimator.estimator)
    return estimator


def _validate_cv_fold_distribution(X, y, cv, smote_k=None):
    """Validate that each CV fold has enough samples per class.

    If SMOTE is enabled, each training fold must have at least k+1 samples
    for every class so that neighbor search is valid.
    """
    min_samples_required = (smote_k + 1) if (smote_k is not None and smote_k > 0) else 2
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        class_counts = y_train_fold.value_counts()
        if (class_counts < min_samples_required).any():
            raise ValueError(
                f"CV Fold {fold_idx}: Class imbalance too severe. "
                f"Min {min_samples_required} samples per class required. "
                f"Got: {dict(class_counts)}"
            )
    print(
        f"  ✓ CV fold distribution validated "
        f"({cv.get_n_splits()} folds OK; min/class/fold={min_samples_required})"
    )


def _build_cv_pipeline(clf, smote_k=None):
    """Leakage-safe CV pipeline: SMOTE (optional) → classifier."""
    if smote_k is not None:
        return ImbPipeline([
            ("smote",      SMOTE(random_state=SEED, k_neighbors=smote_k)),
            ("classifier", clf),
        ])
    return clf


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (23 base classifiers + 1 VotingClassifier = 24)
# ═══════════════════════════════════════════════════════════════════════════
def get_base_models():
    """Return ordered dict of 23 base classifiers."""
    return {
        "Logistic Regression":       _ensure_calibrated(LogisticRegression(max_iter=5000, random_state=SEED, class_weight="balanced")),
        "Ridge Classifier":          _wrap(RidgeClassifier(random_state=SEED, class_weight="balanced")),
        "Lasso Logistic Regression": _ensure_calibrated(LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=SEED, class_weight="balanced")),
        "Decision Tree":             _ensure_calibrated(DecisionTreeClassifier(random_state=SEED, class_weight="balanced")),
        "Random Forest":             _ensure_calibrated(RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", n_jobs=-1)),
        "Extra Trees":               _ensure_calibrated(ExtraTreesClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", n_jobs=-1)),
        "Gradient Boosting":         _ensure_calibrated(GradientBoostingClassifier(n_estimators=200, random_state=SEED)),
        "XGBoost":                   _ensure_calibrated(XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, verbosity=0)),
        "LightGBM":                  _ensure_calibrated(LGBMClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", verbose=-1, force_col_wise=True)),
        "CatBoost":                  _ensure_calibrated(CatBoostClassifier(iterations=200, random_state=SEED, verbose=0, auto_class_weights="Balanced")),
        "SVM (Linear)":              _wrap(LinearSVC(max_iter=5000, random_state=SEED, class_weight="balanced")),
        "SVM (RBF)":                 _ensure_calibrated(SVC(kernel="rbf", probability=True, random_state=SEED, class_weight="balanced")),
        "KNN":                       _ensure_calibrated(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Gaussian Naive Bayes":      _ensure_calibrated(GaussianNB()),
        "AdaBoost":                  _ensure_calibrated(AdaBoostClassifier(n_estimators=200, random_state=SEED, algorithm="SAMME")),
        "Bagging Classifier":        _ensure_calibrated(BaggingClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        "SGD Classifier":            _wrap(SGDClassifier(max_iter=5000, random_state=SEED, class_weight="balanced", loss="modified_huber")),
        "Perceptron":                _wrap(Perceptron(max_iter=5000, random_state=SEED, class_weight="balanced")),
        "Passive Aggressive":        _wrap(SGDClassifier(loss="hinge", penalty=None, learning_rate="constant", eta0=1.0, max_iter=5000, random_state=SEED, class_weight="balanced")),
        "QDA":                       _ensure_calibrated(QuadraticDiscriminantAnalysis(reg_param=0.5)),
        "LDA":                       _ensure_calibrated(LinearDiscriminantAnalysis()),
        "MLP Classifier":            _ensure_calibrated(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=SEED, early_stopping=True)),
        "Hist Gradient Boosting":    _ensure_calibrated(HistGradientBoostingClassifier(max_iter=200, random_state=SEED, class_weight="balanced")),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING FOR TOP 3 MODELS
# ═══════════════════════════════════════════════════════════════════════════
def tune_top_models(ranked, fitted, base_mdls, Xtr_p, ytr, Xtr_s, ytr_s, Xte_p, yte, smote_k=None):
    """
    Fine-tune top 3 models with GridSearchCV.

    FIX-B: Receives Xtr_s/ytr_s (SMOTE-balanced) and trains final tuned
           models on those, matching the training regime of untuned models.
    FIX-F: Uses N_FOLDS (not hardcoded 3) for consistency.

    GridSearchCV inner CV uses the unbalanced Xtr_p/ytr with a leakage-safe
    SMOTE pipeline so that each inner fold applies SMOTE independently.
    If smote_k is None, tuning runs without SMOTE to match the global policy.
    Final fit of the best estimator uses the full Xtr_s/ytr_s.
    """
    print(f"\n{'═'*72}\n  HYPERPARAMETER TUNING (top 3 models)\n{'═'*72}")

    top3 = ranked.head(3)["Model"].tolist()

    param_grids = {
        "Random Forest": {
            "n_estimators":      [150, 200, 250],
            "max_depth":         [8, 12, 15, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators":      [150, 200, 250],
            "learning_rate":     [0.05, 0.08, 0.1],
            "max_depth":         [3, 5, 7],
            "min_samples_split": [2, 5],
            "subsample":         [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators":      [150, 200, 250],
            "max_depth":         [3, 5, 7],
            "learning_rate":     [0.05, 0.1, 0.15],
            "subsample":         [0.8, 1.0],
            "colsample_bytree":  [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators":  [150, 200, 250],
            "max_depth":     [5, 7, 10],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves":    [20, 31, 50],
            "subsample":     [0.8, 1.0],
        },
        "AdaBoost": {
            "n_estimators":  [150, 200, 250],
            "learning_rate": [0.5, 0.8, 1.0],
        },
        # Voting Classifier is rebuilt post-tuning (FIX-E), no grid needed.
        "Voting Classifier": {},
    }

    tuned_results = []
    for name in top3:
        if name not in param_grids:
            print(f"  [{name:<30s}] No tuning grid defined; skipping.")
            orig_result = ranked[ranked["Model"] == name]
            if not orig_result.empty:
                tuned_results.append(orig_result.iloc[0].to_dict())
            continue

        if param_grids[name] == {}:
            print(f"  [{name:<30s}] Ensemble model; rebuilt post-tuning (FIX-E).")
            orig_result = ranked[ranked["Model"] == name]
            if not orig_result.empty:
                tuned_results.append(orig_result.iloc[0].to_dict())
            continue

        if name not in base_mdls:
            print(f"  [{name:<30s}] Model not in base_mdls; skipping.")
            continue

        try:
            base_clf = _extract_classifier(clone(base_mdls[name]))

            # FIX-F: Use N_FOLDS for GridSearchCV inner CV (was hardcoded 3)
            # Wrap in leakage-safe SMOTE pipeline for inner CV folds
            smote_k_inner = smote_k
            cv_pipe = _build_cv_pipeline(clone(base_clf), smote_k_inner)

            # Prefix param keys if using ImbPipeline
            if smote_k_inner is not None:
                grid = {f"classifier__{k}": v for k, v in param_grids[name].items()}
            else:
                grid = param_grids[name]

            gs = GridSearchCV(
                cv_pipe, grid,
                cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
                scoring="roc_auc", n_jobs=-1, verbose=0,
            )
            gs.fit(Xtr_p, ytr)

            # Extract best raw classifier and refit on SMOTE-balanced full set
            best_raw = _extract_classifier(gs.best_estimator_)
            best_clf = _ensure_calibrated(clone(best_raw).set_params(
                **{k: v for k, v in gs.best_params_.items()
                   if not k.startswith("smote") and not k.startswith("classifier__")}
            ) if not smote_k_inner else clone(best_raw))

            # FIX-B: Fit final tuned model on SMOTE-balanced data
            best_clf.fit(Xtr_s, ytr_s)

            yp   = best_clf.predict(Xte_p)
            ypr  = best_clf.predict_proba(Xte_p)[:, 1]
            auc_tuned = roc_auc_score(yte, ypr)
            acc_tuned = accuracy_score(yte, yp)

            orig_row  = ranked[ranked["Model"] == name].iloc[0]
            auc_orig  = orig_row["Test ROC-AUC"]
            auc_gain  = auc_tuned - auc_orig

            result_row = orig_row.to_dict()
            result_row["Test ROC-AUC"]  = auc_tuned
            result_row["Test Accuracy"] = acc_tuned
            result_row["Tuning_Gain"]   = auc_gain
            tuned_results.append(result_row)

            # Store tuned model
            fitted[name] = best_clf

            sign = "+" if auc_gain >= 0 else ""
            print(f"  [{name:<30s}] AUC {auc_orig:.4f} → {auc_tuned:.4f}  ({sign}{auc_gain:.4f}) ✓")

        except Exception as e:
            print(f"  [{name:<30s}] Tuning failed ({e}); keeping original.")
            orig_result = ranked[ranked["Model"] == name]
            if not orig_result.empty:
                tuned_results.append(orig_result.iloc[0].to_dict())

    # Rebuild ranked dataframe with tuned results
    other_rows  = ranked[~ranked["Model"].isin(top3)].to_dict("records")
    ranked_tuned = (
        pd.DataFrame(tuned_results + other_rows)
        .sort_values("Test ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    ranked_tuned.index      = ranked_tuned.index + 1
    ranked_tuned.index.name = "Rank"

    # ── FIX-E: Rebuild VotingClassifier from POST-TUNED top-3 ──────────
    vc_label  = "Voting Classifier"
    new_top3  = ranked_tuned.head(3)["Model"].tolist()
    # Exclude VotingClassifier itself from its own components
    vc_sources = [n for n in new_top3 if n != vc_label and n in fitted][:3]

    if len(vc_sources) >= 2:
        print(f"\n  [FIX-E] Rebuilding {vc_label} from post-tuned top-3: {vc_sources}")
        vc_estimators = [(n, clone(fitted[n])) for n in vc_sources]
        vc = VotingClassifier(estimators=vc_estimators, voting="soft", n_jobs=-1)
        try:
            vc.fit(Xtr_s, ytr_s)
            yp_vc   = vc.predict(Xte_p)
            ypr_vc  = vc.predict_proba(Xte_p)[:, 1]
            auc_vc  = roc_auc_score(yte, ypr_vc)
            acc_vc  = accuracy_score(yte, yp_vc)
            f1_vc   = f1_score(yte, yp_vc, zero_division=0)
            prec_vc = precision_score(yte, yp_vc, zero_division=0)
            rec_vc  = recall_score(yte, yp_vc, zero_division=0)
            fitted[vc_label] = vc

            # Update or insert VotingClassifier row
            vc_row = {
                "Model":         vc_label,
                "CV Accuracy":   ranked_tuned.loc[ranked_tuned["Model"] == vc_label, "CV Accuracy"].values[0]
                                 if vc_label in ranked_tuned["Model"].values else np.nan,
                "CV ROC-AUC":    ranked_tuned.loc[ranked_tuned["Model"] == vc_label, "CV ROC-AUC"].values[0]
                                 if vc_label in ranked_tuned["Model"].values else np.nan,
                "Test Accuracy": acc_vc,
                "Precision":     prec_vc,
                "Recall":        rec_vc,
                "F1-score":      f1_vc,
                "Test ROC-AUC":  auc_vc,
                "Tuning_Gain":   0.0,
            }
            # Replace existing VC row and re-sort
            ranked_tuned = ranked_tuned[ranked_tuned["Model"] != vc_label]
            ranked_tuned = (
                pd.concat([ranked_tuned, pd.DataFrame([vc_row])], ignore_index=True)
                .sort_values("Test ROC-AUC", ascending=False)
                .reset_index(drop=True)
            )
            ranked_tuned.index      = ranked_tuned.index + 1
            ranked_tuned.index.name = "Rank"
            print(f"         Rebuilt {vc_label}  AUC={auc_vc:.4f} ✓")
        except Exception as e:
            print(f"         Rebuild failed ({e}); keeping original {vc_label}.")

    print(f"\n  [NOTE] Tuning applied to top 3 models; {vc_label} rebuilt from tuned components.")
    return ranked_tuned, fitted


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN ALL 24 MODELS
# ═══════════════════════════════════════════════════════════════════════════
def train_all_models(Xtr_p, ytr, Xte_p, yte, smote_k=None):
    """
    Leakage-safe training with proper stratification.
    Returns: ranked, fitted, Xtr_s, ytr_s, base_mdls
    """
    cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    base_mdls = get_base_models()
    rows, fitted = [], {}

    # Apply SMOTE once on full training set for final model training
    if smote_k is not None:
        try:
            sm = SMOTE(random_state=SEED, k_neighbors=smote_k)
            Xtr_s, ytr_s = sm.fit_resample(Xtr_p, ytr)
        except Exception as e:
            print(f"  [WARNING] SMOTE failed: {e}. Using original training data.")
            Xtr_s, ytr_s = Xtr_p.copy(), ytr.copy()
    else:
        Xtr_s, ytr_s = Xtr_p.copy(), ytr.copy()

    total = len(base_mdls) + 1   # 23 base + 1 VotingClassifier = 24

    # ── Models 1-23: base classifiers ─────────────────────────────────
    for i, (name, clf) in enumerate(base_mdls.items(), 1):
        try:
            cv_estimator = _build_cv_pipeline(clone(clf), smote_k)
            cvr = cross_validate(
                cv_estimator, Xtr_p, ytr, cv=cv,
                scoring={"acc": "accuracy", "auc": "roc_auc"},
                n_jobs=-1, error_score="raise",
            )
            clf.fit(Xtr_s, ytr_s)
            yp  = clf.predict(Xte_p)
            ypr = clf.predict_proba(Xte_p)[:, 1]

            row = {
                "Model":         name,
                "CV Accuracy":   cvr["test_acc"].mean(),
                "CV ROC-AUC":    cvr["test_auc"].mean(),
                "Test Accuracy": accuracy_score(yte, yp),
                "Precision":     precision_score(yte, yp, zero_division=0),
                "Recall":        recall_score(yte, yp, zero_division=0),
                "F1-score":      f1_score(yte, yp, zero_division=0),
                "Test ROC-AUC":  roc_auc_score(yte, ypr),
            }
            rows.append(row)
            fitted[name] = clf
            print(f"  [{i:2d}/{total}] {name:<32s}  AUC={row['Test ROC-AUC']:.4f} ✓")

        except Exception as e:
            print(f"  [{i:2d}/{total}] {name:<32s}  FAILED ({e})")
            rows.append({"Model": name, **{k: np.nan for k in
                ["CV Accuracy", "CV ROC-AUC", "Test Accuracy",
                 "Precision", "Recall", "F1-score", "Test ROC-AUC"]}})

    # ── Model 24: VotingClassifier ─────────────────────────────────────
    tmp  = pd.DataFrame(rows).dropna(subset=["CV ROC-AUC"])
    top3 = tmp.nlargest(3, "CV ROC-AUC")["Model"].tolist()
    vc_label = "Voting Classifier"
    print(f"\n  [{total:2d}/{total}] Building {vc_label} from top 3: {top3}")

    vc_estimators = [(n, clone(base_mdls[n])) for n in top3 if n in base_mdls]
    vc = VotingClassifier(estimators=vc_estimators, voting="soft", n_jobs=-1)

    try:
        vc_cv = _build_cv_pipeline(clone(vc), smote_k)
        cvr = cross_validate(
            vc_cv, Xtr_p, ytr, cv=cv,
            scoring={"acc": "accuracy", "auc": "roc_auc"},
            n_jobs=-1,
        )
        vc.fit(Xtr_s, ytr_s)
        yp  = vc.predict(Xte_p)
        ypr = vc.predict_proba(Xte_p)[:, 1]

        row = {
            "Model":         vc_label,
            "CV Accuracy":   cvr["test_acc"].mean(),
            "CV ROC-AUC":    cvr["test_auc"].mean(),
            "Test Accuracy": accuracy_score(yte, yp),
            "Precision":     precision_score(yte, yp, zero_division=0),
            "Recall":        recall_score(yte, yp, zero_division=0),
            "F1-score":      f1_score(yte, yp, zero_division=0),
            "Test ROC-AUC":  roc_auc_score(yte, ypr),
        }
        rows.append(row)
        fitted[vc_label] = vc
        print(f"         AUC={row['Test ROC-AUC']:.4f} ✓")

    except Exception as e:
        print(f"         FAILED ({e})")
        rows.append({"Model": vc_label, **{k: np.nan for k in
            ["CV Accuracy", "CV ROC-AUC", "Test Accuracy",
             "Precision", "Recall", "F1-score", "Test ROC-AUC"]}})

    ranked = (
        pd.DataFrame(rows)
        .sort_values("Test ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index      = ranked.index + 1
    ranked.index.name = "Rank"

    return ranked, fitted, Xtr_s, ytr_s, base_mdls


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_1_rank_models(ranked):
    print(f"\n{'═'*72}\n  ANALYSIS 1 : MODEL RANKING\n{'═'*72}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(ranked.to_string())

    print("\n  ── TOP 5 MODELS ──")
    for i, r in ranked.head(5).iterrows():
        print(f"    {i}. {r['Model']:<32s}  ROC-AUC = {r['Test ROC-AUC']:.4f}")

    ranked.head(5).to_csv(os.path.join(RES_DIR, "top5_models.csv"))
    ranked.to_csv(os.path.join(RES_DIR, "model_comparison.csv"))
    print(f"\n  Saved → {RES_DIR}/model_comparison.csv")
    print(f"  Saved → {RES_DIR}/top5_models.csv")
    return ranked


def analysis_2_roc_curves(ranked, fitted, Xte_p, yte):
    print(f"\n{'═'*72}\n  ANALYSIS 2 : ROC CURVES (top 5)\n{'═'*72}")

    # Only include models that trained successfully
    top5   = [m for m in ranked.head(5)["Model"].tolist() if m in fitted]
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, c, ls in zip(top5, colors, styles):
        ypr = fitted[name].predict_proba(Xte_p)[:, 1]
        fpr, tpr, _ = roc_curve(yte, ypr)
        auc = roc_auc_score(yte, ypr)
        ax.plot(fpr, tpr, color=c, lw=2.2, linestyle=ls,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="Random (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic — Top 5 Models", fontweight="bold")
    ax.legend(loc="lower right", framealpha=.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=.25)
    fig.tight_layout()

    p = os.path.join(FIG_DIR, "roc_curves_top5.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved → {p}")


def analysis_3_confusion_matrix(ranked, fitted, Xte_p, yte):
    print(f"\n{'═'*72}\n  ANALYSIS 3 : CONFUSION MATRIX (best model)\n{'═'*72}")

    best_name = ranked.iloc[0]["Model"]
    if best_name not in fitted:
        print(f"  ⚠  Best model '{best_name}' not in fitted dict; skipping.")
        return best_name, None

    clf = fitted[best_name]
    yp  = clf.predict(Xte_p)

    test_class_dist = pd.Series(yte).value_counts()
    if len(test_class_dist) < 2:
        print(f"  [WARNING] Test set has only 1 class. Metrics will be misleading.")
    elif test_class_dist.min() < 5:
        print(f"  [WARNING] Test set is heavily imbalanced "
              f"({test_class_dist.values[0]} vs {test_class_dist.values[1]} samples). "
              "Some metrics may be unreliable.")

    cm = confusion_matrix(yte, yp)

    # FIX-G: Guard against single-class predictions producing a 1×1 matrix
    if cm.shape != (2, 2):
        print(f"  [WARNING] Confusion matrix is {cm.shape}, not 2×2. "
              "Model may be predicting only one class. Skipping detailed metrics.")
        return best_name, cm

    TN, FP, FN, TP = cm.ravel()
    total       = TP + TN + FP + FN
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv         = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    accuracy    = (TP + TN) / total

    print(f"\n  Best Model : {best_name}")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Accuracy       : {accuracy:.4f}              │")
    print(f"  │  Sensitivity    : {sensitivity:.4f}  (Recall)     │")
    print(f"  │  Specificity    : {specificity:.4f}              │")
    print(f"  │  Precision      : {precision:.4f}  (PPV)        │")
    print(f"  │  NPV            : {npv:.4f}              │")
    print(f"  │  F1-score       : {f1:.4f}              │")
    print(f"  │  TP={TP:5d}  FP={FP:5d}  FN={FN:5d}  TN={TN:5d}  │")
    print(f"  └─────────────────────────────────────────┘")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No NAFLD", "NAFLD"],
                yticklabels=["No NAFLD", "NAFLD"],
                linewidths=.8, linecolor="white", ax=ax,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")

    stats_text = (f"Sensitivity = {sensitivity:.4f}\n"
                  f"Specificity = {specificity:.4f}\n"
                  f"Precision   = {precision:.4f}\n"
                  f"F1-score    = {f1:.4f}")
    ax.text(2.6, 1.0, stats_text, fontsize=10, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="gray"))
    fig.tight_layout()

    p = os.path.join(FIG_DIR, "confusion_matrix_best.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved → {p}")

    rpt = classification_report(yte, yp, target_names=["No NAFLD", "NAFLD"])
    fp  = os.path.join(RES_DIR, "classification_report.txt")
    with open(fp, "w") as f:
        f.write(f"Classification Report — {best_name}\n{'='*50}\n{rpt}\n\n")
        f.write(f"Confusion Matrix Counts : TP={TP}, FP={FP}, FN={FN}, TN={TN}, Total={total}\n")
        f.write(f"Accuracy Check          : (TP+TN)/Total = ({TP}+{TN})/{total} = {accuracy:.4f}\n")
        f.write(f"Sensitivity (Recall) : {sensitivity:.4f}\n")
        f.write(f"Specificity          : {specificity:.4f}\n")
        f.write(f"Precision (PPV)      : {precision:.4f}\n")
        f.write(f"NPV                  : {npv:.4f}\n")
        f.write(f"F1-score             : {f1:.4f}\n")
        f.write(f"Accuracy             : {accuracy:.4f}\n")
    print(f"  Saved → {fp}")
    return best_name, cm


def analysis_4_feature_importance(fitted, feat_names):
    print(f"\n{'═'*72}\n  ANALYSIS 4 : FEATURE IMPORTANCE\n{'═'*72}")

    tree_model_names = ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM"]
    palettes         = ["Blues_d", "Oranges_d", "Greens_d", "Purples_d"]
    importance_dict  = {}

    for name, pal in zip(tree_model_names, palettes):
        mdl      = fitted.get(name)
        base_mdl = _extract_classifier(mdl)
        if base_mdl is None or not hasattr(base_mdl, "feature_importances_"):
            continue

        imp = base_mdl.feature_importances_
        if len(imp) != len(feat_names):
            print(f"  [WARNING] {name}: importance length {len(imp)} != "
                  f"feat_names length {len(feat_names)}. Skipping.")
            continue

        idx       = np.argsort(imp)[::-1][:10]
        top_names = [feat_names[j] for j in idx]
        top_vals  = imp[idx]
        importance_dict[name] = dict(zip(top_names, top_vals))

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=top_vals, y=top_names, palette=pal, ax=ax)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 10 Features — {name}", fontweight="bold")
        fig.tight_layout()

        fp = os.path.join(FIG_DIR, f"feature_importance_{name.lower().replace(' ', '_')}.png")
        fig.savefig(fp)
        plt.close(fig)
        print(f"  {name:<22s} → {fp}")

    if len(importance_dict) >= 2:
        all_feats = sorted({f for d in importance_dict.values() for f in d})
        heat = pd.DataFrame({m: {f: d.get(f, 0) for f in all_feats}
                             for m, d in importance_dict.items()})
        heat = heat.div(heat.max(axis=0), axis=1).fillna(0)
        heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(heat) * 0.35)))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5, ax=ax)
        ax.set_title("Cross-Model Feature Importance Comparison", fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        fp = os.path.join(FIG_DIR, "feature_importance_comparison.png")
        fig.savefig(fp)
        plt.close(fig)
        print(f"  Cross-model heatmap   → {fp}")

    return importance_dict


def analysis_5_shap(fitted, Xte_p, feat_names):
    print(f"\n{'═'*72}\n  ANALYSIS 5 : SHAP EXPLAINABILITY\n{'═'*72}")

    shap_candidates = ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting", "CatBoost"]
    mdl_name, mdl = None, None
    for c in shap_candidates:
        if c in fitted:
            mdl_name, mdl = c, fitted[c]
            break

    if mdl is None:
        print("  ⚠  No tree-based model available for SHAP.")
        return

    print(f"  Using model : {mdl_name}")
    base_mdl = _extract_classifier(mdl)

    n_sample   = min(500, Xte_p.shape[0])
    Xsample    = Xte_p[:n_sample]
    Xsample_df = pd.DataFrame(Xsample, columns=feat_names[:Xsample.shape[1]])

    try:
        explainer = shap.TreeExplainer(base_mdl)
        shap_vals = explainer.shap_values(Xsample)

        # FIX-H: Robust handling of all SHAP output formats
        if isinstance(shap_vals, list):
            # Binary: list of 2 arrays; multiclass: list of N arrays
            # Guard against degenerate single-element list
            if len(shap_vals) >= 2:
                shap_vals = shap_vals[1]  # Positive class
            else:
                shap_vals = shap_vals[0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # (samples, features, classes) → positive class

    except Exception as e:
        print(f"  ⚠  SHAP failed: {e}")
        return

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, Xsample_df, plot_type="dot", max_display=15, show=False)
    plt.title(f"SHAP Summary — {mdl_name}", fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "shap_summary_plot.png")
    plt.savefig(p1)
    plt.close("all")
    print(f"  Summary plot → {p1}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, Xsample_df, plot_type="bar", max_display=15, show=False)
    plt.title(f"SHAP Feature Importance — {mdl_name}", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "shap_bar_plot.png")
    plt.savefig(p2)
    plt.close("all")
    print(f"  Bar plot     → {p2}")


def analysis_6_interpretable_model(fitted, ranked, Xtr_s, ytr_s, Xte_p, yte, feat_names):
    print(f"\n{'═'*72}\n  ANALYSIS 6 : INTERPRETABLE SIMPLE MODEL (top 5 features)\n{'═'*72}")

    imp = None
    for cand in ["Random Forest", "XGBoost", "Gradient Boosting", "LightGBM"]:
        base_mdl = _extract_classifier(fitted.get(cand))
        if base_mdl is not None and hasattr(base_mdl, "feature_importances_"):
            if len(base_mdl.feature_importances_) == len(feat_names):
                imp = base_mdl.feature_importances_
                break

    if imp is None:
        print("  ⚠  No tree model with matching feature_importances_ found.")
        return

    top5_idx   = np.argsort(imp)[::-1][:5]
    top5_names = [feat_names[j] for j in top5_idx]
    print(f"  Top 5 features : {top5_names}")

    Xtr5 = Xtr_s[:, top5_idx]
    Xte5 = Xte_p[:, top5_idx]

    lr = LogisticRegression(max_iter=5000, random_state=SEED, class_weight="balanced")
    lr.fit(Xtr5, ytr_s)
    ypr_lr = lr.predict_proba(Xte5)[:, 1]
    auc_lr = roc_auc_score(yte, ypr_lr)
    acc_lr = accuracy_score(yte, lr.predict(Xte5))

    best_name = ranked.iloc[0]["Model"]
    best_auc  = ranked.iloc[0]["Test ROC-AUC"]
    best_acc  = ranked.iloc[0]["Test Accuracy"]
    auc_advantage = best_auc - auc_lr

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  {'Model':<38s} {'Accuracy':>8s}  {'ROC-AUC':>8s}  │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  {'Logistic Regression (5 features)':<38s} {acc_lr:>8.4f}  {auc_lr:>8.4f}  │")
    print(f"  │  {best_name:<38s} {best_acc:>8.4f}  {best_auc:>8.4f}  │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  AUC Advantage of Best Model over Simple LR  :  +{auc_advantage:.4f}  │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    print(f"\n  Logistic Regression Coefficients (top 5 features):")
    for fname, coef in sorted(zip(top5_names, lr.coef_[0]),
                               key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"    {fname:<30s}  {coef:>+.4f}  ({direction})")

    comp = pd.DataFrame([
        {"Model": "Logistic Regression (5 features)", "Accuracy": acc_lr,
         "ROC-AUC": auc_lr, "Features": ", ".join(top5_names),
         "AUC Advantage of Best Model": auc_advantage},
        {"Model": best_name, "Accuracy": best_acc, "ROC-AUC": best_auc,
         "Features": "all", "AUC Advantage of Best Model": 0.0},
    ])
    fp = os.path.join(RES_DIR, "interpretable_model_comparison.csv")
    comp.to_csv(fp, index=False)
    print(f"\n  Saved → {fp}")


def analysis_7_external_validation(fitted, ranked, preprocessor, num_cols, cat_cols):
    print(f"\n{'═'*72}\n  ANALYSIS 7 : EXTERNAL VALIDATION\n{'═'*72}")

    ext_path = os.path.join("data", "external_validation.csv")
    if not os.path.exists(ext_path):
        print(f"  ℹ  External dataset not found at: {ext_path}")
        print(f"     Place a CSV with columns matching the training data and a '{TARGET}' column.")
        return

    ext_df = pd.read_csv(ext_path)
    if TARGET not in ext_df.columns:
        print(f"  ⚠  '{TARGET}' column missing in external data.")
        return

    Xe = ext_df.drop(columns=[TARGET])
    ye = ext_df[TARGET].astype(int)

    try:
        Xe_p = preprocessor.transform(Xe)
        print("  ✓ Used original training preprocessor")
    except Exception as e:
        print(f"  ⚠  Preprocessor mismatch ({e}); skipping external validation.")
        return

    best_name = ranked.iloc[0]["Model"]
    if best_name not in fitted:
        print(f"  ⚠  Best model '{best_name}' not in fitted dict; skipping.")
        return

    clf = fitted[best_name]
    try:
        yp  = clf.predict(Xe_p)
        ypr = clf.predict_proba(Xe_p)[:, 1]
        ext_metrics = {
            "Accuracy":  accuracy_score(ye, yp),
            "Precision": precision_score(ye, yp, zero_division=0),
            "Recall":    recall_score(ye, yp, zero_division=0),
            "F1-score":  f1_score(ye, yp, zero_division=0),
            "ROC-AUC":   roc_auc_score(ye, ypr),
        }
        print(f"\n  External Validation Results ({best_name}):")
        for k, v in ext_metrics.items():
            print(f"    {k:<12s} : {v:.4f}")
        pd.DataFrame([ext_metrics]).to_csv(
            os.path.join(RES_DIR, "external_validation.csv"), index=False)
    except Exception as e:
        print(f"  ⚠  Prediction failed: {e}")


def analysis_8_mcnemar(ranked, fitted, Xte_p, yte):
    """
    FIX-I: Guard against failed models missing from fitted{}.
    FIX-J: Explicitly document Edwards' continuity correction in output.
    """
    print(f"\n{'═'*72}\n  ANALYSIS 8 : STATISTICAL COMPARISON (McNemar's Test)\n{'═'*72}")

    # FIX-I: Find top-2 models that actually trained successfully
    available = ranked[ranked["Model"].isin(fitted.keys())]
    if len(available) < 2:
        print("  ⚠  Fewer than 2 successfully trained models; skipping McNemar's test.")
        return

    m1_name = available.iloc[0]["Model"]
    m2_name = available.iloc[1]["Model"]

    yp1 = fitted[m1_name].predict(Xte_p)
    yp2 = fitted[m2_name].predict(Xte_p)

    tb = mcnemar_table(y_target=np.array(yte),
                       y_model1=np.array(yp1),
                       y_model2=np.array(yp2))
    b, c = tb[0, 1], tb[1, 0]

    if (b + c) == 0:
        chi2, pval = 0.0, 1.0
    else:
        # FIX-J: Edwards' continuity-corrected McNemar statistic (documented)
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        pval = 1 - chi2_dist.cdf(chi2, df=1)

    alpha = 0.05
    sig   = "YES ✓" if pval < alpha else "NO"

    print(f"\n  Model 1     : {m1_name}")
    print(f"  Model 2     : {m2_name}")
    print(f"  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  Test       : McNemar's test with Edwards' continuity        │")
    print(f"  │               correction  χ² = (|b-c|-1)²/(b+c)            │")
    print(f"  │  χ² stat    : {chi2:.4f}                                     │")
    print(f"  │  p-value    : {pval:.6f}                                   │")
    print(f"  │  Significant (α=0.05): {sig:<36s}│")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    result = pd.DataFrame([{
        "Model_1":        m1_name,
        "Model_2":        m2_name,
        "Chi2_corrected": chi2,
        "p_value":        pval,
        "Significant":    pval < alpha,
        "b":              b,
        "c":              c,
        "note":           "Edwards continuity-corrected McNemar statistic",
    }])
    fp = os.path.join(RES_DIR, "mcnemar_test.csv")
    result.to_csv(fp, index=False)
    print(f"  Saved → {fp}")


def analysis_9_save_all(ranked, fitted, Xtr_s, ytr_s):
    """
    FIX-A: Saves best model AFTER refitting on Xtr_s/ytr_s (SMOTE-balanced),
    matching the training regime used during the competition phase.
    Previously this incorrectly refitted on the unbalanced Xtr_p/ytr.
    """
    print(f"\n{'═'*72}\n  ANALYSIS 9 : SAVE ALL RESULTS\n{'═'*72}")

    best_name = ranked.iloc[0]["Model"]
    if best_name not in fitted:
        print(f"  ⚠  Best model '{best_name}' missing from fitted dict; cannot save.")
    else:
        mdl      = fitted[best_name]
        # FIX-A: Refit on SMOTE-balanced data, not raw Xtr_p
        mdl.fit(Xtr_s, ytr_s)
        mdl_path = os.path.join(MDL_DIR, "best_nafld_model.pkl")
        joblib.dump(mdl, mdl_path)
        print(f"  Model saved → {mdl_path}  (trained on SMOTE-balanced data)")

    print(f"\n  ── ALL GENERATED FILES ──")
    skip = {"venv", ".venv", "__pycache__", ".git", "NLP"}
    for root, _, files in sorted(os.walk(".")):
        if any(s in root for s in skip):
            continue
        for fn in sorted(files):
            if fn.endswith((".png", ".csv", ".txt", ".pkl")):
                fp = os.path.join(root, fn)
                sz = os.path.getsize(fp)
                print(f"    {fp:<55s}  {sz/1024:>7.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  NAFLD Research Analysis — Extended Evaluation Suite  v2.3       ║")
    print("║  Publication-Quality for IEEE / Springer                         ║")
    print("║  FIX-A: Saved model trained on SMOTE-balanced data               ║")
    print("║  FIX-B: Tuned models trained on SMOTE-balanced data              ║")
    print("║  FIX-C: Recursive classifier unwrapping                          ║")
    print("║  FIX-D: Binary numeric columns stay numeric                      ║")
    print("║  FIX-E: VotingClassifier rebuilt from post-tuned top-3           ║")
    print("║  FIX-F: GridSearchCV uses N_FOLDS consistently                   ║")
    print("║  FIX-G: cm.ravel() crash guard for single-class predictions      ║")
    print("║  FIX-H: SHAP single-element list guard                           ║")
    print("║  FIX-I: McNemar skips models missing from fitted{}               ║")
    print("║  FIX-J: Edwards' correction documented in output & CSV           ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading data ...")
    X, y = load_data()

    if len(X) < 100:
        raise ValueError(f"Dataset too small ({len(X)} rows). Need ≥ 100.")
    min_class = y.value_counts().min()
    if min_class < N_FOLDS:
        raise ValueError(
            f"Minority class too small for {N_FOLDS}-fold CV (found {min_class} samples).")

    # ── Train / Test Split ─────────────────────────────────────────────
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    print(f"\n  Train/Test Split (stratified):")
    print(f"    Train size: {len(ytr)} samples")
    print(f"    Test size:  {len(yte)} samples")
    print(f"    Train balance: {dict(ytr.value_counts())}")
    print(f"    Test balance:  {dict(yte.value_counts())}")

    # ── Preprocessing ──────────────────────────────────────────────────
    num_cols, cat_cols = detect_column_types(Xtr)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(Xtr)
    Xtr_p = preprocessor.transform(Xtr)
    Xte_p = preprocessor.transform(Xte)

    cv_test = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    _validate_cv_fold_distribution(Xtr_p, ytr, cv_test, smote_k=None)

    # ── SMOTE setup ────────────────────────────────────────────────────
    min_class_tr = int(ytr.value_counts().min())
    if min_class_tr < 3:
        smote_k = None
        print(f"[INFO] Minority class too small for SMOTE. Running without SMOTE.")
    else:
        smote_k = max(2, min(5, min_class_tr - 1))
        print(f"[INFO] SMOTE enabled with k_neighbors={smote_k}")

    _validate_cv_fold_distribution(Xtr_p, ytr, cv_test, smote_k=smote_k)

    feat_names = get_feature_names(preprocessor, num_cols, cat_cols)
    if len(feat_names) != Xtr_p.shape[1]:
        raise ValueError(
            f"Feature name mismatch: {len(feat_names)} names vs {Xtr_p.shape[1]} columns.")
    print(f"  Xtr={Xtr_p.shape}  Xte={Xte_p.shape}  features={len(feat_names)}\n")

    # ── Train all 24 models ────────────────────────────────────────────
    print("\nTraining 24 models ...")
    ranked, fitted, Xtr_s, ytr_s, base_mdls = train_all_models(
        Xtr_p, ytr, Xte_p, yte, smote_k)

    # ── Hyperparameter tuning of top 3 ────────────────────────────────
    # FIX-B: Pass Xtr_s/ytr_s so tuned models use SMOTE-balanced data
    ranked, fitted = tune_top_models(
        ranked, fitted, base_mdls,
        Xtr_p, ytr, Xtr_s, ytr_s,
        Xte_p, yte,
        smote_k=smote_k,
    )

    # ── 9 Research Analyses ────────────────────────────────────────────
    analysis_1_rank_models(ranked)
    analysis_2_roc_curves(ranked, fitted, Xte_p, yte)
    analysis_3_confusion_matrix(ranked, fitted, Xte_p, yte)
    analysis_4_feature_importance(fitted, feat_names)
    analysis_5_shap(fitted, Xte_p, feat_names)
    # FIX-B reflected here: pass Xtr_s/ytr_s to analysis_6 too
    analysis_6_interpretable_model(fitted, ranked, Xtr_s, ytr_s, Xte_p, yte, feat_names)
    analysis_7_external_validation(fitted, ranked, preprocessor, num_cols, cat_cols)
    analysis_8_mcnemar(ranked, fitted, Xte_p, yte)
    # FIX-A reflected here: pass Xtr_s/ytr_s
    analysis_9_save_all(ranked, fitted, Xtr_s, ytr_s)

    # ── Final summary ──────────────────────────────────────────────────
    best = ranked.iloc[0]
    print(f"\n{'═'*72}")
    print(f"  RESEARCH ANALYSIS COMPLETE  (v2.3)")
    print(f"{'═'*72}")
    print(f"  Best Model  : {best['Model']}")
    print(f"  ROC-AUC     : {best['Test ROC-AUC']:.4f}")
    print(f"  Analyses    : 9/9 executed")
    print(f"  Tuning      : Top 3 models optimised; VotingClassifier rebuilt")
    print(f"  Figures     : {FIG_DIR}/")
    print(f"  Results     : {RES_DIR}/")
    print(f"  Model       : {MDL_DIR}/best_nafld_model.pkl")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()