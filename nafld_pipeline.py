"""
=============================================================================
Machine Learning-Based Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD)
Using Clinical and Lifestyle Data

Complete research-grade ML pipeline — 24 classifiers.
Publication-quality code for IEEE / Springer submission.

Dataset : NHANES DEMO_J.xpt
Target  : disease (binary 0/1) — derived if absent in raw file.
=============================================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os, sys, joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# classifiers
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron,
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Constants ───────────────────────────────────────────────────────────────
SEED          = 42
TEST_SIZE     = 0.30
N_FOLDS       = 5
TARGET        = "disease"
DATA_PATH     = os.path.join("data", "DEMO_J.xpt")
MODEL_PATH    = os.path.join("models", "best_nafld_model.pkl")
FIG_DIR       = "figures"
RES_DIR       = "results"

# Columns to drop (IDs, survey weights, scheduling, language/proxy flags)
DROP_COLS = [
    "SEQN", "SDDSRVYR", "RIDSTATR",
    "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA",
    "RIDAGEMN", "RIDEXAGM", "RIDEXMON",
    "SIALANG", "SIAPROXY", "SIAINTRP",
    "FIALANG", "FIAPROXY", "FIAINTRP",
    "MIALANG", "MIAPROXY", "MIAINTRP",
    "AIALANGA",
]

np.random.seed(SEED)
for d in [FIG_DIR, RES_DIR, "models"]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    """Load DEMO_J.xpt, drop survey-design cols, prepare target."""
    print("=" * 72)
    print("STEP 1 : DATA LOADING")
    print("=" * 72)

    df = pd.read_sas(DATA_PATH, format="xport", encoding="utf-8")
    print(f"  Raw shape   : {df.shape}")
    print(f"  Columns     : {list(df.columns)}\n")
    print(df.head())

    # drop survey / ID cols
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"\n  Dropped {len(to_drop)} survey-design / ID columns.")
    print(f"  Remaining   : {df.shape[1]} columns\n")

    # ── Target handling ────────────────────────────────────────────────
    if TARGET not in df.columns:
        print(f"  ⚠  '{TARGET}' column NOT in dataset — deriving proxy target.")
        print("     (Replace with clinical NAFLD labels for publication.)\n")
        np.random.seed(SEED)
        score = np.zeros(len(df))
        if "RIDAGEYR" in df.columns:
            score += (df["RIDAGEYR"].fillna(0) >= 45).astype(float) * 0.35
        if "RIAGENDR" in df.columns:
            score += (df["RIAGENDR"].fillna(0) == 1).astype(float) * 0.15
        if "INDFMPIR" in df.columns:
            score += (df["INDFMPIR"].fillna(5) < 1.5).astype(float) * 0.15
        if "DMDEDUC2" in df.columns:
            score += (df["DMDEDUC2"].fillna(3) <= 2).astype(float) * 0.10
        score += np.random.uniform(0, 0.4, len(df))
        df[TARGET] = (score >= np.percentile(score, 75)).astype(int)

    df.dropna(subset=[TARGET], inplace=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print(f"  Final X     : {X.shape}")
    print(f"  Final y     : {y.shape}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def detect_column_types(X):
    """Auto-detect numerical vs categorical columns."""
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=["number"]).columns.tolist()
    for c in num.copy():
        if X[c].nunique() <= 10:
            cat.append(c); num.remove(c)
    return num, cat


def build_preprocessor(num_cols, cat_cols):
    """ColumnTransformer: median+scale for num, mode+OHE for cat."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def show_class_dist(y, label=""):
    """Print class distribution."""
    vc = y.value_counts().sort_index()
    total = len(y)
    print(f"  Class distribution ({label}):")
    for cls, cnt in vc.items():
        print(f"    {cls}: {cnt}  ({cnt/total*100:.1f}%)")
    print(f"    Imbalance ratio: {vc.min()/vc.max():.3f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN-TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════
def split_data(X, y):
    """Stratified 70 / 30 split."""
    print("=" * 72)
    print("STEP 3 : TRAIN-TEST SPLIT  (70-30, stratified)")
    print("=" * 72)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"  Train : {Xtr.shape[0]}   Test : {Xte.shape[0]}")
    show_class_dist(ytr, "Train")
    show_class_dist(yte, "Test")
    return Xtr, Xte, ytr, yte


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — 24 MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
def _wrap(clf):
    """Wrap classifiers that lack predict_proba with CalibratedClassifierCV."""
    return CalibratedClassifierCV(clf, cv=3, method="sigmoid")


def get_models():
    """Return ordered dict of 23 base models (Voting added later)."""
    return {
        "Logistic Regression":       LogisticRegression(max_iter=5000, random_state=SEED, class_weight="balanced"),
        "Ridge Classifier":          _wrap(RidgeClassifier(random_state=SEED, class_weight="balanced")),
        "Lasso Logistic Regression": LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=SEED, class_weight="balanced"),
        "Decision Tree":             DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        "Random Forest":             RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "Extra Trees":               ExtraTreesClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting":         GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "XGBoost":                   XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, verbosity=0, use_label_encoder=False),
        "LightGBM":                  LGBMClassifier(n_estimators=200, random_state=SEED, class_weight="balanced", verbose=-1, force_col_wise=True),
        "CatBoost":                  CatBoostClassifier(iterations=200, random_state=SEED, verbose=0, auto_class_weights="Balanced"),
        "SVM (Linear)":              _wrap(LinearSVC(max_iter=5000, random_state=SEED, class_weight="balanced")),
        "SVM (RBF)":                 SVC(kernel="rbf", probability=True, random_state=SEED, class_weight="balanced"),
        "KNN":                       KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gaussian Naive Bayes":      GaussianNB(),
        "AdaBoost":                  AdaBoostClassifier(n_estimators=200, random_state=SEED),
        "Bagging Classifier":        BaggingClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SGD Classifier":            _wrap(SGDClassifier(max_iter=5000, random_state=SEED, class_weight="balanced", loss="modified_huber")),
        "Perceptron":                _wrap(Perceptron(max_iter=5000, random_state=SEED, class_weight="balanced")),
        "Passive Aggressive":        _wrap(SGDClassifier(loss="hinge", penalty=None, learning_rate="constant", eta0=1.0, max_iter=5000, random_state=SEED, class_weight="balanced")),
        "QDA":                       QuadraticDiscriminantAnalysis(reg_param=0.5),
        "LDA":                       LinearDiscriminantAnalysis(),
        "MLP Classifier":            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=SEED, early_stopping=True),
        "Hist Gradient Boosting":    HistGradientBoostingClassifier(max_iter=200, random_state=SEED, class_weight="balanced"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 & 5 — CROSS-VALIDATE + TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
def train_all(Xtr, Xte, ytr, yte, preprocessor):
    """Preprocess, SMOTE, 5-fold CV, train 24 models, evaluate on test."""
    print("=" * 72)
    print("STEP 4-5 : PREPROCESSING → SMOTE → 5-FOLD CV → TRAIN 24 MODELS")
    print("=" * 72)

    # fit preprocessor on train only
    Xtr_p = preprocessor.fit_transform(Xtr)
    Xte_p = preprocessor.transform(Xte)

    # SMOTE on training set only
    sm = SMOTE(random_state=SEED)
    Xtr_s, ytr_s = sm.fit_resample(Xtr_p, ytr)
    print(f"  After SMOTE  : {Xtr_s.shape[0]} samples (balanced)")
    show_class_dist(ytr_s, "Train+SMOTE")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = get_models()
    rows, fitted = [], {}

    for i, (name, clf) in enumerate(models.items(), 1):
        tag = f"[{i:2d}/24]"
        try:
            # 5-fold CV on training data
            cvr = cross_validate(clf, Xtr_s, ytr_s, cv=cv,
                                 scoring={"acc": "accuracy", "auc": "roc_auc"},
                                 n_jobs=-1, error_score="raise")
            cv_acc = cvr["test_acc"].mean()
            cv_auc = cvr["test_auc"].mean()

            # train on full SMOTE'd train, evaluate on held-out test
            clf.fit(Xtr_s, ytr_s)
            yp   = clf.predict(Xte_p)
            ypr  = clf.predict_proba(Xte_p)[:, 1]

            row = {
                "Model":           name,
                "CV Accuracy":     cv_acc,
                "CV ROC-AUC":      cv_auc,
                "Test Accuracy":   accuracy_score(yte, yp),
                "Precision":       precision_score(yte, yp, zero_division=0),
                "Recall":          recall_score(yte, yp, zero_division=0),
                "F1-score":        f1_score(yte, yp, zero_division=0),
                "Test ROC-AUC":    roc_auc_score(yte, ypr),
            }
            rows.append(row)
            fitted[name] = clf
            print(f"  {tag} {name:<32s}  AUC={row['Test ROC-AUC']:.4f} ✓")

        except Exception as e:
            print(f"  {tag} {name:<32s}  FAILED  ({e})")
            rows.append({"Model": name, **{k: np.nan for k in
                ["CV Accuracy","CV ROC-AUC","Test Accuracy",
                 "Precision","Recall","F1-score","Test ROC-AUC"]}})

    # ── Model 24: Voting Classifier (top 3 by CV ROC-AUC) ─────────────
    print(f"\n  [24/24] Building Voting Classifier (top 3 CV models) ...")
    tmp = pd.DataFrame(rows).dropna(subset=["CV ROC-AUC"])
    top3 = tmp.nlargest(3, "CV ROC-AUC")["Model"].tolist()
    print(f"         → {top3}")

    estimators = [(n, fitted[n]) for n in top3 if n in fitted]
    vc = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    try:
        cvr = cross_validate(vc, Xtr_s, ytr_s, cv=cv,
                             scoring={"acc":"accuracy","auc":"roc_auc"},
                             n_jobs=-1)
        vc.fit(Xtr_s, ytr_s)
        yp  = vc.predict(Xte_p)
        ypr = vc.predict_proba(Xte_p)[:, 1]
        row = {
            "Model":         "Voting Classifier",
            "CV Accuracy":   cvr["test_acc"].mean(),
            "CV ROC-AUC":    cvr["test_auc"].mean(),
            "Test Accuracy": accuracy_score(yte, yp),
            "Precision":     precision_score(yte, yp, zero_division=0),
            "Recall":        recall_score(yte, yp, zero_division=0),
            "F1-score":      f1_score(yte, yp, zero_division=0),
            "Test ROC-AUC":  roc_auc_score(yte, ypr),
        }
        rows.append(row)
        fitted["Voting Classifier"] = vc
        print(f"         AUC={row['Test ROC-AUC']:.4f} ✓")
    except Exception as e:
        print(f"         FAILED ({e})")

    return pd.DataFrame(rows), fitted, Xtr_s, ytr_s, Xte_p


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — PERFORMANCE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
def compare(df_res):
    """Rank by Test ROC-AUC, print table and top-5."""
    print("\n" + "=" * 72)
    print("STEP 6 : PERFORMANCE COMPARISON")
    print("=" * 72)

    ranked = (df_res
              .sort_values("Test ROC-AUC", ascending=False)
              .reset_index(drop=True))
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n", ranked.to_string())

    print("\n  ── TOP 5 MODELS ──")
    for i, r in ranked.head(5).iterrows():
        print(f"    {i}. {r['Model']:<32s}  ROC-AUC = {r['Test ROC-AUC']:.4f}")

    ranked.to_csv(os.path.join(RES_DIR, "model_comparison.csv"))
    print(f"\n  Saved → {RES_DIR}/model_comparison.csv")
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — ROC CURVES  (top 5)
# ═══════════════════════════════════════════════════════════════════════════
def plot_roc(ranked, fitted, Xte, yte):
    """ROC curves for top 5 models on a single plot."""
    print("\n" + "=" * 72)
    print("STEP 7 : ROC CURVE PLOTTING  (top 5)")
    print("=" * 72)

    top5 = ranked.head(5)["Model"].tolist()
    colors = sns.color_palette("husl", len(top5))

    plt.figure(figsize=(10, 8))
    for name, c in zip(top5, colors):
        ypr = fitted[name].predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, ypr)
        auc = roc_auc_score(yte, ypr)
        plt.plot(fpr, tpr, color=c, lw=2,
                 label=f"{name}  (AUC = {auc:.4f})")
    plt.plot([0,1],[0,1],"k--",lw=1,label="Random Chance")
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate",  fontsize=13)
    plt.title("ROC Curves — Top 5 Models", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG_DIR, "roc_curves_top5.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — FEATURE IMPORTANCE  (4 tree-based models)
# ═══════════════════════════════════════════════════════════════════════════
def plot_importances(fitted, feat_names):
    """Bar plots of top-15 features for RF, XGB, LGBM, CatBoost."""
    print("\n" + "=" * 72)
    print("STEP 8 : FEATURE IMPORTANCE")
    print("=" * 72)

    targets = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    for name in targets:
        mdl = fitted.get(name)
        if mdl is None:
            continue
        imp = mdl.feature_importances_
        n   = min(len(feat_names), len(imp))
        idx = np.argsort(imp[:n])[::-1][:15]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=imp[idx],
                    y=[feat_names[i] for i in idx],
                    palette="viridis")
        plt.title(f"Top 15 Features — {name}",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Importance"); plt.ylabel("Feature")
        plt.tight_layout()
        fp = os.path.join(FIG_DIR,
             f"feature_importance_{name.lower().replace(' ','_')}.png")
        plt.savefig(fp, dpi=300, bbox_inches="tight"); plt.close()
        print(f"  {name:<20s} → {fp}")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRA — CONFUSION MATRICES  (top 5)
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion(ranked, fitted, Xte, yte):
    """Confusion matrices for top 5 models."""
    print("\n" + "=" * 72)
    print("EXTRA : CONFUSION MATRICES  (top 5)")
    print("=" * 72)

    top5 = ranked.head(5)["Model"].tolist()
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    for ax, name in zip(axes, top5):
        yp = fitted[name].predict(Xte)
        cm = confusion_matrix(yte, yp)
        ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=11, fontweight="bold")
    plt.suptitle("Confusion Matrices — Top 5",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "confusion_matrices_top5.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRA — MODEL COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
def plot_comparison_chart(ranked):
    """Grouped horizontal bar chart of all models."""
    print("\n" + "=" * 72)
    print("EXTRA : MODEL COMPARISON BAR CHART")
    print("=" * 72)

    df = ranked.dropna(subset=["Test ROC-AUC"]).sort_values(
         "Test ROC-AUC", ascending=True).copy()
    metrics = ["Test ROC-AUC","Test Accuracy","F1-score","Precision","Recall"]
    colors  = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(df)); h = 0.15
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.barh(y + (i-2)*h, df[m], height=h, label=m, color=c, alpha=.85)
    ax.set_yticks(y); ax.set_yticklabels(df["Model"], fontsize=10)
    ax.set_xlabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1.05); ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "model_comparison_chart.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRA — BEST MODEL CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════
def best_report(ranked, fitted, Xte, yte):
    """Print and save classification report for #1 model."""
    print("\n" + "=" * 72)
    print("EXTRA : CLASSIFICATION REPORT  (best model)")
    print("=" * 72)

    name = ranked.iloc[0]["Model"]
    yp   = fitted[name].predict(Xte)
    rpt  = classification_report(yte, yp, target_names=["No NAFLD","NAFLD"])
    print(f"\n  Model: {name}\n")
    print(rpt)

    fp = os.path.join(RES_DIR, "classification_report.txt")
    with open(fp, "w") as f:
        f.write(f"Classification Report — {name}\n{'='*50}\n{rpt}")
    print(f"  Saved → {fp}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — BEST MODEL SELECTION + SAVE
# ═══════════════════════════════════════════════════════════════════════════
def save_best(ranked, fitted, Xtr, ytr):
    """Retrain best model on full training data and serialise."""
    print("\n" + "=" * 72)
    print("STEP 9 : BEST MODEL SELECTION & SAVING")
    print("=" * 72)

    best = ranked.iloc[0]
    name, auc = best["Model"], best["Test ROC-AUC"]
    print(f"\n  🏆 Best Model   : {name}")
    print(f"  🎯 Test ROC-AUC : {auc:.4f}")

    mdl = fitted[name]
    mdl.fit(Xtr, ytr)   # retrain on full train set
    joblib.dump(mdl, MODEL_PATH)
    print(f"  💾 Saved        : {MODEL_PATH}")
    return name, auc


# ═══════════════════════════════════════════════════════════════════════════
# HELPER — extract feature names from ColumnTransformer
# ═══════════════════════════════════════════════════════════════════════════
def feature_names(preprocessor, num_cols, cat_cols):
    names = list(num_cols)
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            names += ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            names += [f"cat_{i}" for i in range(len(cat_cols))]
    return np.array(names)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  NAFLD Prediction Pipeline — 24 ML Models                        ║")
    print("║  Publication-Quality Research Code                               ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # 1  Load
    X, y = load_data()

    # 2  Preprocessing setup
    print("\n" + "=" * 72)
    print("STEP 2 : DATA PREPROCESSING")
    print("=" * 72)
    show_class_dist(y, "Full dataset")
    num_cols, cat_cols = detect_column_types(X)
    print(f"  Numerical  ({len(num_cols)}): {num_cols}")
    print(f"  Categorical({len(cat_cols)}): {cat_cols}\n")
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 3  Split
    Xtr, Xte, ytr, yte = split_data(X, y)

    # 4-5  Train all 24
    res_df, fitted, Xtr_s, ytr_s, Xte_p = train_all(
        Xtr, Xte, ytr, yte, preprocessor)

    # 6  Compare
    ranked = compare(res_df)

    # 7  ROC curves
    plot_roc(ranked, fitted, Xte_p, yte)

    # 8  Feature importance
    fn = feature_names(preprocessor, num_cols, cat_cols)
    plot_importances(fitted, fn)

    # extras — confusion matrices, classification report, bar chart
    plot_confusion(ranked, fitted, Xte_p, yte)
    best_report(ranked, fitted, Xte_p, yte)
    plot_comparison_chart(ranked)

    # 9  Save best
    bname, bauc = save_best(ranked, fitted, Xtr_s, ytr_s)

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"  Best Model  : {bname}")
    print(f"  ROC-AUC     : {bauc:.4f}")
    print(f"  Model file  : {MODEL_PATH}")
    print(f"  Figures     : {FIG_DIR}/")
    print(f"    • roc_curves_top5.png")
    print(f"    • confusion_matrices_top5.png")
    print(f"    • model_comparison_chart.png")
    print(f"    • feature_importance_*.png")
    print(f"  Results     : {RES_DIR}/")
    print(f"    • model_comparison.csv")
    print(f"    • classification_report.txt")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
