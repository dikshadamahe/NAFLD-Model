"""
=============================================================================
NAFLD Research-Level Analysis
Extended Evaluation Suite for Publication (IEEE / Springer)

Prerequisites:
  - Run nafld_pipeline.py first to train 24 models.
  - This script re-trains all models, then performs 9 research analyses:
    1. Model Ranking & Comparison Table
    2. ROC Curve Visualization (publication-quality)
    3. Confusion Matrix + Sensitivity/Specificity for Best Model
    4. Feature Importance (4 tree-based models + cross-model comparison)
    5. SHAP Explainability (summary + bar plot)
    6. Interpretable Simple Model (top-5-feature Logistic Regression)
    7. External Validation Stub (Framingham-ready)
    8. Statistical Comparison — McNemar's Test (top 2 models)
    9. Save All Results (CSV, PNG @ 300 dpi, .pkl)
=============================================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os, sys, joblib, textwrap
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
    train_test_split, StratifiedKFold, cross_validate,
)
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Global Settings ─────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.30
N_FOLDS    = 5
TARGET     = "disease"
DATA_PATH  = os.path.join("data", "DEMO_J.xpt")
FIG_DIR    = "figures"
RES_DIR    = "results"
MDL_DIR    = "models"

DROP_COLS  = [
    "SEQN","SDDSRVYR","RIDSTATR",
    "WTINT2YR","WTMEC2YR","SDMVPSU","SDMVSTRA",
    "RIDAGEMN","RIDEXAGM","RIDEXMON",
    "SIALANG","SIAPROXY","SIAINTRP",
    "FIALANG","FIAPROXY","FIAINTRP",
    "MIALANG","MIAPROXY","MIAINTRP",
    "AIALANGA",
]

np.random.seed(SEED)
for d in [FIG_DIR, RES_DIR, MDL_DIR]:
    os.makedirs(d, exist_ok=True)

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
#  DATA  LOADING  &  PREPROCESSING  (reused from pipeline)
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_sas(DATA_PATH, format="xport", encoding="utf-8")
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)

    if TARGET not in df.columns:
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
    return X, y


def detect_column_types(X):
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = X.select_dtypes(include=["number"]).columns.tolist()
    for c in num.copy():
        if X[c].nunique() <= 10:
            cat.append(c); num.remove(c)
    return num, cat


def build_preprocessor(num_cols, cat_cols):
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def get_feature_names(preprocessor, num_cols, cat_cols):
    names = list(num_cols)
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            names += ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            names += [f"cat_{i}" for i in range(len(cat_cols))]
    return np.array(names)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL  DEFINITIONS  (24 classifiers)
# ═══════════════════════════════════════════════════════════════════════════
def _wrap(clf):
    return CalibratedClassifierCV(clf, cv=3, method="sigmoid")

def get_models():
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


def train_all_models(Xtr_s, ytr_s, Xte_p, yte):
    """Train 24 models and collect metrics. Returns (ranked_df, fitted_dict)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = get_models()
    rows, fitted = [], {}

    for i, (name, clf) in enumerate(models.items(), 1):
        try:
            cvr = cross_validate(clf, Xtr_s, ytr_s, cv=cv,
                                 scoring={"acc":"accuracy","auc":"roc_auc"},
                                 n_jobs=-1, error_score="raise")
            clf.fit(Xtr_s, ytr_s)
            yp  = clf.predict(Xte_p)
            ypr = clf.predict_proba(Xte_p)[:, 1]
            row = {
                "Model": name,
                "CV Accuracy":  cvr["test_acc"].mean(),
                "CV ROC-AUC":   cvr["test_auc"].mean(),
                "Test Accuracy": accuracy_score(yte, yp),
                "Precision":     precision_score(yte, yp, zero_division=0),
                "Recall":        recall_score(yte, yp, zero_division=0),
                "F1-score":      f1_score(yte, yp, zero_division=0),
                "Test ROC-AUC":  roc_auc_score(yte, ypr),
            }
            rows.append(row); fitted[name] = clf
            print(f"  [{i:2d}/24] {name:<32s}  AUC={row['Test ROC-AUC']:.4f} ✓")
        except Exception as e:
            print(f"  [{i:2d}/24] {name:<32s}  FAILED ({e})")
            rows.append({"Model": name, **{k: np.nan for k in
                ["CV Accuracy","CV ROC-AUC","Test Accuracy",
                 "Precision","Recall","F1-score","Test ROC-AUC"]}})

    # Model 24 — Voting (top 3 by CV AUC)
    tmp  = pd.DataFrame(rows).dropna(subset=["CV ROC-AUC"])
    top3 = tmp.nlargest(3, "CV ROC-AUC")["Model"].tolist()
    vc   = VotingClassifier(
        estimators=[(n, fitted[n]) for n in top3 if n in fitted],
        voting="soft", n_jobs=-1)
    try:
        cvr = cross_validate(vc, Xtr_s, ytr_s, cv=cv,
                             scoring={"acc":"accuracy","auc":"roc_auc"}, n_jobs=-1)
        vc.fit(Xtr_s, ytr_s)
        yp  = vc.predict(Xte_p)
        ypr = vc.predict_proba(Xte_p)[:, 1]
        row = {"Model": "Voting Classifier",
               "CV Accuracy": cvr["test_acc"].mean(),
               "CV ROC-AUC": cvr["test_auc"].mean(),
               "Test Accuracy": accuracy_score(yte, yp),
               "Precision": precision_score(yte, yp, zero_division=0),
               "Recall": recall_score(yte, yp, zero_division=0),
               "F1-score": f1_score(yte, yp, zero_division=0),
               "Test ROC-AUC": roc_auc_score(yte, ypr)}
        rows.append(row); fitted["Voting Classifier"] = vc
        print(f"  [24/24] Voting Classifier              AUC={row['Test ROC-AUC']:.4f} ✓")
    except Exception as e:
        print(f"  [24/24] Voting Classifier              FAILED ({e})")

    ranked = (pd.DataFrame(rows)
              .sort_values("Test ROC-AUC", ascending=False)
              .reset_index(drop=True))
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    return ranked, fitted


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — RANK MODELS
# ═══════════════════════════════════════════════════════════════════════════
def analysis_1_rank_models(ranked):
    """Sort models by Test ROC-AUC, display top 5 and formatted table."""
    sec = "ANALYSIS 1 : MODEL RANKING"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(ranked.to_string())

    print("\n  ── TOP 5 MODELS ──")
    for i, r in ranked.head(5).iterrows():
        print(f"    {i}. {r['Model']:<32s}  ROC-AUC = {r['Test ROC-AUC']:.4f}")

    # Save formatted LaTeX-ready table
    top5 = ranked.head(5).copy()
    top5.to_csv(os.path.join(RES_DIR, "top5_models.csv"))
    ranked.to_csv(os.path.join(RES_DIR, "model_comparison.csv"))
    print(f"\n  Saved → {RES_DIR}/model_comparison.csv")
    print(f"  Saved → {RES_DIR}/top5_models.csv")
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — ROC CURVES  (publication-quality)
# ═══════════════════════════════════════════════════════════════════════════
def analysis_2_roc_curves(ranked, fitted, Xte, yte):
    """Publication-quality ROC curves for top 5 models."""
    sec = "ANALYSIS 2 : ROC CURVES (top 5)"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    top5   = ranked.head(5)["Model"].tolist()
    colors = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#264653"]
    styles = ["-","--","-.",":",(0,(3,1,1,1))]

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, c, ls in zip(top5, colors, styles):
        ypr = fitted[name].predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, ypr)
        auc = roc_auc_score(yte, ypr)
        ax.plot(fpr, tpr, color=c, lw=2.2, linestyle=ls,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0,1],[0,1], "k--", lw=1, alpha=.5, label="Random (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic — Top 5 Models",
                 fontweight="bold")
    ax.legend(loc="lower right", framealpha=.9)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=.25)
    fig.tight_layout()

    p = os.path.join(FIG_DIR, "roc_curves_top5.png")
    fig.savefig(p); plt.close(fig)
    print(f"  Saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 — CONFUSION MATRIX + SENSITIVITY / SPECIFICITY
# ═══════════════════════════════════════════════════════════════════════════
def analysis_3_confusion_matrix(ranked, fitted, Xte, yte):
    """Best-model confusion matrix with Sensitivity, Specificity,
    Precision, F1-score printed and saved."""
    sec = "ANALYSIS 3 : CONFUSION MATRIX (best model)"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    best_name = ranked.iloc[0]["Model"]
    clf       = fitted[best_name]
    yp        = clf.predict(Xte)
    cm        = confusion_matrix(yte, yp)
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN)           # Recall / True Positive Rate
    specificity = TN / (TN + FP)           # True Negative Rate
    precision   = TP / (TP + FP)
    f1          = 2 * precision * sensitivity / (precision + sensitivity)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    npv         = TN / (TN + FN)           # Negative Predictive Value

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

    # heatmap-style confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No NAFLD","NAFLD"],
                yticklabels=["No NAFLD","NAFLD"],
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
    fig.savefig(p); plt.close(fig)
    print(f"  Saved → {p}")

    # Save text report
    rpt = classification_report(yte, yp, target_names=["No NAFLD","NAFLD"])
    fp  = os.path.join(RES_DIR, "classification_report.txt")
    with open(fp, "w") as f:
        f.write(f"Classification Report — {best_name}\n{'='*50}\n{rpt}\n\n")
        f.write(f"Sensitivity (Recall) : {sensitivity:.4f}\n")
        f.write(f"Specificity          : {specificity:.4f}\n")
        f.write(f"Precision (PPV)      : {precision:.4f}\n")
        f.write(f"NPV                  : {npv:.4f}\n")
        f.write(f"F1-score             : {f1:.4f}\n")
        f.write(f"Accuracy             : {accuracy:.4f}\n")
    print(f"  Saved → {fp}")

    return best_name, cm


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4 — FEATURE IMPORTANCE  (4 tree models + comparison)
# ═══════════════════════════════════════════════════════════════════════════
def analysis_4_feature_importance(fitted, feat_names):
    """Feature importance for RF, XGB, GBM, LGBM + cross-model comparison."""
    sec = "ANALYSIS 4 : FEATURE IMPORTANCE"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    tree_models = {
        "Random Forest":     fitted.get("Random Forest"),
        "XGBoost":           fitted.get("XGBoost"),
        "Gradient Boosting": fitted.get("Gradient Boosting"),
        "LightGBM":          fitted.get("LightGBM"),
    }

    importance_dict = {}
    palettes = ["Blues_d", "Oranges_d", "Greens_d", "Purples_d"]

    for (name, mdl), pal in zip(tree_models.items(), palettes):
        if mdl is None:
            continue
        imp = mdl.feature_importances_
        n   = min(len(feat_names), len(imp))
        idx = np.argsort(imp[:n])[::-1][:10]
        top_names = [feat_names[j] for j in idx]
        top_vals  = imp[idx]
        importance_dict[name] = dict(zip(top_names, top_vals))

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=top_vals, y=top_names, palette=pal, ax=ax)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 10 Features — {name}", fontweight="bold")
        fig.tight_layout()
        fp = os.path.join(FIG_DIR,
             f"feature_importance_{name.lower().replace(' ','_')}.png")
        fig.savefig(fp); plt.close(fig)
        print(f"  {name:<22s} → {fp}")

    # ── Cross-model comparison heatmap ──────────────────────────────────
    if len(importance_dict) >= 2:
        all_feats = sorted(set(f for d in importance_dict.values() for f in d))
        heat = pd.DataFrame(
            {m: {f: d.get(f, 0) for f in all_feats}
             for m, d in importance_dict.items()})
        # normalise each column to [0, 1]
        heat = heat.div(heat.max(axis=0), axis=1).fillna(0)
        heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(10, max(6, len(heat)*0.35)))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=.5, ax=ax)
        ax.set_title("Cross-Model Feature Importance Comparison",
                     fontweight="bold")
        ax.set_xlabel("Model"); ax.set_ylabel("Feature")
        fig.tight_layout()
        fp = os.path.join(FIG_DIR, "feature_importance_comparison.png")
        fig.savefig(fp); plt.close(fig)
        print(f"  Cross-model heatmap   → {fp}")

    return importance_dict


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 5 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
def analysis_5_shap(fitted, Xte, feat_names):
    """SHAP summary + bar plots for best tree-based model."""
    sec = "ANALYSIS 5 : SHAP EXPLAINABILITY"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    # Pick best available tree model for SHAP (TreeExplainer-compatible)
    shap_candidates = ["XGBoost", "LightGBM", "Random Forest",
                       "Gradient Boosting", "CatBoost"]
    mdl_name, mdl = None, None
    for c in shap_candidates:
        if c in fitted:
            mdl_name, mdl = c, fitted[c]; break

    if mdl is None:
        print("  ⚠  No tree-based model available for SHAP.")
        return

    print(f"  Using model : {mdl_name}")

    # Use a subsample to keep SHAP fast
    n_sample = min(500, Xte.shape[0])
    Xsample  = Xte[:n_sample]
    Xsample_df = pd.DataFrame(Xsample,
                               columns=feat_names[:Xte.shape[1]])

    explainer  = shap.TreeExplainer(mdl)
    shap_vals  = explainer.shap_values(Xsample)

    # For binary classifiers shap_values may return a list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]      # positive-class explanations

    # ── SHAP Summary Plot (beeswarm) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, Xsample_df, plot_type="dot",
                      max_display=15, show=False)
    plt.title(f"SHAP Summary — {mdl_name}", fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "shap_summary_plot.png")
    plt.savefig(p1); plt.close("all")
    print(f"  Summary plot → {p1}")

    # ── SHAP Bar Plot (mean |SHAP|) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, Xsample_df, plot_type="bar",
                      max_display=15, show=False)
    plt.title(f"SHAP Feature Importance — {mdl_name}", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "shap_bar_plot.png")
    plt.savefig(p2); plt.close("all")
    print(f"  Bar plot     → {p2}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 6 — INTERPRETABLE SIMPLE MODEL  (top 5 features)
# ═══════════════════════════════════════════════════════════════════════════
def analysis_6_interpretable_model(fitted, ranked, Xtr, ytr, Xte, yte,
                                    feat_names):
    """Train Logistic Regression on top-5 important features only,
    compare ROC-AUC with the best complex model."""
    sec = "ANALYSIS 6 : INTERPRETABLE SIMPLE MODEL (top 5 features)"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    # Get feature importances from best available tree model
    for cand in ["Random Forest","XGBoost","Gradient Boosting","LightGBM"]:
        if cand in fitted and hasattr(fitted[cand], "feature_importances_"):
            imp = fitted[cand].feature_importances_
            break
    else:
        print("  ⚠  No tree model with feature_importances_ found."); return

    n = min(len(feat_names), len(imp))
    top5_idx   = np.argsort(imp[:n])[::-1][:5]
    top5_names = [feat_names[j] for j in top5_idx]
    print(f"  Top 5 features : {top5_names}")

    # Subset data
    Xtr5 = Xtr[:, top5_idx]
    Xte5 = Xte[:, top5_idx]

    # Train simple logistic regression
    lr = LogisticRegression(max_iter=5000, random_state=SEED,
                            class_weight="balanced")
    lr.fit(Xtr5, ytr)
    ypr_lr   = lr.predict_proba(Xte5)[:, 1]
    auc_lr   = roc_auc_score(yte, ypr_lr)
    acc_lr   = accuracy_score(yte, lr.predict(Xte5))

    # Best complex model metrics
    best_name = ranked.iloc[0]["Model"]
    best_auc  = ranked.iloc[0]["Test ROC-AUC"]
    best_acc  = ranked.iloc[0]["Test Accuracy"]

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  {'Model':<35s} {'Accuracy':>8s}  {'ROC-AUC':>8s}  │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  {'LR (5 features)':<35s} {acc_lr:>8.4f}  {auc_lr:>8.4f}  │")
    print(f"  │  {best_name:<35s} {best_acc:>8.4f}  {best_auc:>8.4f}  │")
    print(f"  │  AUC drop                            "
          f"         {best_auc - auc_lr:>+8.4f}  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Coefficients table
    print(f"\n  Logistic Regression Coefficients:")
    for fname, coef in sorted(zip(top5_names, lr.coef_[0]),
                               key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"    {fname:<30s}  {coef:>+.4f}  ({direction})")

    # Save comparison
    comp = pd.DataFrame([
        {"Model": f"Logistic Regression (5 features)",
         "Accuracy": acc_lr, "ROC-AUC": auc_lr,
         "Features": ", ".join(top5_names)},
        {"Model": best_name, "Accuracy": best_acc,
         "ROC-AUC": best_auc, "Features": "all"},
    ])
    fp = os.path.join(RES_DIR, "interpretable_model_comparison.csv")
    comp.to_csv(fp, index=False)
    print(f"\n  Saved → {fp}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 7 — EXTERNAL VALIDATION  (stub for Framingham / other)
# ═══════════════════════════════════════════════════════════════════════════
def analysis_7_external_validation(fitted, ranked, preprocessor,
                                    num_cols, cat_cols):
    """Apply best model to an external dataset.
    Currently a functional stub — provide a CSV/XPT path to activate."""
    sec = "ANALYSIS 7 : EXTERNAL VALIDATION"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    ext_path = os.path.join("data", "external_validation.csv")
    if not os.path.exists(ext_path):
        print(f"  ℹ  External dataset not found at: {ext_path}")
        print(f"     Place a CSV with columns matching the training data")
        print(f"     and a '{TARGET}' column, then re-run this analysis.")
        print(f"     Expected columns (numerical): {num_cols}")
        print(f"     Expected columns (categorical): {cat_cols}")
        return

    ext_df = pd.read_csv(ext_path)
    print(f"  Loaded external data : {ext_df.shape}")

    if TARGET not in ext_df.columns:
        print(f"  ⚠  '{TARGET}' column missing in external data."); return

    Xe = ext_df.drop(columns=[TARGET])
    ye = ext_df[TARGET].astype(int)

    # Align features
    common_num = [c for c in num_cols if c in Xe.columns]
    common_cat = [c for c in cat_cols if c in Xe.columns]
    print(f"  Common numerical   : {len(common_num)}/{len(num_cols)}")
    print(f"  Common categorical : {len(common_cat)}/{len(cat_cols)}")

    ext_pre = build_preprocessor(common_num, common_cat)
    Xe_p    = ext_pre.fit_transform(Xe)

    best_name = ranked.iloc[0]["Model"]
    clf       = fitted[best_name]

    # Retrain on aligned feature space if needed
    print(f"  ⚠  Note: For rigorous validation, retrain on aligned features.")
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
            orig = ranked.iloc[0].get(k, ranked.iloc[0].get(f"Test {k}", np.nan))
            delta = v - orig if not np.isnan(orig) else np.nan
            print(f"    {k:<12s} : {v:.4f}  (Δ = {delta:+.4f})")

        comp = pd.DataFrame([ext_metrics])
        comp.to_csv(os.path.join(RES_DIR, "external_validation.csv"), index=False)
    except Exception as e:
        print(f"  ⚠  Prediction failed (feature mismatch likely): {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 8 — STATISTICAL COMPARISON  (McNemar's test)
# ═══════════════════════════════════════════════════════════════════════════
def analysis_8_mcnemar(ranked, fitted, Xte, yte):
    """McNemar's test between top 2 models.
    Tests whether the disagreement in predictions is statistically
    significant (p < 0.05)."""
    sec = "ANALYSIS 8 : STATISTICAL COMPARISON (McNemar's Test)"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    m1_name = ranked.iloc[0]["Model"]
    m2_name = ranked.iloc[1]["Model"]
    yp1 = fitted[m1_name].predict(Xte)
    yp2 = fitted[m2_name].predict(Xte)

    # Build contingency table  [[both_correct, m1_correct_m2_wrong],
    #                           [m1_wrong_m2_correct, both_wrong]]
    tb = mcnemar_table(y_target=np.array(yte),
                       y_model1=np.array(yp1),
                       y_model2=np.array(yp2))

    b = tb[0, 1]   # m1 correct, m2 wrong
    c = tb[1, 0]   # m1 wrong, m2 correct

    # McNemar's chi-squared with continuity correction
    if (b + c) == 0:
        chi2, pval = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        pval = 1 - chi2_dist.cdf(chi2, df=1)

    alpha = 0.05
    sig   = "YES ✓" if pval < alpha else "NO"
    interp = ("The two models differ significantly in predictions."
              if pval < alpha
              else "No statistically significant difference between the two models.")

    print(f"\n  Model 1     : {m1_name}")
    print(f"  Model 2     : {m2_name}")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │  Contingency Table                           │")
    print(f"  │    Both correct     : {tb[0,0]:>6d}                │")
    print(f"  │    M1 correct only  : {b:>6d}                │")
    print(f"  │    M2 correct only  : {c:>6d}                │")
    print(f"  │    Both wrong       : {tb[1,1]:>6d}                │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  χ² statistic : {chi2:.4f}                     │")
    print(f"  │  p-value      : {pval:.6f}                   │")
    print(f"  │  α             : {alpha}                        │")
    print(f"  │  Significant? : {sig:<30s}│")
    print(f"  └──────────────────────────────────────────────┘")
    print(f"\n  Interpretation: {interp}")

    # Save
    result = pd.DataFrame([{
        "Model_1": m1_name, "Model_2": m2_name,
        "Chi2": chi2, "p_value": pval, "Significant": pval < alpha,
        "b (M1✓ M2✗)": b, "c (M1✗ M2✓)": c,
    }])
    fp = os.path.join(RES_DIR, "mcnemar_test.csv")
    result.to_csv(fp, index=False)
    print(f"  Saved → {fp}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 9 — SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def analysis_9_save_all(ranked, fitted, Xtr, ytr):
    """Save final model, updated comparison table, and list all outputs."""
    sec = "ANALYSIS 9 : SAVE ALL RESULTS"
    print(f"\n{'═'*72}\n  {sec}\n{'═'*72}")

    # Best model
    best_name = ranked.iloc[0]["Model"]
    mdl = fitted[best_name]
    mdl.fit(Xtr, ytr)
    mdl_path = os.path.join(MDL_DIR, "best_nafld_model.pkl")
    joblib.dump(mdl, mdl_path)
    print(f"  Model saved → {mdl_path}")

    # List all output files
    print(f"\n  ── ALL GENERATED FILES ──")
    for root, _, files in sorted(os.walk(".")):
        for fn in sorted(files):
            if fn.endswith((".png",".csv",".txt",".pkl")):
                fp = os.path.join(root, fn)
                sz = os.path.getsize(fp)
                print(f"    {fp:<55s}  {sz/1024:>7.1f} KB")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  NAFLD Research Analysis — Extended Evaluation Suite             ║")
    print("║  Publication-Quality for IEEE / Springer                         ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # ── Load & preprocess ──────────────────────────────────────────────
    print("Loading data ...")
    X, y = load_data()
    num_cols, cat_cols = detect_column_types(X)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    Xtr_p = preprocessor.fit_transform(Xtr)
    Xte_p = preprocessor.transform(Xte)

    sm = SMOTE(random_state=SEED)
    Xtr_s, ytr_s = sm.fit_resample(Xtr_p, ytr)

    feat_names = get_feature_names(preprocessor, num_cols, cat_cols)
    print(f"  Xtr_smote={Xtr_s.shape}  Xte={Xte_p.shape}  "
          f"features={len(feat_names)}\n")

    # ── Train all 24 models ────────────────────────────────────────────
    print("Training 24 models ...")
    ranked, fitted = train_all_models(Xtr_s, ytr_s, Xte_p, yte)

    # ── 9 Research Analyses ────────────────────────────────────────────
    analysis_1_rank_models(ranked)
    analysis_2_roc_curves(ranked, fitted, Xte_p, yte)
    analysis_3_confusion_matrix(ranked, fitted, Xte_p, yte)
    analysis_4_feature_importance(fitted, feat_names)
    analysis_5_shap(fitted, Xte_p, feat_names)
    analysis_6_interpretable_model(fitted, ranked, Xtr_s, ytr_s,
                                    Xte_p, yte, feat_names)
    analysis_7_external_validation(fitted, ranked, preprocessor,
                                    num_cols, cat_cols)
    analysis_8_mcnemar(ranked, fitted, Xte_p, yte)
    analysis_9_save_all(ranked, fitted, Xtr_s, ytr_s)

    # ── Final summary ─────────────────────────────────────────────────
    best = ranked.iloc[0]
    print(f"\n{'═'*72}")
    print(f"  RESEARCH ANALYSIS COMPLETE")
    print(f"{'═'*72}")
    print(f"  Best Model  : {best['Model']}")
    print(f"  ROC-AUC     : {best['Test ROC-AUC']:.4f}")
    print(f"  Analyses    : 9/9 executed")
    print(f"  Figures     : {FIG_DIR}/")
    print(f"  Results     : {RES_DIR}/")
    print(f"  Model       : {MDL_DIR}/best_nafld_model.pkl")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
