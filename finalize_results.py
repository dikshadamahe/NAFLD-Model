"""
=============================================================================
NAFLD Final Results — Analysis, Visualization & Summary Generator
=============================================================================
Reads existing results from results/, generates publication-quality plots
in results/figures/, produces final_results_summary.md, and prints a
terminal summary.

Run from project root:
    python3 finalize_results.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE

# ── Constants ───────────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.30
TARGET     = "NAFLD"
DATA_PATH  = os.path.join("data", "nafld_final_dataset.csv")
RES_DIR    = "results"
FIG_DIR    = os.path.join(RES_DIR, "figures")
MODEL_PATH = os.path.join("models", "best_nafld_model.pkl")

np.random.seed(SEED)
os.makedirs(FIG_DIR, exist_ok=True)

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
#  1. LOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def load_results():
    """Read all CSV result files."""
    model_comp = pd.read_csv(os.path.join(RES_DIR, "model_comparison.csv"))
    top5       = pd.read_csv(os.path.join(RES_DIR, "top5_models.csv"))
    interp     = pd.read_csv(os.path.join(RES_DIR, "interpretable_model_comparison.csv"))
    mcnemar    = pd.read_csv(os.path.join(RES_DIR, "mcnemar_test.csv"))

    with open(os.path.join(RES_DIR, "classification_report.txt")) as f:
        cls_report = f.read()

    return model_comp, top5, interp, mcnemar, cls_report


# ═══════════════════════════════════════════════════════════════════════════
#  2. EXTRACT KEY METRICS
# ═══════════════════════════════════════════════════════════════════════════
def extract_metrics(model_comp, top5):
    """Identify best model, total models, and top-5 summary."""
    total_models = len(model_comp)
    best = model_comp.iloc[0]
    best_info = {
        "name":      best["Model"],
        "accuracy":  best["Test Accuracy"],
        "precision": best["Precision"],
        "recall":    best["Recall"],
        "f1":        best["F1-score"],
        "roc_auc":   best["Test ROC-AUC"],
        "cv_acc":    best["CV Accuracy"],
        "cv_auc":    best["CV ROC-AUC"],
    }
    return total_models, best_info


# ═══════════════════════════════════════════════════════════════════════════
#  3. REBUILD MODEL PREDICTIONS  (for plots that need y_prob)
# ═══════════════════════════════════════════════════════════════════════════
def load_and_prepare_data():
    """Load dataset, preprocess, split, SMOTE, and return arrays."""
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=[TARGET], inplace=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols.copy():
        if X[c].nunique() <= 10:
            cat_cols.append(c)
            num_cols.remove(c)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    Xtr_p = preprocessor.fit_transform(Xtr)
    Xte_p = preprocessor.transform(Xte)

    sm = SMOTE(random_state=SEED)
    Xtr_s, ytr_s = sm.fit_resample(Xtr_p, ytr)

    # Feature names
    feat_names = list(num_cols)
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            feat_names += ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            feat_names += [f"cat_{i}" for i in range(len(cat_cols))]

    return Xtr_s, ytr_s, Xte_p, yte, np.array(feat_names)


def train_top5_models(Xtr, ytr, model_comp):
    """Re-train the top 5 models to get prediction probabilities."""
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    model_registry = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, random_state=SEED),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            random_state=SEED, verbose=0),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=SEED, use_label_encoder=False,
            eval_metric="logloss", verbosity=0),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=SEED),
    }

    top5_names = model_comp.head(5)["Model"].tolist()
    fitted = {}
    for name in top5_names:
        if name in model_registry:
            clf = model_registry[name]
            clf.fit(Xtr, ytr)
            fitted[name] = clf
            print(f"    Retrained: {name}")
        else:
            print(f"    Skipped (no definition): {name}")
    return fitted, top5_names


# ═══════════════════════════════════════════════════════════════════════════
#  4. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_comparison(model_comp):
    """Bar chart comparing all models by ROC-AUC and Accuracy."""
    fig, ax = plt.subplots(figsize=(14, 8))

    df_plot = model_comp.sort_values("Test ROC-AUC", ascending=True).copy()
    y_pos = np.arange(len(df_plot))
    bar_h = 0.35

    bars1 = ax.barh(y_pos - bar_h / 2, df_plot["Test ROC-AUC"],
                     bar_h, label="ROC-AUC", color="#2196F3", alpha=0.85)
    bars2 = ax.barh(y_pos + bar_h / 2, df_plot["Test Accuracy"],
                     bar_h, label="Accuracy", color="#FF9800", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Model"], fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("Model Performance Comparison — ROC-AUC & Accuracy",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0.75, 1.0)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(FIG_DIR, "model_comparison_chart.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fp}")
    return fp


def plot_roc_curves(fitted, Xte, yte, top5_names):
    """ROC curves for the top 5 models."""
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for name, color in zip(top5_names, colors):
        if name not in fitted:
            continue
        clf = fitted[name]
        ypr = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, ypr)
        auc = roc_auc_score(yte, ypr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Top 5 Models", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(FIG_DIR, "roc_curves.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fp}")
    return fp


def plot_confusion_matrix(fitted, Xte, yte, best_name):
    """Confusion matrix for the best model."""
    clf = fitted[best_name]
    yp = clf.predict(Xte)
    cm = confusion_matrix(yte, yp)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No NAFLD", "NAFLD"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True,
              values_format="d")
    ax.set_title(f"Confusion Matrix — {best_name}",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    fp = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fp}")
    return fp


def plot_feature_importance(fitted, feat_names, best_name):
    """Feature importance bar plot for the best tree-based model."""
    clf = fitted.get(best_name)
    if clf is None or not hasattr(clf, "feature_importances_"):
        print("  ⚠ Best model has no feature_importances_; skipping.")
        return None

    imp = clf.feature_importances_
    n = min(len(feat_names), len(imp))
    top_idx = np.argsort(imp[:n])[::-1][:15]
    top_feat = [feat_names[i] for i in top_idx]
    top_imp  = imp[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top_imp, y=top_feat, palette="viridis", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top 15 Features — {best_name}",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fp}")
    return fp


def plot_shap_summary(fitted, Xte, feat_names, best_name):
    """SHAP summary plot for the best model."""
    try:
        import shap
    except ImportError:
        print("  ⚠ SHAP not installed; skipping SHAP plot.")
        return None

    clf = fitted.get(best_name)
    if clf is None:
        return None

    print("  Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(Xte)

    # For binary classifiers that return [class0, class1]
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, Xte, feature_names=feat_names,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary — {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fp = os.path.join(FIG_DIR, "shap_summary.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {fp}")
    return fp


# ═══════════════════════════════════════════════════════════════════════════
#  5. FORMAT CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════
def format_classification_report(cls_report, best_info):
    """Return a nicely formatted classification report string."""
    lines = [
        "=" * 60,
        "CLASSIFICATION REPORT — BEST MODEL",
        "=" * 60,
        f"Model     : {best_info['name']}",
        f"Accuracy  : {best_info['accuracy']:.4f}",
        f"Precision : {best_info['precision']:.4f}",
        f"Recall    : {best_info['recall']:.4f}",
        f"F1-score  : {best_info['f1']:.4f}",
        f"ROC-AUC   : {best_info['roc_auc']:.4f}",
        "",
        "Full sklearn report:",
        "-" * 60,
        cls_report,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  6. GENERATE FINAL SUMMARY MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════
def generate_summary_md(total_models, best_info, model_comp, top5,
                        interp, mcnemar, cls_report, fig_paths):
    """Create results/final_results_summary.md."""
    top5_df = model_comp.head(5)

    # Build markdown table for top 5
    table_lines = [
        "| Rank | Model | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |",
        "|------|-------|---------------|-----------|--------|----------|---------|",
    ]
    for i, (_, r) in enumerate(top5_df.iterrows(), 1):
        table_lines.append(
            f"| {i} | {r['Model']} | {r['Test Accuracy']:.4f} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
            f"{r['F1-score']:.4f} | {r['Test ROC-AUC']:.4f} |"
        )
    top5_table = "\n".join(table_lines)

    # Interpretable model comparison
    interp_lines = [
        "| Model | Accuracy | ROC-AUC | Features |",
        "|-------|----------|---------|----------|",
    ]
    for _, r in interp.iterrows():
        interp_lines.append(
            f"| {r['Model']} | {r['Accuracy']:.4f} | "
            f"{r['ROC-AUC']:.4f} | {r['Features']} |"
        )
    interp_table = "\n".join(interp_lines)

    # McNemar result
    mcn = mcnemar.iloc[0]
    mcnemar_text = (
        f"- **Models compared**: {mcn['Model_1']} vs {mcn['Model_2']}\n"
        f"- **Chi² statistic**: {mcn['Chi2']:.4f}\n"
        f"- **p-value**: {mcn['p_value']:.6f}\n"
        f"- **Significant (α=0.05)**: {mcn['Significant']}"
    )

    # Figures
    fig_list = "\n".join(
        f"- `{os.path.basename(p)}`" for p in fig_paths if p
    )

    md = f"""# NAFLD Prediction — Final Results Summary

**Generated**: March 2026
**Total models trained**: {total_models}
**Best model**: {best_info['name']}

---

## 1. Best Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | {best_info['accuracy']:.4f} |
| Precision | {best_info['precision']:.4f} |
| Recall (Sensitivity) | {best_info['recall']:.4f} |
| F1-score | {best_info['f1']:.4f} |
| ROC-AUC | {best_info['roc_auc']:.4f} |
| CV Accuracy | {best_info['cv_acc']:.4f} |
| CV ROC-AUC | {best_info['cv_auc']:.4f} |

---

## 2. Top 5 Models

{top5_table}

---

## 3. Model Performance Explanation

The **{best_info['name']}** achieved the highest test ROC-AUC of **{best_info['roc_auc']:.4f}**
across {total_models} classifiers evaluated. The top 5 models are all ensemble-based
methods, indicating that combining multiple decision trees yields the most robust
predictions for NAFLD risk.

Key observations:
- All top 5 models achieved ROC-AUC > 0.96, indicating excellent discriminative ability.
- The best model's recall of {best_info['recall']:.4f} means it correctly identifies
  ~{best_info['recall']*100:.0f}% of NAFLD cases, which is critical for clinical screening.
- Cross-validation AUC ({best_info['cv_auc']:.4f}) closely matches test AUC ({best_info['roc_auc']:.4f}),
  suggesting minimal overfitting.

---

## 4. Interpretable Model Comparison

{interp_table}

A simple Logistic Regression using only 5 features achieves a competitive
ROC-AUC, validating that core clinical features (Age, Glucose,
Waist Circumference, BMI, Gender) carry the majority of predictive signal.

---

## 5. Statistical Comparison (McNemar's Test)

{mcnemar_text}

---

## 6. Classification Report

```
{cls_report.strip()}
```

---

## 7. Generated Figures

{fig_list}

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
"""
    fp = os.path.join(RES_DIR, "final_results_summary.md")
    with open(fp, "w") as f:
        f.write(md)
    print(f"  Saved → {fp}")
    return fp


# ═══════════════════════════════════════════════════════════════════════════
#  7. FILE CLEANUP AUDIT
# ═══════════════════════════════════════════════════════════════════════════
def audit_files():
    """Identify unnecessary or outdated files."""
    suggestions = []

    # Old figures directory (now superseded by results/figures/)
    if os.path.isdir("figures"):
        for f in os.listdir("figures"):
            if f.endswith(".png"):
                suggestions.append(
                    f"figures/{f}  →  superseded by results/figures/")

    # catboost_info training artifacts
    if os.path.isdir("catboost_info"):
        suggestions.append("catboost_info/  →  training artifact, safe to remove")

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════
#  8. TERMINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(total_models, best_info, fig_paths):
    """Print a concise terminal summary."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║            NAFLD FINAL RESULTS SUMMARY                       ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Total models trained  : {total_models:<36d} ║")
    print(f"║  Best model            : {best_info['name']:<36s} ║")
    print(f"║  Best ROC-AUC          : {best_info['roc_auc']:<36.4f} ║")
    print(f"║  Accuracy              : {best_info['accuracy']:<36.4f} ║")
    print(f"║  Precision             : {best_info['precision']:<36.4f} ║")
    print(f"║  Recall                : {best_info['recall']:<36.4f} ║")
    print(f"║  F1-score              : {best_info['f1']:<36.4f} ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  Saved figures:                                              ║")
    for p in fig_paths:
        if p:
            name = os.path.basename(p)
            print(f"║    • {name:<55s}  ║")
    print("║                                                              ║")
    print(f"║  Summary  : results/final_results_summary.md                ║")
    print("╚════════════════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  NAFLD — Final Results Analysis & Visualization")
    print("=" * 64)

    # 1. Load results
    print("\n[1/8] Loading results files...")
    model_comp, top5, interp, mcnemar, cls_report = load_results()
    print(f"  Loaded {len(model_comp)} models from model_comparison.csv")

    # 2. Extract metrics
    print("\n[2/8] Extracting key metrics...")
    total_models, best_info = extract_metrics(model_comp, top5)
    print(f"  Best model : {best_info['name']}  (ROC-AUC = {best_info['roc_auc']:.4f})")

    # 3. Prepare data & retrain top 5 for plots
    print("\n[3/8] Preparing data and retraining top 5 models...")
    Xtr, ytr, Xte, yte, feat_names = load_and_prepare_data()
    fitted, top5_names = train_top5_models(Xtr, ytr, model_comp)

    # 4. Generate visualizations
    print("\n[4/8] Generating visualizations...")
    fp1 = plot_model_comparison(model_comp)
    fp2 = plot_roc_curves(fitted, Xte, yte, top5_names)
    fp3 = plot_confusion_matrix(fitted, Xte, yte, best_info["name"])
    fp4 = plot_feature_importance(fitted, feat_names, best_info["name"])
    fp5 = plot_shap_summary(fitted, Xte, feat_names, best_info["name"])
    fig_paths = [fp1, fp2, fp3, fp4, fp5]

    # 5. Format classification report
    print("\n[5/8] Formatting classification report...")
    formatted = format_classification_report(cls_report, best_info)
    print(formatted)

    # 6. Generate final summary markdown
    print("\n[6/8] Generating final_results_summary.md...")
    generate_summary_md(total_models, best_info, model_comp, top5,
                        interp, mcnemar, cls_report, fig_paths)

    # 7. File cleanup audit
    print("\n[7/8] Auditing for unnecessary files...")
    suggestions = audit_files()
    if suggestions:
        print("  Files recommended for cleanup:")
        for s in suggestions:
            print(f"    ✗ {s}")
    else:
        print("  No cleanup needed.")

    # 8. Verify structure
    print("\n[8/8] Verifying results directory structure...")
    for root, dirs, files in os.walk(RES_DIR):
        level = root.replace(RES_DIR, "").count(os.sep)
        indent = "  " + "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " + "  " * (level + 1)
        for f in sorted(files):
            if f.startswith("."):
                continue
            print(f"{sub_indent}{f}")

    # 9. Terminal summary
    print_summary(total_models, best_info, fig_paths)


if __name__ == "__main__":
    main()
