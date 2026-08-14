#!/usr/bin/env python
"""Generate standalone model-specific notebooks for the bioinformatics portfolio."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared code blocks (self-contained: imports, data, EDA, preprocessing)
# ---------------------------------------------------------------------------
SHARED_SETUP = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

# ============================================================
# Load Wisconsin Breast Cancer dataset (UCI open database)
# 30 cell-nucleus image features; target 0=malignant, 1=benign
# ============================================================
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
X = cancer.data
y = cancer.target
feature_names = list(cancer.feature_names)
print(f"Dataset shape: {df.shape}  |  Classes: {np.bincount(y).tolist()} "
      "(0=malignant, 1=benign)")
print(f"Missing values: {df.isna().sum().sum()}")

# ============================================================
# Preprocessing — stratified split + StandardScaler (fit on train only)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")'''

EVAL_TEMPLATE = '''# ============================================================
# Train {NAME} & evaluate
# ============================================================
{IMPORT}
model = {CTOR}
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cv_scores = cross_val_score(model, X_train_sc, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring="roc_auc")

print(f"Test  accuracy : {acc:.4f}  |  precision: {prec:.4f}")
print(f"Recall         : {rec:.4f}  |  F1      : {f1:.4f}")
print(f"Test  ROC-AUC  : {roc_auc:.4f}")
print(f"CV    ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
axes[0].set_title("{NAME} — Confusion matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[1].plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("{NAME} — ROC curve")
axes[1].legend(loc="lower right")
plt.tight_layout(); plt.show()'''

# ---------------------------------------------------------------------------
# Model-specific tail code
# ---------------------------------------------------------------------------
LR_TAIL = '''# ============================================================
# Interpretability — top coefficients (log-odds)
# ============================================================
coef_df = pd.DataFrame({"feature": feature_names,
                        "coef": model.coef_[0]}).sort_values("coef", key=abs,
                                                              ascending=False)
print("Top 8 features driving the model (|log-odds coefficient|):")
display(coef_df.head(8))'''

KNN_TAIL = '''from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# Hyperparameter tuning — number of neighbors k
# ============================================================
k_range = range(1, 31)
cv_knn = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_sc, y_train, cv=5, scoring="roc_auc")
    cv_knn.append(scores.mean())
best_k = k_range[np.argmax(cv_knn)]

plt.figure(figsize=(8, 4))
plt.plot(k_range, cv_knn, "o-", color="#1f77b4")
plt.axvline(best_k, color="red", ls="--", label=f"best k = {best_k}")
plt.xlabel("Number of neighbors k"); plt.ylabel("5-fold CV ROC-AUC")
plt.title("KNN hyperparameter tuning"); plt.legend()
plt.tight_layout(); plt.show()

model = KNeighborsClassifier(n_neighbors=best_k)'''

RF_TAIL = '''# ============================================================
# Built-in feature importance
# ============================================================
imp = pd.DataFrame({"feature": feature_names,
                    "importance": model.feature_importances_}
                   ).sort_values("importance", ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(data=imp.head(10), x="importance", y="feature", color="#2ca02c")
plt.title("Top 10 features by Random Forest importance")
plt.tight_layout(); plt.show()'''

KMEANS_CODE = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

# ============================================================
# Load data + scale (full dataset, unsupervised)
# ============================================================
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
print(f"Dataset shape: {X.shape}")

# ============================================================
# Choose k — elbow method + silhouette score
# ============================================================
inertias, silhouettes, K_range = [], [], range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_sc, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(K_range, inertias, "o-", color="#d62728")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow method")
axes[1].plot(K_range, silhouettes, "o-", color="#1f77b4")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette analysis")
plt.tight_layout(); plt.show()
best_k = int(K_range[np.argmax(silhouettes)])
print(f"Best k by silhouette score: {best_k}")

# ============================================================
# K-Means clustering (k = 2) + validation vs true labels
# ============================================================
km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_sc)
labels_km = km.labels_
ari = adjusted_rand_score(y, labels_km)
sil = silhouette_score(X_sc, labels_km)
print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")
print(f"Silhouette score                   : {sil:.4f}")

# ============================================================
# PCA projection for 2D visualization
# ============================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
print(f"Explained variance (2 PCs): {pca.explained_variance_ratio_.sum():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm",
                alpha=0.7, edgecolors="k", linewidths=0.3)
axes[0].set_title("True labels (PCA space)")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap="coolwarm",
                alpha=0.7, edgecolors="k", linewidths=0.3)
axes[1].set_title("K-Means clusters (PCA space)")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.tight_layout(); plt.show()'''

LINEAR_REG_CODE = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

# ============================================================
# Load Diabetes dataset (UCI open database)
# 442 patients, 10 baseline variables -> disease progression
# ============================================================
diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target
feat_d = diabetes.feature_names
print(f"Diabetes dataset shape: {X_d.shape}")

# ============================================================
# Train Linear Regression
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.20,
                                          random_state=42)
linreg = LinearRegression()
linreg.fit(X_tr, y_tr)
y_pred = linreg.predict(X_te)

r2  = r2_score(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
print(f"R²                  : {r2:.4f}")
print(f"Mean squared error  : {mse:.2f}")
print(f"Mean absolute error : {mae:.2f}")

# ============================================================
# Coefficients (standardized inputs -> standardized slopes)
# ============================================================
coef_d = pd.DataFrame({"feature": feat_d,
                       "coef": linreg.coef_}).sort_values("coef")
plt.figure(figsize=(7, 4))
sns.barplot(data=coef_d, x="coef", y="feature", color="#7f7f7f")
plt.axvline(0, color="k", lw=1)
plt.title("Linear Regression coefficients (standardized)")
plt.tight_layout(); plt.show()

# ============================================================
# Predicted vs actual
# ============================================================
plt.figure(figsize=(6, 5))
plt.scatter(y_te, y_pred, alpha=0.7, edgecolors="k", linewidths=0.3)
lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
plt.plot(lims, lims, "r--", lw=2, label="Perfect fit")
plt.xlabel("Actual progression"); plt.ylabel("Predicted progression")
plt.title(f"Linear Regression — R² = {r2:.3f}")
plt.legend(); plt.tight_layout(); plt.show()'''

COMPARE_CODE = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

# ============================================================
# Load data + preprocess (shared pipeline)
# ============================================================
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# Compare all classifiers via 5-fold cross-validation
# ============================================================
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=9),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
cv_results = {}
for name, m in models.items():
    cv_results[name] = cross_val_score(m, X_train_sc, y_train, cv=cv,
                                       scoring="roc_auc")

plt.figure(figsize=(9, 5))
sns.boxplot(data=pd.DataFrame(cv_results), palette="Set2")
sns.stripplot(data=pd.DataFrame(cv_results), color="black", alpha=0.4, jitter=True)
plt.ylabel("5-fold CV ROC-AUC")
plt.title("Cross-validated performance of classifiers")
plt.xticks(rotation=15)
plt.tight_layout(); plt.show()

# ============================================================
# Summary table on the held-out test set
# ============================================================
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
rows = []
for name, m in models.items():
    m.fit(X_train_sc, y_train)
    y_pred = m.predict(X_test_sc)
    y_prob = m.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    rows.append({"model": name,
                 "accuracy": accuracy_score(y_test, y_pred),
                 "f1": f1_score(y_test, y_pred),
                 "roc_auc": auc(fpr, tpr)})
results_df = pd.DataFrame(rows).set_index("model")
display(results_df.round(4).sort_values("f1", ascending=False))'''

# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------
KERNEL = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.13"},
}


def cell(cell_type, source):
    lines = source.splitlines()
    src = [line + "\n" for line in lines]
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "id": "c" + str(abs(hash(source)) % 10**8),
        "metadata": {},
        "outputs": [] if cell_type == "code" else [],
        "source": src,
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_eval(name, import_line, ctor):
    return EVAL_TEMPLATE.replace("{NAME}", name)\
                        .replace("{IMPORT}", import_line)\
                        .replace("{CTOR}", ctor)


def md(title):
    return cell("markdown", title)


# --- define the 7 notebooks ---
notebooks = {}

notebooks["01_Logistic_Regression.ipynb"] = [
    md("# 🧬 Logistic Regression — Breast Cancer Classification\n"
       "Self-contained pipeline: load UCI data → scale → train → evaluate."),
    cell("code", SHARED_SETUP),
    cell("code", build_eval(
        "Logistic Regression",
        "from sklearn.linear_model import LogisticRegression",
        "LogisticRegression(max_iter=2000, random_state=42)")),
    cell("code", LR_TAIL),
]

notebooks["02_KNN.ipynb"] = [
    md("# 🧬 K-Nearest Neighbors — Breast Cancer Classification\n"
       "Self-contained pipeline with k hyperparameter tuning."),
    cell("code", SHARED_SETUP),
    cell("code", KNN_TAIL),
    cell("code", build_eval(
        "KNN",
        "from sklearn.neighbors import KNeighborsClassifier",
        "KNeighborsClassifier(n_neighbors=best_k)")),
]

notebooks["03_SVM.ipynb"] = [
    md("# 🧬 Support Vector Machine (RBF) — Breast Cancer Classification\n"
       "Self-contained pipeline with RBF-kernel SVM."),
    cell("code", SHARED_SETUP),
    cell("code", build_eval(
        "SVM (RBF)",
        "from sklearn.svm import SVC",
        'SVC(kernel="rbf", probability=True, random_state=42)')),
]

notebooks["04_Random_Forest.ipynb"] = [
    md("# 🧬 Random Forest — Breast Cancer Classification\n"
       "Self-contained pipeline with built-in feature importance."),
    cell("code", SHARED_SETUP),
    cell("code", build_eval(
        "Random Forest",
        "from sklearn.ensemble import RandomForestClassifier",
        "RandomForestClassifier(n_estimators=200, random_state=42)")),
    cell("code", RF_TAIL),
]

notebooks["05_KMeans_Clustering.ipynb"] = [
    md("# 🧬 K-Means Clustering — Unsupervised Cell Grouping\n"
       "Self-contained unsupervised workflow: elbow → silhouette → clustering → PCA."),
    cell("code", KMEANS_CODE),
]

notebooks["06_Linear_Regression.ipynb"] = [
    md("# 🧬 Linear Regression — Diabetes Disease Progression\n"
       "Self-contained regression workflow on the UCI Diabetes dataset."),
    cell("code", LINEAR_REG_CODE),
]

# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------
for fname, cells in notebooks.items():
    path = os.path.join(BASE, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1, ensure_ascii=False)
    print(f"Wrote {fname}")

print("\nDone — all notebooks generated.")
