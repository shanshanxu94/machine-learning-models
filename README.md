# 🧬 Machine Learning Models for Bioinformatics

A portfolio of classical and deep machine-learning models applied to **open-source biomedical datasets** — built for bioinformatics / data-science roles.

**Author:** Shanshan Xu · [GitHub](https://github.com/shanshanxu94)

---

## 📂 Notebooks

### Classical ML (sklearn)
| # | Notebook | Model | Task / Dataset |
|---|----------|-------|----------------|
| 01 | `01_Logistic_Regression.ipynb` | Logistic Regression | Breast-cancer classification (Wisconsin, UCI) + interpretable coefficients |
| 02 | `02_KNN.ipynb` | K-Nearest Neighbors | Breast-cancer classification (with k tuning) |
| 03 | `03_SVM.ipynb` | SVM (Linear vs RBF) | Breast-cancer classification (kernel comparison) |
| 04 | `04_Random_Forest.ipynb` | Random Forest | Breast-cancer classification + feature importance |
| 05 | `05_KMeans_Clustering.ipynb` | K-Means | Unsupervised cell grouping (elbow/silhouette + PCA) |
| 06 | `06_Linear_Regression.ipynb` | Linear Regression | Diabetes disease progression (UCI) |
| 07 | `07_Model_Comparison.ipynb` | — | Cross-validated comparison of all classifiers |

### Deep Learning (PyTorch)
| # | Notebook | Model | Task / Dataset |
|---|----------|-------|----------------|
| 08 | `08_CNN.ipynb` | CNN | Microscope blood-cell classification (BloodMNIST 224×224, MedMNIST) |
| 09 | `09_RNN_LSTM.ipynb` | Bidirectional LSTM | DNA promoter sequence classification (UCI Molecular Biology) |

---

## 🗂️ Datasets (all open source)

| Dataset | Source | Description |
|---------|--------|-------------|
| **Wisconsin Breast Cancer** | UCI | 30 cell-nucleus image features; benign vs malignant |
| **Diabetes** | UCI | 10 clinical baselines → disease progression |
| **Promoter Gene Sequences** | UCI Molecular Biology | 106 E. coli DNA sequences (57 bp); promoter vs non-promoter |
| **BloodMNIST** | MedMNIST / BCCD | Real microscope images of 8 blood-cell types |

---

## ✨ Highlights

- **Reproducible**: every notebook is self-contained, `random_state=42`, scaler fit on train only (no data leakage)
- **Interpretable**: logistic coefficients, linear-SVM weights, random-forest feature importance
- **Bio-relevant**: cell-nucleus morphometry → diagnosis; clustering mirrors scRNA-seq population discovery; LSTM reads DNA base-by-base; CNN classifies real microscope cells
- **Evaluation**: accuracy / precision / recall / F1, ROC-AUC, confusion matrices, 5-fold cross-validation

---

## 🚀 Usage

```bash
# Python 3.12 environment (conda)
conda create -n py312 python=3.12 -y
conda activate py312
pip install numpy pandas scikit-learn matplotlib seaborn \
            torch torchvision medmnist nbconvert

# Open any notebook
jupyter notebook 01_Logistic_Regression.ipynb
```

All data is downloaded automatically on first run.

---

## 🗂️ Repo structure

```
machine-learning-models/
├── 01_Logistic_Regression.ipynb ... 09_RNN_LSTM.ipynb
├── _gen_notebooks.py              # regenerates notebooks 01-07
├── _gen_notebooks_dl.py           # regenerates notebook 09
├── _gen_notebooks_microscope_hr.py# regenerates notebook 08
└── .gitignore
```
