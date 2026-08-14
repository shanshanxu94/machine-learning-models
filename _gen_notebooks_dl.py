#!/usr/bin/env python
"""Generate deep-learning notebooks (CNN + RNN) for the bioinformatics portfolio."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

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


def md(title):
    return cell("markdown", title)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# 08 — CNN (MNIST image classification, PyTorch)
# ---------------------------------------------------------------------------
CNN_CODE_1 = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Dataset — MNIST (public database, 70k handwritten digits)
# Same CNN architecture transfers directly to cell / pathology images
# ============================================================
DATA_DIR = "data"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)
print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")'''

CNN_CODE_2 = '''# ============================================================
# Visualize samples
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, label = train_ds[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"label={label}")
    ax.axis("off")
plt.suptitle("MNIST samples")
plt.tight_layout(); plt.show()'''

CNN_CODE_3 = '''# ============================================================
# Define CNN architecture
# ============================================================
class CNN(nn.Module):
    """Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> FC."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")'''

CNN_CODE_4 = '''# ============================================================
# Training loop
# ============================================================
def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    total = correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total


EPOCHS = 3
history = []
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(train_loader, model, criterion,
                                      optimizer, device)
    te_acc = evaluate(test_loader, model, device)
    history.append((tr_loss, tr_acc, te_acc))
    print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} | "
          f"train acc {tr_acc:.4f} | test acc {te_acc:.4f}")'''

CNN_CODE_5 = '''# ============================================================
# Evaluation — accuracy, classification report, confusion matrix
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report

@torch.no_grad()
def predict_all(loader, model, device):
    model.eval()
    ys, preds = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


y_true, y_pred = predict_all(test_loader, model, device)
acc = (y_true == y_pred).mean()
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("MNIST CNN — confusion matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ---- misclassified examples ----
mis_idx = np.where(y_true != y_pred)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, _ = test_ds[mis_idx[i]]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"true={y_true[mis_idx[i]]} pred={y_pred[mis_idx[i]]}",
                 color="red")
    ax.axis("off")
plt.suptitle("Misclassified examples")
plt.tight_layout(); plt.show()'''

# ---------------------------------------------------------------------------
# 09 — RNN/LSTM (DNA promoter classification, PyTorch)
# ---------------------------------------------------------------------------
RNN_CODE_1 = '''# ============================================================
# Setup
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Dataset — UCI Molecular Biology (Promoter Gene Sequences)
# 106 E. coli DNA sequences (57 bp); classify promoter (+) vs non-promoter (-)
# ============================================================
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "molecular-biology/promoter-gene-sequences/promoters.data")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
text = urllib.request.urlopen(req, timeout=30).read().decode()

rows = []
for line in text.strip().splitlines():
    parts = line.split(",")
    rows.append((parts[0], parts[2].strip()))

df = pd.DataFrame(rows, columns=["label", "seq"])
df["y"] = (df["label"] == "+").astype(int)
print(f"Samples: {len(df)} | Sequence length: {len(df['seq'].iloc[0])} bp")
print(df["label"].value_counts())
display(df.head())'''

RNN_CODE_2 = '''# ============================================================
# One-hot encode DNA sequences -> (N, seq_len, 4)
# ============================================================
BASES = ["A", "C", "G", "T"]
base2idx = {b: i for i, b in enumerate(BASES)}


def one_hot_encode(seq, alphabet=BASES):
    mat = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):     # UCI sequences are lowercase
        if ch in base2idx:
            mat[i, base2idx[ch]] = 1.0
    return mat


X = np.stack([one_hot_encode(s) for s in df["seq"]])   # (106, 57, 4)
y = df["y"].to_numpy()
print("Feature tensor shape:", X.shape)

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {len(y_train)} | Test: {len(y_test)}")'''

RNN_CODE_3 = '''# ============================================================
# Define bidirectional LSTM model
# ============================================================
class LSTMPromoter(nn.Module):
    """Reads the DNA sequence position-by-position with an LSTM,
    then classifies using the last hidden state."""
    def __init__(self, input_size=4, hidden_size=32, num_layers=1,
                 num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (B, seq, 2*hidden)
        last = out[:, -1, :]           # last timestep
        return self.fc(last)


model = LSTMPromoter().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")


# ---- tensors ----
Xt = torch.tensor(X_train).to(device)
yt = torch.tensor(y_train, dtype=torch.long).to(device)
Xe = torch.tensor(X_test).to(device)
ye = torch.tensor(y_test, dtype=torch.long).to(device)'''

RNN_CODE_4 = '''# ============================================================
# Training loop
# ============================================================
EPOCHS = 200
model.train()
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    out = model(Xt)
    loss = criterion(out, yt)
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 20 == 0:
        with torch.no_grad():
            tr_acc = (out.argmax(1) == yt).float().mean().item()
            te_acc = (model(Xe).argmax(1) == ye).float().mean().item()
        print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | "
              f"train acc {tr_acc:.4f} | test acc {te_acc:.4f}")'''

RNN_CODE_5 = '''# ============================================================
# Evaluation
# ============================================================
with torch.no_grad():
    model.eval()
    y_pred = model(Xe).argmax(1).cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred,
                            target_names=["non-promoter (-)", "promoter (+)"],
                            digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["non-promoter", "promoter"],
            yticklabels=["non-promoter", "promoter"])
plt.title("LSTM promoter classification — confusion matrix")
plt.tight_layout(); plt.show()

# ---- example sequences + predictions ----
print("Example predictions (test set):")
for i in range(8):
    label = "promoter (+)" if y_pred[i] == 1 else "non-promoter (-)"
    true = "promoter (+)" if y_test[i] == 1 else "non-promoter (-)"
    mark = "OK" if y_pred[i] == y_test[i] else "MIS"
    print(f"  [{mark}] true={true:16s} pred={label:16s} "
          f"seq={df['seq'].iloc[len(y_train)+i]}")'''

notebooks = {}

# NOTE: the microscope CNN lives in 08_CNN.ipynb (see _gen_notebooks_microscope_hr.py);
# this script only generates the RNN notebook (renamed 07).

notebooks["07_RNN_LSTM.ipynb"] = [
    md("# 🧬 RNN / LSTM — DNA Promoter Sequence Classification (PyTorch)\n"
       "Public database: **UCI Molecular Biology (Promoter Gene Sequences)** — "
       "106 E. coli promoter regions (57 bp). An LSTM reads the DNA sequence "
       "base-by-base and predicts whether it is a promoter (+)."),
    cell("code", RNN_CODE_1),
    cell("code", RNN_CODE_2),
    cell("code", RNN_CODE_3),
    cell("code", RNN_CODE_4),
    cell("code", RNN_CODE_5),
]

for fname, cells in notebooks.items():
    path = os.path.join(BASE, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1, ensure_ascii=False)
    print(f"Wrote {fname}")

print("\nDone — RNN notebook generated.")
