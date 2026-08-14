#!/usr/bin/env python
"""Generate high-resolution microscope CNN notebook (BloodMNIST 224px)."""
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


C1 = '''# ============================================================
# Setup & download high-resolution microscope dataset (224x224)
# ============================================================
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import BloodMNIST

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- BloodMNIST at native high resolution (224x224) ----
train_ds = BloodMNIST(split="train", download=True, size=224)
test_ds  = BloodMNIST(split="test",  download=True, size=224)
class_names = [train_ds.info["label"][str(i)] for i in range(8)]
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
print(f"Image shape: {np.array(train_ds[0][0]).shape} (RGB)")'''

C2 = '''# ============================================================
# Visualize samples (224x224)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
shown = set()
for ax in axes.flat:
    for img, label in train_ds:
        c = int(label[0])
        if c in shown:
            continue
        shown.add(c)
        ax.imshow(img)
        ax.set_title(f"{c}: {class_names[c].split()[0]}", fontsize=8)
        ax.axis("off")
        break
plt.suptitle("BloodMNIST 224x224 — microscope blood-cell images")
plt.tight_layout(); plt.show()'''

C3 = '''# ============================================================
# DataLoaders + plain CNN (no BatchNorm, no augmentation)
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


class MedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(int(label[0]), dtype=torch.long)


train_loader = DataLoader(MedDataset(train_ds, transform),
                          batch_size=64, shuffle=True)
test_loader  = DataLoader(MedDataset(test_ds, transform),
                          batch_size=64, shuffle=False)


class BloodCNN(nn.Module):
    """3 conv blocks + global average pooling (no BatchNorm)."""
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))


model = BloodCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")'''

C4 = '''# ============================================================
# Training loop (6 epochs, ~3 min/epoch on CPU at 224px)
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


EPOCHS = 6
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(train_loader, model, criterion,
                                      optimizer, device)
    te_acc = evaluate(test_loader, model, device)
    print(f"Epoch {epoch:2d} | loss {tr_loss:.4f} | train acc {tr_acc:.4f} | "
          f"test acc {te_acc:.4f} | {time.time()-t0:.0f}s")'''

C5 = '''# ============================================================
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
print(classification_report(y_true, y_pred, target_names=class_names,
                            digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"{i}\\n{n.split()[0]}" for i, n in enumerate(class_names)],
            yticklabels=[f"{i}\\n{n.split()[0]}" for i, n in enumerate(class_names)])
plt.title("BloodMNIST 224x224 CNN — confusion matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ---- misclassified examples at full resolution ----
mis_idx = np.where(y_true != y_pred)[0][:8]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = np.array(test_ds[mis_idx[i]][0])
    ax.imshow(img)
    ax.set_title(f"true={class_names[y_true[mis_idx[i]]].split()[0]}\\n"
                 f"pred={class_names[y_pred[mis_idx[i]]].split()[0]}",
                 fontsize=8, color="red")
    ax.axis("off")
plt.suptitle("Misclassified examples (224x224)")
plt.tight_layout(); plt.show()'''

notebook = [
    md("# 🧬 CNN on Microscope Images — 224x224 (PyTorch)\n"
       "**Public dataset:** BloodMNIST (MedMNIST) at **224x224** resolution — "
       "real microscope bright-field images of 8 blood-cell types.\n"
       "A plain CNN (no BatchNorm, no augmentation) used directly for "
       "classification."),
    cell("code", C1),
    cell("code", C2),
    cell("code", C3),
    cell("code", C4),
    cell("code", C5),
]

path = os.path.join(BASE, "08_CNN.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(make_notebook(notebook), f, indent=1, ensure_ascii=False)
print("Wrote 08_CNN.ipynb")
