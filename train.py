# ================== IMPORTS ==================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import numpy as np
import random
import time
from tqdm import tqdm
import pandas as pd

# ================== PATHS ==================
TRAIN_DIR = r"D:\train_spectrograms"
VAL_DIR   = r"D:\val_spectrograms"
DEV_DIR   = r"D:\dev_spectrograms"

TSV_TRACK1 = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\ASVspoof5.dev.track_1.tsv"
CODEC_CSV  = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\DATASET\protocols\ASVspoof5.codec.config.csv"

# ================== CONFIG ==================
BATCH_SIZE = 64
EPOCHS = 5   # 🔥 IMPORTANT: reduced (prevents overfitting)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 3

TRAIN_SAMPLES = 100000
VAL_SAMPLES = 80000
DEV_SAMPLES = None

# 🔥 FIX: stronger spoof penalty
CLASS_WEIGHTS = torch.tensor([1.0, 5.0]).to(DEVICE)

# ================== NORMALIZATION ==================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

# ================== CODEC MAP ==================
FILE_TO_CODEC_ID = {}

def load_codec_map(csv_path):
    try:
        df = pd.read_csv(csv_path)
        codec_map = {}
        for _, row in df.iterrows():
            cid = str(row["ID"]).strip()
            codec = str(row["CODEC"]).strip()
            if cid.startswith("C") and cid[1:].isdigit():
                codec_map[cid] = codec
        return codec_map
    except:
        print("⚠️ Codec CSV not found, continuing without names")
        return {}

CODEC_MAP = load_codec_map(CODEC_CSV)

def load_track1_ids(tsv):
    ids = set()
    with open(tsv, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            fid = parts[1].lower()
            ids.add(fid)

            for part in parts:
                base = part.upper().split("_")[0]
                if base in CODEC_MAP:
                    FILE_TO_CODEC_ID[fid] = base
    return ids

dev_ids = load_track1_ids(TSV_TRACK1)

def extract_codec_id(fid):
    return FILE_TO_CODEC_ID.get(fid)

# ================== EER ==================
def compute_eer(labels, scores):
    thresholds = np.linspace(0,1,1000)
    labels, scores = np.array(labels), np.array(scores)

    frr, far = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        spoof = labels == 1
        bona  = labels == 0
        frr.append(np.mean(preds[spoof] != labels[spoof]) if spoof.any() else 0)
        far.append(np.mean(preds[bona] != labels[bona]) if bona.any() else 0)

    idx = np.argmin(np.abs(np.array(frr) - np.array(far)))
    return (frr[idx] + far[idx]) / 2 * 100, thresholds[idx]

# ================== AUGMENTATION ==================
def spec_augment(x):
    _, f, t = x.shape
    fmask = random.randint(0, min(30, f))
    tmask = random.randint(0, min(40, t))
    f0 = random.randint(0, max(0, f - fmask))
    t0 = random.randint(0, max(0, t - tmask))
    x[:, f0:f0+fmask, :] = 0
    x[:, :, t0:t0+tmask] = 0

    if random.random() < 0.3:
        x += torch.randn_like(x) * 0.1

    # 🔥 codec simulation
    if random.random() < 0.5:
        x = torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)

    if random.random() < 0.5:
        x = torch.round(x * 10) / 10

    return x

# ================== DATASET ==================
class SpectrogramDataset(IterableDataset):
    def __init__(self, files, shuffle=False, max_samples=None, augment=False):
        self.files = list(files)
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.augment = augment

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)

        count = 0

        while True:
            for file in files:
                d = torch.load(file, map_location="cpu")
                xs, ys, fids = d["spectrograms"], d["labels"], d.get("file_ids", None)

                for i in range(len(xs)):
                    if self.max_samples and count >= self.max_samples:
                        return

                    x = xs[i].float()
                    if x.shape[0] == 1:
                        x = x.repeat(3,1,1)

                    y = ys[i].long()

                    fid = Path(str(fids[i])).stem.lower() if fids else "unknown"

                    if self.augment:
                        x = spec_augment(x)

                    yield x, y, fid
                    count += 1

            if self.max_samples is None:
                return

# ================== MODEL ==================
model = models.resnet18(weights="DEFAULT")

# 🔥 stronger freezing
for name, param in model.named_parameters():
    if "layer1" in name or "layer2" in name or "layer3" in name:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(model.fc.in_features, 2)
)

model = model.to(DEVICE)

# ================== OPTIM ==================
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-3
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================== LOADERS ==================
train_loader = DataLoader(SpectrogramDataset(Path(TRAIN_DIR).glob("*.pt"), True, TRAIN_SAMPLES, True), batch_size=BATCH_SIZE)
val_loader   = DataLoader(SpectrogramDataset(Path(VAL_DIR).glob("*.pt"), False, VAL_SAMPLES), batch_size=BATCH_SIZE)
dev_loader   = DataLoader(SpectrogramDataset(Path(DEV_DIR).glob("*.pt"), False, DEV_SAMPLES), batch_size=BATCH_SIZE)

train_total = TRAIN_SAMPLES // BATCH_SIZE

# ================== EVAL ==================
def evaluate(model, loader):
    model.eval()
    all_l, all_s = [], []

    with torch.no_grad():
        for x,y,_ in tqdm(loader, desc="DEV"):
            x = normalize(x.to(DEVICE))
            y = y.to(DEVICE)

            probs = torch.softmax(model(x), dim=1)[:,1]

            all_l.extend(y.cpu().numpy())
            all_s.extend(probs.cpu().numpy())

    eer, thr = compute_eer(all_l, all_s)

    preds = (np.array(all_s) >= thr).astype(int)

    print_confusion(all_l, preds, "Dev Set")
    print(f"\nEER: {eer:.2f}% | Threshold: {thr:.3f}")

    return eer

def print_confusion(labels, preds, name):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"\n{name}")
    print(f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

# ================== TRAIN ==================
best_eer = 100
no_improve = 0

for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    model.train()
    correct, total = 0, 0

    bar = tqdm(train_loader, total=train_total)

    for x,y,_ in bar:
        x,y = normalize(x.to(DEVICE)), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

        bar.set_postfix(acc=correct/total)

    eer = evaluate(model, dev_loader)

    scheduler.step()

    if eer < best_eer:
        best_eer = eer
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("⭐ Saved best model")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("🛑 Early stopping")
            break

print(f"\n🏁 BEST EER: {best_eer:.2f}%")