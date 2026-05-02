import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
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
EPOCHS = 4
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SAMPLES = 100000
VAL_SAMPLES   = 80000
DEV_SAMPLES   = None

CLASS_WEIGHTS = torch.tensor([1.0, 6.0]).to(DEVICE)

# ================== NORMALIZATION ==================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

# ================== CODEC ==================
FILE_TO_CODEC_ID = {}

def load_codec_map(path):
    try:
        df = pd.read_csv(path)
        return {str(r["ID"]).strip(): str(r["CODEC"]).strip() for _, r in df.iterrows()}
    except:
        print("⚠️ Codec CSV missing")
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
            for p in parts:
                base = p.upper().split("_")[0]
                if base in CODEC_MAP:
                    FILE_TO_CODEC_ID[fid] = base
    return ids

dev_ids = load_track1_ids(TSV_TRACK1)

def extract_codec_id(fid):
    return FILE_TO_CODEC_ID.get(fid)

# ================== EER ==================
def compute_eer(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    thresholds = np.linspace(0,1,1000)
    frr, far = [], []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        spoof = labels == 1
        bona  = labels == 0

        frr.append(np.mean(preds[spoof] != labels[spoof]) if spoof.any() else 0)
        far.append(np.mean(preds[bona] != labels[bona]) if bona.any() else 0)

    idx = np.argmin(np.abs(np.array(frr)-np.array(far)))
    return (frr[idx]+far[idx])/2*100, thresholds[idx]

# ================== AUGMENT ==================
def spec_augment(x):
    _, f, t = x.shape
    f0 = random.randint(0, max(0,f-30))
    t0 = random.randint(0, max(0,t-40))
    x[:, f0:f0+30, :] = 0
    x[:, :, t0:t0+40] = 0
    if random.random() < 0.3:
        x += torch.randn_like(x)*0.1
    return x

def codec_augment(x):
    if random.random() < 0.7:
        x = torch.nn.functional.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
    if random.random() < 0.7:
        levels = random.choice([8,16,32])
        x = torch.round(x * levels) / levels
    if random.random() < 0.5:
        x = x * random.uniform(0.6,1.4)
    if random.random() < 0.7:
        _, f, _ = x.shape
        f0 = random.randint(0, f//2)
        f1 = random.randint(f0, f)
        x[:, f0:f1, :] *= random.uniform(0.0,0.3)
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
                        if random.random() < 0.7:
                            x = codec_augment(x)

                    yield x, y, fid
                    count += 1

            if self.max_samples is None:
                return

# ================== MODEL ==================
model = models.resnet18(weights="DEFAULT")

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
val_total   = VAL_SAMPLES // BATCH_SIZE

# ================== EVAL ==================
def evaluate_val(model):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x,y,_ in tqdm(val_loader, total=val_total, desc="VAL"):
            x = normalize(x.to(DEVICE))
            y = y.to(DEVICE)
            p = model(x).argmax(1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    print("\nVAL REPORT")
    print(classification_report(labels, preds))

def evaluate_dev(model):
    model.eval()

    all_p, all_l, all_s = [], [], []
    bona_l, bona_s = [], []
    codec_spoof = defaultdict(lambda: {"p":[], "l":[], "s":[]})

    with torch.no_grad():
        for x,y,fids in tqdm(dev_loader, desc="DEV"):
            x = normalize(x.to(DEVICE))
            y = y.to(DEVICE)

            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)

            for i,fid in enumerate(fids):
                if fid not in dev_ids: continue

                p = preds[i].item()
                l = y[i].item()
                s = probs[i,1].item()

                all_p.append(p); all_l.append(l); all_s.append(s)

                if l == 0:
                    bona_l.append(l); bona_s.append(s)
                else:
                    cid = extract_codec_id(fid)
                    if cid:
                        codec_spoof[cid]["p"].append(p)
                        codec_spoof[cid]["l"].append(l)
                        codec_spoof[cid]["s"].append(s)

    print("\n=== OVERALL DEV ===")
    tn, fp, fn, tp = confusion_matrix(all_l, all_p).ravel()
    print(f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    eer, thr = compute_eer(all_l, all_s)
    print(f"EER: {eer:.2f}% | Thr: {thr:.3f}")

    print("\n=== PER CODEC ===")
    for cid in codec_spoof:
        eval_l = bona_l + codec_spoof[cid]["l"]
        eval_s = bona_s + codec_spoof[cid]["s"]

        eer_c,_ = compute_eer(eval_l, eval_s)
        acc = np.mean(np.array(codec_spoof[cid]["p"]) == np.array(codec_spoof[cid]["l"]))

        print(f"{cid} | {CODEC_MAP.get(cid,'?')} | Acc={acc:.4f} | EER={eer_c:.2f}%")

    return eer

# ================== TRAIN ==================
best_eer = 100
no_improve = 0
start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*50}")

    model.train()
    correct, total = 0,0

    bar = tqdm(train_loader, total=train_total, desc="TRAIN")

    for x,y,_ in bar:
        x,y = normalize(x.to(DEVICE)), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)

        bar.set_postfix(acc=correct/total, loss=loss.item())

    evaluate_val(model)
    eer = evaluate_dev(model)

    scheduler.step()

    epoch_time = (time.time()-epoch_start)/60
    eta = (EPOCHS-epoch-1)*((time.time()-start)/(epoch+1))/60

    print(f"\nTrain Acc: {correct/total:.4f}")
    print(f"Time: {epoch_time:.1f} min | ETA: {eta:.1f} min")

    if eer < best_eer:
        best_eer = eer
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("⭐ Saved best model")
    else:
        no_improve += 1
        print(f"No improve ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print("Early stopping")
            break

print(f"\nBEST EER: {best_eer:.2f}%")