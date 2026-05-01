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
TRAIN_DIR      = r"D:\train_spectrograms"
VAL_DIR        = r"D:\val_spectrograms"
DEV_DIR        = r"D:\dev_spectrograms"

TSV_TRACK1     = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\ASVspoof5.dev.track_1.tsv"
CODEC_CSV      = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\DATASET\protocols\ASVspoof5.codec.config.csv"

# ================== CONFIG ==================
BATCH_SIZE    = 64
EPOCHS        = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE      = 4  # Increased slightly to give the model time to generalize

TRAIN_SAMPLES = 100000
VAL_SAMPLES   = 80000
DEV_SAMPLES   = None 

CLASS_WEIGHTS = torch.tensor([1.0, 3.0]).to(DEVICE)

# ================== NORMALIZATION ==================
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)

# ================== CODEC CONFIG ==================
FILE_TO_CODEC_ID = {} 

def load_codec_map(csv_path):
    try:
        df = pd.read_csv(csv_path)
        codec_map = {}
        for _, row in df.iterrows():
            cid   = str(row["ID"]).strip()
            codec = str(row["CODEC"]).strip()
            if cid.startswith("C") and cid[1:].isdigit() and codec not in ("nan", "NaN"):
                if cid not in codec_map:
                    codec_map[cid] = codec
        return codec_map
    except Exception as e:
        print(f"⚠️ Warning: Could not load codec map. {e}")
        return {}

CODEC_MAP = load_codec_map(CODEC_CSV)
print("Codec map loaded:", CODEC_MAP)

def extract_codec_id(fid):
    return FILE_TO_CODEC_ID.get(fid)

# ================== TSV HUNTER ==================
def load_track1_ids(tsv):
    ids = set()
    with open(tsv, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            fid = parts[1].strip().lower()
            ids.add(fid)
            
            for part in parts:
                p_upper = part.upper().strip()
                base_c = p_upper.split('_')[0] 
                
                if base_c in CODEC_MAP:
                    FILE_TO_CODEC_ID[fid] = base_c
                elif base_c.startswith("AC") and base_c.replace("AC", "C") in CODEC_MAP:
                    FILE_TO_CODEC_ID[fid] = base_c.replace("AC", "C")
    return ids

dev_ids = load_track1_ids(TSV_TRACK1)
print(f"Track 1 Dev IDs mapped: {len(dev_ids):,}")
print(f"Total files identified as CODEC COMPRESSED: {len(FILE_TO_CODEC_ID):,}")

# ================== EER ==================
def compute_eer(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    thresholds = np.linspace(0, 1, 1000)
    frr_list, far_list = [], []
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        spoof_idx = labels == 1
        frr = np.mean(preds[spoof_idx] != labels[spoof_idx]) if spoof_idx.any() else 0.0
        bon_idx = labels == 0
        far = np.mean(preds[bon_idx] != labels[bon_idx]) if bon_idx.any() else 0.0
        frr_list.append(frr)
        far_list.append(far)
    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)
    idx = np.argmin(np.abs(frr_arr - far_arr))
    eer = (frr_arr[idx] + far_arr[idx]) / 2 * 100
    return eer, thresholds[idx]

def print_confusion(labels, preds, name):
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    frr = fn / max(1, fn + tp) * 100
    far = fp / max(1, fp + tn) * 100
    print(f"\n  Confusion Matrix — {name}")
    print(f"  {'':15s}  Pred Bonafide  Pred Spoof")
    print(f"  {'True Bonafide':15s}  {tn:>13,}  {fp:>10,}")
    print(f"  {'True Spoof':15s}  {fn:>13,}  {tp:>10,}")
    print(f"  FRR (spoof missed)     : {frr:.2f}%")
    print(f"  FAR (bonafide flagged) : {far:.2f}%")

# ================== ADVANCED TELECOM AUGMENTATION ==================
def spec_augment(x):
    _, freq, time_len = x.shape
    
    # 1. Frequency Masking (Simulates Codec Bandwidth Limits)
    f  = random.randint(0, min(40, freq))
    f0 = random.randint(0, max(0, freq - f))
    x[:, f0:f0 + f, :]  = 0
    
    # 2. Time Masking (Simulates packet loss / dropouts)
    t  = random.randint(0, min(50, time_len))
    t0 = random.randint(0, max(0, time_len - t))
    x[:, :, t0:t0 + t]  = 0
    
    # 3. Telecom Noise Injection (Forces robust feature learning)
    if random.random() < 0.3:  # 30% chance to add static noise
        noise = torch.randn_like(x) * 0.1
        x = x + noise
        
    return x

# ================== DATASET ==================
class SpectrogramDataset(IterableDataset):
    def __init__(self, files, shuffle=False, max_samples=None, augment=False):
        self.files       = list(files)
        self.shuffle     = shuffle
        self.max_samples = max_samples
        self.augment     = augment

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("No multi-worker support")

        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)
        count = 0

        while True:
            for file in files:
                d        = torch.load(file, map_location="cpu")
                x_all    = d["spectrograms"]
                y_all    = d["labels"]
                file_ids = d.get("file_ids", None)

                idxs = list(range(len(x_all)))
                if self.shuffle:
                    random.shuffle(idxs)

                for i in idxs:
                    if self.max_samples and count >= self.max_samples:
                        return

                    x = x_all[i].float()
                    if x.shape[0] == 1:
                        x = x.repeat(3, 1, 1)

                    y = y_all[i].long()

                    if file_ids is not None:
                        raw = str(file_ids[i])
                        raw_stem = Path(raw).stem
                        for ext in [".flac", ".wav", ".pt"]:
                            if raw_stem.lower().endswith(ext):
                                raw_stem = raw_stem[:-len(ext)]
                        fid = raw_stem.strip().lower()
                    else:
                        fid = "unknown"

                    if self.augment:
                        x = spec_augment(x)

                    yield x, y, fid            
                    count += 1
            
            if self.max_samples is None:
                return

# ================== MODEL ==================
model = models.resnet18(weights="DEFAULT")

for pname, param in model.named_parameters():
    if "layer1" in pname or "layer2" in pname:
        param.requires_grad = False

# Increased Dropout to 0.6 to fight extreme overfitting
model.fc = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(DEVICE)

# ================== OPTIMISER ==================
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=0.05)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=1e-3  # Increased weight decay (L2 Regularization)
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ],
    milestones=[2]
)

# ================== DATA ==================
train_files = sorted(Path(TRAIN_DIR).glob("*.pt"))
val_files   = sorted(Path(VAL_DIR).glob("*.pt"))
dev_files   = sorted(Path(DEV_DIR).glob("*.pt"))

print(f"Train: {len(train_files)} files | Val: {len(val_files)} files | Dev: {len(dev_files)} files")

train_loader = DataLoader(SpectrogramDataset(train_files, True,  TRAIN_SAMPLES, True), batch_size=BATCH_SIZE)
val_loader   = DataLoader(SpectrogramDataset(val_files,   False, VAL_SAMPLES),         batch_size=BATCH_SIZE)
dev_loader   = DataLoader(SpectrogramDataset(dev_files,   False, DEV_SAMPLES),         batch_size=BATCH_SIZE)

train_total = TRAIN_SAMPLES // BATCH_SIZE
val_total   = VAL_SAMPLES   // BATCH_SIZE

# ================== VAL EVAL ==================
def evaluate_val(model, loader, total_batches):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, total=total_batches, desc="VAL"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = normalize(x)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\n  VAL")
    print(classification_report(all_labels, all_preds, target_names=["Bonafide", "Spoof"]))
    return sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

# ================== DEV EVAL ==================
def evaluate_both(model, loader):
    model.eval()

    all_p, all_l, all_s = [], [], []
    bona_l, bona_s = [], []
    
    codec_spoof_p = defaultdict(list)
    codec_spoof_l = defaultdict(list)
    codec_spoof_s = defaultdict(list)

    with torch.no_grad():
        for x, y, fids in tqdm(loader, desc="DEV"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = normalize(x)

            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)

            for i, fid in enumerate(fids):
                if fid not in dev_ids:
                    continue

                p = preds[i].item()
                l = y[i].item()
                s = probs[i, 1].item()

                all_p.append(p); all_l.append(l); all_s.append(s)

                if l == 0: 
                    bona_l.append(l)
                    bona_s.append(s)
                elif l == 1: 
                    cid = extract_codec_id(fid)
                    if cid:
                        codec_spoof_p[cid].append(p)
                        codec_spoof_l[cid].append(l)
                        codec_spoof_s[cid].append(s)

    # ══════════════════════════════════════════════════
    #  OVERALL TRACK 1 EVALUATION
    # ══════════════════════════════════════════════════
    print("\n" + "="*55)
    print("  OVERALL TRACK 1 EVALUATION (All data combined)")
    print("="*55)
    if all_l:
        print_confusion(all_l, all_p, "Overall Dev Set")
        overall_eer, overall_thr = compute_eer(all_l, all_s)
        print(f"\n  OVERALL EER : {overall_eer:.2f}%  (threshold = {overall_thr:.3f})")
    else:
        overall_eer = 100.0

    # ══════════════════════════════════════════════════
    #  OFFICIAL ASVSPOOF PER-CODEC BREAKDOWN
    # ══════════════════════════════════════════════════
    if codec_spoof_l:
        print("\n" + "="*55)
        print("  PER-CODEC ROBUSTNESS")
        print("="*55)
        print(f"  {'Codec':<6}  {'Name':<20}  {'N(Spoofs)':>10}  {'Spoof Acc':>9}  {'EER':>7}")
        print(f"  {'-'*6}  {'-'*20}  {'-'*10}  {'-'*9}  {'-'*7}")

        for cid in sorted(codec_spoof_l.keys(), key=lambda x: int(x.replace('C', '')) if x.replace('C','').isdigit() else 99):
            eval_l = bona_l + codec_spoof_l[cid]
            eval_s = bona_s + codec_spoof_s[cid]
            
            ps = codec_spoof_p[cid]
            ls = codec_spoof_l[cid]
            spoof_acc = sum(p == l for p, l in zip(ps, ls)) / len(ls)
            
            eer, _ = compute_eer(eval_l, eval_s)
            name = CODEC_MAP.get(cid, f"Codec {cid}")
            print(f"  {cid:<6}  {name:<20}  {len(ls):>10,}  {spoof_acc:>9.4f}  {eer:>6.2f}%")

    overall_acc = sum(p == l for p, l in zip(all_p, all_l)) / max(1, len(all_l)) if all_l else 0
    return overall_acc, overall_eer

# ================== TRAIN ==================
total_start = time.time()
best_eer    = 100.0  # WE NOW TARGET THE LOWEST EER
no_improve  = 0

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\n{'='*55}")
    print(f"🔥 Epoch {epoch+1}/{EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
    print(f"{'='*55}")

    model.train()
    correct, total = 0, 0
    train_bar = tqdm(train_loader, total=train_total, desc=f"Train {epoch+1}")

    for x, y, _ in train_bar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = normalize(x)

        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds    = out.argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
        train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    train_acc = correct / total

    val_acc = evaluate_val(model, val_loader, val_total)
    overall_acc, overall_eer = evaluate_both(model, dev_loader)

    scheduler.step()

    epoch_time = time.time() - epoch_start
    eta        = (EPOCHS - epoch - 1) * (time.time() - total_start) / (epoch + 1)

    print(f"\n{'─'*55}")
    print(f"  Train : {train_acc:.4f}")
    print(f"  Dev   : Acc={overall_acc:.4f}  EER={overall_eer:.2f}%  ← Checkpoint signal")
    print(f"  Time  : {epoch_time/60:.1f} min  |  ETA: {eta/60:.1f} min")
    print(f"{'─'*55}")

    # SAVING BASED ON EER DECREASE, NOT ACCURACY INCREASE
    if overall_eer < best_eer:
        best_eer = overall_eer
        no_improve = 0
        torch.save({
            "epoch":       epoch + 1,
            "model":       model.state_dict(),
            "overall_acc": overall_acc,  
            "overall_eer": overall_eer,
        }, "best_model.pth")
        print(f"  ⭐ Saved! New Best Dev EER: {overall_eer:.2f}%")
    else:
        no_improve += 1
        print(f"  No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print("🛑 Early stopping triggered. Training Complete.")
            break

print(f"\n🏁 DONE  |  Best Dev EER Achieved: {best_eer:.2f}%")