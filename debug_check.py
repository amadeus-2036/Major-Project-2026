import os
import json
import torch
from pathlib import Path
import random
from tqdm import tqdm

# ================== PATHS ==================
TRAIN_DIR = r"D:\train_spectrograms"
VAL_DIR   = r"D:\val_spectrograms"

PROTOCOL_PATH = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\MAJOR PROJECT FINAL EXTRA\ASVspoof5.train.tsv"
SPLIT_FILE = "speaker_split.json"

# ================== LOAD PROTOCOL ==================
def load_protocol(tsv_path):
    mapping = {}
    speaker_map = {}

    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            file_id    = parts[1]
            label      = 0 if parts[8] == "bonafide" else 1

            mapping[file_id] = label
            speaker_map[file_id] = speaker_id

    return mapping, speaker_map

# ================== CHECK 1: FILE OVERLAP ==================
def check_file_overlap():
    train_files = set([p.name for p in Path(TRAIN_DIR).glob("*.pt")])
    val_files   = set([p.name for p in Path(VAL_DIR).glob("*.pt")])

    overlap = train_files.intersection(val_files)

    print("\n🔍 FILE OVERLAP CHECK")
    print("Overlap count:", len(overlap))

    if len(overlap) > 0:
        print("❌ LEAKAGE: SAME FILES IN TRAIN & VAL")
    else:
        print("✅ No file overlap")

# ================== CHECK 2: SPEAKER OVERLAP ==================
def check_speaker_overlap():
    print("\n🔍 SPEAKER OVERLAP CHECK")

    protocol, speaker_map = load_protocol(PROTOCOL_PATH)
    split = json.load(open(SPLIT_FILE))

    train_speakers = set()
    val_speakers = set()

    for file_id, spk in speaker_map.items():
        if split[spk] == "train":
            train_speakers.add(spk)
        else:
            val_speakers.add(spk)

    overlap = train_speakers.intersection(val_speakers)

    print("Train speakers:", len(train_speakers))
    print("Val speakers:", len(val_speakers))
    print("Overlap speakers:", len(overlap))

    if len(overlap) > 0:
        print("❌ LEAKAGE: SAME SPEAKER IN TRAIN & VAL")
    else:
        print("✅ No speaker overlap")

# ================== CHECK 3: LABEL DISTRIBUTION ==================
def check_labels():
    print("\n🔍 LABEL DISTRIBUTION CHECK")

    def count_labels(folder):
        counts = {0:0, 1:0}

        files = list(Path(folder).glob("*.pt"))
        for f in tqdm(files[:50], desc=f"Checking {folder}"):
            d = torch.load(f, map_location="cpu")
            labels = d["labels"]

            for l in labels:
                counts[int(l)] += 1

        return counts

    train_counts = count_labels(TRAIN_DIR)
    val_counts   = count_labels(VAL_DIR)

    print("\nTrain label distribution:", train_counts)
    print("Val label distribution:", val_counts)

# ================== CHECK 4: RANDOM SAMPLE VISUAL ==================
def inspect_samples():
    print("\n🔍 SAMPLE INSPECTION")

    train_file = random.choice(list(Path(TRAIN_DIR).glob("*.pt")))
    val_file   = random.choice(list(Path(VAL_DIR).glob("*.pt")))

    print("Train file:", train_file.name)
    print("Val file:", val_file.name)

    d1 = torch.load(train_file, map_location="cpu")
    d2 = torch.load(val_file, map_location="cpu")

    print("Train sample shape:", d1["spectrograms"][0].shape)
    print("Val sample shape:", d2["spectrograms"][0].shape)

# ================== CHECK 5: QUICK MODEL SANITY ==================
def quick_model_check():
    print("\n🔍 QUICK MODEL CHECK (EXTREME TEST)")

    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load few samples ONLY
    train_files = list(Path(TRAIN_DIR).glob("*.pt"))[:5]
    val_files   = list(Path(VAL_DIR).glob("*.pt"))[:5]

    def load_samples(files):
        X, Y = [], []
        for f in files:
            d = torch.load(f, map_location="cpu")
            for i in range(min(50, len(d["labels"]))):
                X.append(d["spectrograms"][i])
                Y.append(d["labels"][i])
        return torch.stack(X), torch.tensor(Y)

    X_train, y_train = load_samples(train_files)
    X_val, y_val     = load_samples(val_files)

    X_train = X_train.expand(-1,3,-1,-1).to(device)
    y_train = y_train.to(device)

    X_val = X_val.expand(-1,3,-1,-1).to(device)
    y_val = y_val.to(device)

    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # train few steps
    for _ in range(20):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
    val_acc   = (model(X_val).argmax(1) == y_val).float().mean().item()

    print(f"\nTrain Acc (tiny set): {train_acc:.4f}")
    print(f"Val   Acc (tiny set): {val_acc:.4f}")

# ================== MAIN ==================
if __name__ == "__main__":
    print("🚀 RUNNING FULL DATA PIPELINE CHECK")

    check_file_overlap()        # 🔥 critical
    check_speaker_overlap()    # 🔥 critical
    check_labels()             # sanity
    inspect_samples()          # sanity
    quick_model_check()        # 🔥 strongest test

    print("\n✅ ALL CHECKS DONE")