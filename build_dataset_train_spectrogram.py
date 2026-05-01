# ================== CHANGES ==================
# 1. UNIQUE shard naming (train_*, val_*)
# 2. Clear directories BEFORE build
# 3. Safe speaker split already included

import os
import torch
import torchaudio
from pathlib import Path
import torch.nn.functional as F
import time
import hashlib
import shutil

# ================== CONFIG ==================
RAW_AUDIO_DIR = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\MAJOR PROJECT FINAL EXTRA\flac_T"

TRAIN_OUT_DIR = r"D:\train_spectrograms"
VAL_OUT_DIR   = r"D:\val_spectrograms"

PROTOCOL_PATH = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\MAJOR PROJECT FINAL EXTRA\ASVspoof5.train.tsv"

SAMPLE_RATE = 16000
N_MELS = 128
CHUNK_SECONDS = 2
SHARD_SIZE = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== CLEAN OLD DATA ==================
def clean_dirs():
    if os.path.exists(TRAIN_OUT_DIR):
        shutil.rmtree(TRAIN_OUT_DIR)
    if os.path.exists(VAL_OUT_DIR):
        shutil.rmtree(VAL_OUT_DIR)

    os.makedirs(TRAIN_OUT_DIR)
    os.makedirs(VAL_OUT_DIR)

# ================== MEL ==================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=N_MELS
).to(DEVICE)

# ================== HASH ==================
def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

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

# ================== CHUNK ==================
def split_into_chunks(waveform, sr):
    chunk_size = int(sr * CHUNK_SECONDS)
    return [
        waveform[:, i:i+chunk_size]
        for i in range(0, waveform.shape[1], chunk_size)
        if waveform[:, i:i+chunk_size].shape[1] == chunk_size
    ]

# ================== PROCESS ==================
def process_chunks_batch(chunks):
    chunks = torch.stack(chunks).to(DEVICE)
    mel = mel_transform(chunks)
    log_mel = torch.log(mel + 1e-9)

    log_mel = F.interpolate(log_mel, size=(224, 224), mode='bilinear')
    return log_mel.half().cpu()

# ================== SHARD PROCESS ==================
def process_shards(file_list, save_dir, protocol, prefix):
    shard_specs, shard_labels = [], []
    shard_id = 0

    total = len(file_list)
    start = time.time()

    for idx, audio_path in enumerate(file_list):
        file_id = audio_path.stem
        label = protocol[file_id]

        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            continue

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        chunks = split_into_chunks(waveform, SAMPLE_RATE)
        if not chunks:
            continue

        specs = process_chunks_batch(chunks)

        shard_specs.extend(specs)
        shard_labels.extend([label] * len(specs))

        if (idx + 1) % SHARD_SIZE == 0:
            torch.save({
                "spectrograms": torch.stack(shard_specs),
                "labels": torch.tensor(shard_labels)
            }, Path(save_dir) / f"{prefix}_shard_{shard_id}.pt")

            print(f"✅ {prefix} shard {shard_id} ({len(shard_labels)})")

            shard_id += 1
            shard_specs, shard_labels = [], []

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start
            avg = elapsed / (idx + 1)
            eta = (total - idx - 1) * avg
            print(f"[{idx+1}/{total}] {avg:.2f}s/file | ETA {eta/60:.1f} min")

    if shard_specs:
        torch.save({
            "spectrograms": torch.stack(shard_specs),
            "labels": torch.tensor(shard_labels)
        }, Path(save_dir) / f"{prefix}_shard_{shard_id}.pt")

# ================== MAIN ==================
def main():
    print("🚀 CLEAN BUILD START")
    clean_dirs()  # 🔥 CRITICAL

    protocol, speaker_map = load_protocol(PROTOCOL_PATH)

    audio_files = list(Path(RAW_AUDIO_DIR).rglob("*.flac"))
    filtered = [p for p in audio_files if p.stem in protocol]

    train_files, val_files = [], []

    for p in filtered:
        speaker = speaker_map[p.stem]
        if stable_hash(speaker) % 10 < 8:
            train_files.append(p)
        else:
            val_files.append(p)

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    process_shards(train_files, TRAIN_OUT_DIR, protocol, "train")
    process_shards(val_files, VAL_OUT_DIR, protocol, "val")

    print("✅ DONE (NO LEAKAGE NOW)")

if __name__ == "__main__":
    main()