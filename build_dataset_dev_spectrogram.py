# ================== DEV DATASET BUILDER ==================
# ✅ Supports BOTH track1 + track2 TSV formats
# ✅ Stores file_id (CRITICAL)
# ✅ SAME spectrogram pipeline as training

import os
import torch
import torchaudio
from pathlib import Path
import torch.nn.functional as F
import time
import shutil

# ================== CONFIG ==================
RAW_AUDIO_DIR = r"D:\flac_D"

DEV_OUT_DIR = r"D:\dev_spectrograms"

TSV_TRACK1 = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\ASVspoof5.dev.track_1.tsv"
TSV_TRACK2 = r"C:\Users\Deepal\Desktop\College\SEM VI\MAJOR PROJECT FINAL\ASVspoof5.dev.track_2.trial.tsv"

SAMPLE_RATE = 16000
N_MELS = 128
CHUNK_SECONDS = 2
SHARD_SIZE = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== CLEAN ==================
def clean_dir():
    if os.path.exists(DEV_OUT_DIR):
        shutil.rmtree(DEV_OUT_DIR)
    os.makedirs(DEV_OUT_DIR)

# ================== MEL ==================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=N_MELS
).to(DEVICE)

# ================== LOAD TSVs (IMPORTANT FIX) ==================
def load_protocols(tsv1, tsv2):
    mapping = {}

    def load(tsv):
        with open(tsv, "r") as f:
            for line in f:
                parts = line.strip().split()

                # -------- COMMON --------
                file_id = parts[1]

                # -------- HANDLE BOTH FORMATS --------
                if "bonafide" in parts:
                    label = 0
                elif "spoof" in parts:
                    label = 1
                else:
                    continue  # skip weird lines

                mapping[file_id] = label

    load(tsv1)
    load(tsv2)

    print(f"Loaded total files from TSVs: {len(mapping)}")
    return mapping

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
def process_files(file_list, protocol):
    shard_specs, shard_labels, shard_file_ids = [], [], []
    shard_id = 0

    total = len(file_list)
    start = time.time()

    for idx, audio_path in enumerate(file_list):
        file_id = audio_path.stem

        if file_id not in protocol:
            continue

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

        # ✅ STORE
        shard_specs.extend(specs)
        shard_labels.extend([label] * len(specs))
        shard_file_ids.extend([file_id] * len(specs))  # 🔥 CRITICAL

        # ===== SAVE =====
        if (idx + 1) % SHARD_SIZE == 0:
            torch.save({
                "spectrograms": torch.stack(shard_specs),
                "labels": torch.tensor(shard_labels),
                "file_ids": shard_file_ids
            }, Path(DEV_OUT_DIR) / f"dev_shard_{shard_id}.pt")

            print(f"✅ Saved shard {shard_id} ({len(shard_labels)})")

            shard_id += 1
            shard_specs, shard_labels, shard_file_ids = [], [], []

        # ===== ETA =====
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start
            avg = elapsed / (idx + 1)
            eta = (total - idx - 1) * avg
            print(f"[{idx+1}/{total}] {avg:.2f}s/file | ETA {eta/60:.1f} min")

    # FINAL SAVE
    if shard_specs:
        torch.save({
            "spectrograms": torch.stack(shard_specs),
            "labels": torch.tensor(shard_labels),
            "file_ids": shard_file_ids
        }, Path(DEV_OUT_DIR) / f"dev_shard_{shard_id}.pt")

# ================== MAIN ==================
def main():
    print("🚀 BUILDING DEV DATASET")

    clean_dir()

    protocol = load_protocols(TSV_TRACK1, TSV_TRACK2)

    audio_files = list(Path(RAW_AUDIO_DIR).rglob("*.flac"))
    print(f"Total audio found: {len(audio_files)}")

    process_files(audio_files, protocol)

    print("✅ DEV BUILD COMPLETE")

# ================== ENTRY ==================
if __name__ == "__main__":
    main()