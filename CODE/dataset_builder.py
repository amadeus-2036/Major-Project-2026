# src/dataset_builder.py

import os
import torch
from tqdm import tqdm
from multiprocessing import Pool

from audio_utils import load_audio, split_into_chunks
from spectrogram_utils import process_chunk
from config import RAW_AUDIO_DIR, SAVE_DIR, BATCH_SIZE_SAVE, NUM_WORKERS


def get_label(file_name):
    if "spoof" in file_name:
        return 1
    else:
        return 0


def process_file(file_name):
    file_path = os.path.join(RAW_AUDIO_DIR, file_name)

    results = []

    try:
        audio = load_audio(file_path)
        chunks = split_into_chunks(audio)
        label = get_label(file_name)

        for chunk in chunks:
            spec = process_chunk(chunk)
            results.append((spec, label))

    except Exception as e:
        print(f"Error: {file_name}, {e}")

    return results


def build_dataset():
    os.makedirs(SAVE_DIR, exist_ok=True)

    audio_files = os.listdir(RAW_AUDIO_DIR)

    pool = Pool(NUM_WORKERS)

    batch_specs = []
    batch_labels = []
    save_index = 0

    for result in tqdm(pool.imap(process_file, audio_files), total=len(audio_files)):

        for spec, label in result:
            batch_specs.append(spec)
            batch_labels.append(label)

            if len(batch_specs) >= BATCH_SIZE_SAVE:
                save_batch(batch_specs, batch_labels, save_index)

                batch_specs = []
                batch_labels = []
                save_index += 1

    # save remaining
    if len(batch_specs) > 0:
        save_batch(batch_specs, batch_labels, save_index)

    pool.close()
    pool.join()


def save_batch(specs, labels, idx):
    specs_tensor = torch.stack(specs)
    labels_tensor = torch.tensor(labels)

    path = os.path.join(SAVE_DIR, f"spec_batch_{idx}.pt")

    torch.save({
        "spectrograms": specs_tensor,
        "labels": labels_tensor
    }, path)

    print(f"Saved {path}")

    print("Saving to:", SAVE_DIR)