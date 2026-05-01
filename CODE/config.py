# src/config.py

SAMPLE_RATE = 16000
CHUNK_DURATION = 4
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

N_MELS = 128

BATCH_SIZE_SAVE = 500

RAW_AUDIO_DIR = "C:/Users/Deepal/Desktop/College/SEM VI/MAJOR PROJECT FINAL/DATASET/trial_flac_T"
SAVE_DIR = "D:/train_spectrograms"

NUM_WORKERS = 6  # adjust based on CPU cores
DEVICE = "cuda"  # used later in training, not here heavily