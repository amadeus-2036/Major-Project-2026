import librosa
from CODE.config import SAMPLE_RATE, CHUNK_SIZE

def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    return y

def split_into_chunks(y):
    chunks = []

    for i in range(0, len(y), CHUNK_SIZE):
        chunk = y[i:i+CHUNK_SIZE]

        if len(chunk) == CHUNK_SIZE:
            chunks.append(chunk)

    return chunks