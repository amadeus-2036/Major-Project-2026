import torch
import torchaudio
import torch.nn.functional as F
from CODE.config import SAMPLE_RATE, N_MELS

# module-level placeholder — filled by init_spectrogram_transform() in each worker
_mel_transform = None

def init_spectrogram_transform():
    """Called once per worker process via init_worker. Never called per-file."""
    global _mel_transform
    _mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )

def process_chunk(chunk):
    waveform = torch.tensor(chunk).float().unsqueeze(0)

    mel     = _mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)             # (1, n_mels, T)

    log_mel = log_mel.unsqueeze(0)
    log_mel = F.interpolate(                     # (1, 1, 224, 224)
        log_mel,
        size=(224, 224),
        mode='bilinear',
        align_corners=False                      # suppress torchvision warning
    )
    log_mel = log_mel.squeeze(0)                 # (1, 224, 224)

    # ✅ float16 + 1 channel = 6x smaller than your original float32 3-channel
    # repeat(3) happens at training time for free via expand() — zero memory copy
    return log_mel.half()                        # (1, 224, 224) float16