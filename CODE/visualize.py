import torch
import matplotlib.pyplot as plt

# load data
data = torch.load("D:/train_spectrograms/spec_batch_0.pt")

specs = data["spectrograms"]

# pick one sample
spec = specs[10]  # shape (3, 224, 224)

# convert to single channel (take one channel)
spec = spec[0].numpy()

# plot
plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Log-Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.show()