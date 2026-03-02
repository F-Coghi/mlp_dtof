# Train a simple fully-connected autoencoder on the 1D signals loaded from .npz.
# The goal is to compress each signal into a 4D latent vector and then reconstruct it.
# Saves the trained model and a quick reconstruction plot for one sample.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from DatasetNpz import npzDataset
import numpy as np
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tabella_path = os.path.join(BASE_DIR, "..", "ProcessedPoisson", "legenda.txt")
npz_dir = os.path.dirname(tabella_path)
df = pd.read_csv(tabella_path, delim_whitespace=True)
df = df[df.iloc[:, 0].apply(lambda x: os.path.exists(os.path.join(npz_dir, f'{x.rsplit('.mat')[0]}.npz')))]
dataset = npzDataset(df, npz_dir)

# not so large, neither small, but the batch_size worked fine on the machine used.
dataloader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=32)
# It's possible to play with the batch_size

# Get input shape from first sample
sample, _ = dataset[0]
input_dim = sample.numel()
print("input_dim:", input_dim)
print("sample shape:", sample.shape)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        # make sure it's [B, input_dim]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        # reshape back like original sample (usually 1D anyway)
        out = out.view(x.size(0), *sample.shape)
        return out

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#print(device)

model = Autoencoder(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# lr smaller than 6e-3 learning is hampered (loss don't change much over epochs) and reconstructed 
# data is very bad. lr 5e-3 gives quickest convergence to smallest loss of 0.0010, 
# similar result with lr 4e-3. 
criterion = nn.MSELoss()

# Training loop
# epoch 10 seems good (any greater don't see an effective improvement in the loss)
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for arr, _ in dataloader:
        arr = arr.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(arr)
        loss = criterion(output, arr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * arr.size(0)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}')

save_dir = os.path.join(BASE_DIR, "Autoencoder")
os.makedirs(save_dir, exist_ok=True)
torch.save(model, os.path.join(save_dir, "autoencoder_model"))

# plot a reconstruction
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    arr, _ = dataset[0]
    arr = arr.to(device).unsqueeze(0)
    output = model(arr).cpu().squeeze().numpy()
    arr = arr.cpu().squeeze().numpy()
    plt.figure()
    plt.plot(arr)
    plt.plot(output)
    plt.xscale('log')
    plt.title('Reconstructed')
    plt.savefig("Sample.png")
