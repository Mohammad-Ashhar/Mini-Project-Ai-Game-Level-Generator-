"""
Toy training loop to experiment with representing levels as small images / grids.
This is NOT a production model — it's a scaffold to help you try training quickly.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GridDataset(Dataset):
    def __init__(self, grids):
        # grids: list of numpy arrays shaped (H,W)
        self.grids = [torch.tensor(g, dtype=torch.float32) for g in grids]

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        # return normalized float grid
        g = self.grids[idx]
        return g.unsqueeze(0)  # (1,H,W)

class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16,8,3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8,1,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out

def train(grids, epochs=5):
    ds = GridDataset(grids)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = TinyAutoencoder()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for e in range(epochs):
        total = 0.0
        for b in dl:
            b = b
            out = model(b)
            loss = loss_fn(out, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b.size(0)
        print(f"Epoch {e+1}/{epochs} Loss: {total/len(ds):.6f}")
    torch.save(model.state_dict(), "toy_autoencoder.pt")
    print("Saved toy_autoencoder.pt")

if __name__ == "__main__":
    # toy random data to test
    grids = [np.random.randint(0, 2, size=(12,40)) for _ in range(100)]
    train(grids, epochs=3)
