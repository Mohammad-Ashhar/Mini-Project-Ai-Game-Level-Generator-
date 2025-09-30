# Train a tiny conditional generator that maps (seed,difficulty) -> tile grid (classes 0..4)
# Produces weights file project/ml/ml_model_weights.pt

import os
import math
import random
import hashlib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

OUT_PATH = os.path.dirname(__file__)
WEIGHTS_FILE = os.path.join(OUT_PATH, "ml_model_weights.pt")

# -------------------------
# Simple procedural generator for training data (platformer-like)
# -------------------------
def seed_to_int(s):
    if s is None:
        return random.getrandbits(64)
    h = hashlib.sha256(str(s).encode("utf8")).hexdigest()
    return int(h[:16], 16)

def gen_platformer_grid(width=80, height=24, difficulty=3, seed=None):
    rnd = random.Random(seed_to_int(seed))
    width = max(8, int(width))
    height = max(6, int(height))
    grid = [[0 for _ in range(width)] for _ in range(height)]
    ground_y = max(1, height - 2)
    gap_prob = 0.12 + max(0, difficulty) * 0.06
    for x in range(width):
        if rnd.random() > gap_prob:
            grid[ground_y][x] = 1
    # platforms
    platform_chance = 0.05 + max(0, difficulty) * 0.02
    for x in range(2, width - 2):
        if rnd.random() < platform_chance:
            max_len = min(6, width - x - 1)
            if max_len <= 0: continue
            plat_len = rnd.randint(1, max_len)
            max_platform_y = max(1, ground_y - 2)
            if max_platform_y <= 1:
                y = 1
            else:
                y = rnd.randint(2, max_platform_y)
            for dx in range(plat_len):
                if (x+dx) < width:
                    grid[y][x+dx] = 1
    # collectibles/enemies above platforms/ground
    for x in range(width):
        for y in range(height):
            if grid[y][x] == 1:
                above = y - 1
                if above >= 0:
                    if rnd.random() < 0.03 + max(0, difficulty) * 0.01 and grid[above][x] == 0:
                        grid[above][x] = 3
                    if rnd.random() < 0.02 + max(0, difficulty) * 0.015 and grid[above][x] == 0:
                        grid[above][x] = 4
    # obstacles
    obstacle_count = int(width * (0.02 + max(0, difficulty) * 0.02))
    for _ in range(max(0, obstacle_count)):
        x = rnd.randint(1, max(1, width - 2))
        y = max(0, ground_y - 1)
        if grid[y][x] == 0:
            grid[y][x] = 2
    return np.array(grid, dtype=np.int64)

# -------------------------
# Dataset
# -------------------------
BASE_W = 80
BASE_H = 24
NUM_CLASSES = 5  # 0..4

class SeedGridDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.samples = []
        for i in range(n_samples):
            seed = f"train-{i}-{random.getrandbits(32)}"
            difficulty = random.randint(1, 6)
            grid = gen_platformer_grid(BASE_W, BASE_H, difficulty, seed)
            self.samples.append((seed, difficulty, grid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seed, difficulty, grid = self.samples[idx]
        # target: torch.LongTensor shape (H,W) with values 0..4
        target = torch.from_numpy(grid).long()
        # produce latent vector deterministically from seed and difficulty
        z = seed_to_int(seed) & ((1<<63)-1)
        rng = np.random.RandomState(z % (2**32))
        z_vec = rng.randn(128).astype(np.float32)
        cond = np.array([float(difficulty)/10.0], dtype=np.float32)  # small conditioning scalar
        inp = np.concatenate([z_vec, cond], axis=0).astype(np.float32)  # shape (129,)
        return torch.from_numpy(inp), target

# -------------------------
# Small decoder model: z(129) -> logits (C,H,W)
# -------------------------
class SmallDecoder(nn.Module):
    def __init__(self, z_dim=129, base_channels=64, out_h=BASE_H, out_w=BASE_W, n_classes=NUM_CLASSES):
        super().__init__()
        self.z_dim = z_dim
        self.out_h = out_h
        self.out_w = out_w
        mid_h = out_h // 4
        mid_w = out_w // 4
        self.fc = nn.Linear(z_dim, base_channels*4*mid_h*mid_w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)
        mid_h = max(1, self.out_h // 4)
        mid_w = max(1, self.out_w // 4)
        x = self.fc(z)
        x = x.view(B, -1, mid_h, mid_w)
        x = self.deconv(x)  # (B, n_classes, H, W)
        return x

# -------------------------
# Training routine
# -------------------------
def train(num_epochs=6, batch_size=8, lr=1e-3, n_samples=800):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeedGridDataset(n_samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = SmallDecoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running = 0.0
        for inp, target in tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inp = inp.to(device)  # (B,129)
            target = target.to(device)  # (B,H,W)
            logits = model(inp)  # (B,C,H,W)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * inp.size(0)
        avg = running / len(ds)
        print(f"Epoch {epoch+1} avg loss: {avg:.6f}")

    # save
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print("Saved model weights to", WEIGHTS_FILE)

if __name__ == "__main__":
    # adjust these hyperparams as you like
    train(num_epochs=6, batch_size=8, lr=1e-3, n_samples=1000)
