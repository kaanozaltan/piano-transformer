import os

import torch
import wandb
from dotenv import load_dotenv
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()

# ======================
# W&B SETUP
# ======================

PROJECT_NAME = "piano-transformer"
ENTITY_NAME = "jonathanlehmkuhl-rwth-aachen-university"  # e.g. "jonathanlehmkuhl"

wandb.login()
wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, name="test-run")

# ======================
# DUMMY MODEL + DATA
# ======================

model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Dummy data
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ======================
# TRAINING LOOP
# ======================

for epoch in range(3):
    epoch_loss = 0
    for xb, yb in dataloader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

# Finish run
wandb.finish()
