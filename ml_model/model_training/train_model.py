import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Load training data
df_train = pd.read_csv("data/clean/synthetic_training_data.csv")

# Select input features and target
features = df_train[[
    "budget_weight", "safety_weight", "youth_weight", "centrality_weight",
    "norm_rent", "norm_safety", "norm_youth", "norm_centrality"
]].values.astype(np.float32)
targets = df_train["score"].values.astype(np.float32).reshape(-1, 1)

# Dataset definition
class ScoreDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model definition
class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model():
    dataset = ScoreDataset(features, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ScoreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        total_loss = 0
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/score_model.pth")

if __name__ == "__main__":
    train_model()
