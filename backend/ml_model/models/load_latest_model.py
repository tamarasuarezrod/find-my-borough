import json
import os

import torch
import torch.nn as nn

from .model_architecture import ScoreModel


def load_latest_model():
    """
    Loads the most recent model based on the registry file in the current folder.
    """
    base_dir = os.path.dirname(__file__)
    registry_path = os.path.join(base_dir, "model_registry.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError("No model_registry.json found.")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    if not registry:
        raise ValueError("Model registry is empty.")

    latest = registry[-1]
    model_path = os.path.join(base_dir, latest["filename"])
    params = latest.get("hyperparameters", {})

    hidden_dim_1 = int(params.get("hidden_dim_1", 16))
    hidden_dim_2 = int(params.get("hidden_dim_2", 8))

    print(f"✅ Loading model: {latest['filename']}")

    model = ScoreModel(hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
