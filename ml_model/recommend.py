import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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

model = ScoreModel()
model.load_state_dict(torch.load("models/score_model.pth"))
model.eval()

# Prediction function
def predict_score(budget_weight, safety_weight, youth_weight, centrality_weight,
                  norm_rent, norm_safety, norm_youth, norm_centrality):
    with torch.no_grad():
        x = torch.tensor([[budget_weight, safety_weight, youth_weight, centrality_weight,
                           norm_rent, norm_safety, norm_youth, norm_centrality]], dtype=torch.float32)
        return model(x).item()

df = pd.read_csv("data/clean/borough_features.csv")

def recommend_top_n_boroughs(budget_weight, safety_weight, youth_weight, centrality_weight, top_n=5):
    recommendations = []

    for _, row in df.iterrows():
        score = predict_score(
            budget_weight, safety_weight, youth_weight, centrality_weight,
            row["norm_rent"], row["norm_safety"], row["norm_youth"], row["norm_centrality"]
        )

        recommendations.append({
            "borough": row["borough"],
            "norm_rent": row["norm_rent"],
            "norm_safety": row["norm_safety"],
            "norm_youth": row["norm_youth"],
            "norm_centrality": row["norm_centrality"],
            "score": score
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]

# Example usage
if __name__ == "__main__":
    test_profiles = [
        {"label": "Budget-focused", "budget_weight": 1.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"label": "Safety-focused", "budget_weight": 0.0, "safety_weight": 1.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"label": "Urban enthusiast", "budget_weight": 0.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 1.0},
        {"label": "Balanced", "budget_weight": 0.3, "safety_weight": 0.3, "youth_weight": 0.2, "centrality_weight": 0.2},
        {"label": "Luxury young urban", "budget_weight": 0.0, "safety_weight": 0.1, "youth_weight": 0.6, "centrality_weight": 0.3},
    ]

    for profile in test_profiles:
        print(f"\nTop boroughs for {profile['label']}:")
        top = recommend_top_n_boroughs(
            profile["budget_weight"],
            profile["safety_weight"],
            profile["youth_weight"],
            profile["centrality_weight"]
        )
        for rec in top:
            print(f"{rec['borough']:25s} | score={rec['score']:.4f} | rent={rec['norm_rent']:.2f} | safety={rec['norm_safety']:.2f} | youth={rec['norm_youth']:.2f} | centrality={rec['norm_centrality']:.2f}")
