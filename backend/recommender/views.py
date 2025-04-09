import os
import sys
import pandas as pd
import torch
import torch.nn as nn

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

csv_path = os.path.join(settings.BASE_DIR, "..", "data", "clean", "borough_features.csv")
df = pd.read_csv(csv_path)

model_path = os.path.join(settings.BASE_DIR, "..", "models", "score_model.pth")
model = ScoreModel()
model.load_state_dict(torch.load(model_path))
model.eval()

def predict_score(user_weights, borough_features):
    with torch.no_grad():
        x = torch.tensor([user_weights + borough_features], dtype=torch.float32)
        return model(x).item()

def get_recommendations(user_preferences, top_n=5):
    weights = [
        user_preferences.get("budget_weight", 0),
        user_preferences.get("safety_weight", 0),
        user_preferences.get("youth_weight", 0),
        user_preferences.get("centrality_weight", 0),
    ]

    recommendations = []
    for _, row in df.iterrows():
        borough = row["borough"]
        borough_features = [
            row["norm_rent"],
            row["norm_safety"],
            row["norm_youth"],
            row["norm_centrality"]
        ]

        score = predict_score(weights, borough_features)

        recommendations.append({
            "borough": borough,
            "score": round(score, 4),
            "norm_rent": round(row["norm_rent"], 2),
            "norm_safety": round(row["norm_safety"], 2),
            "norm_youth": round(row["norm_youth"], 2),
            "norm_centrality": round(row["norm_centrality"], 2),
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]

@api_view(["POST"])
def recommend_boroughs(request):
    user_preferences = request.data
    recommendations = get_recommendations(user_preferences)
    return Response(recommendations)
