import os
import torch
import torch.nn as nn

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

from borough.models import Borough  # 👈 Importamos el modelo

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

# Cargar modelo
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
    boroughs = Borough.objects.exclude(norm_rent__isnull=True).exclude(norm_crime__isnull=True).exclude(norm_youth__isnull=True)

    for b in boroughs:
        borough_features = [
            b.norm_rent,
            b.norm_crime,
            b.norm_youth,
            b.norm_centrality,
        ]

        score = predict_score(weights, borough_features)

        recommendations.append({
            "borough": b.name.lower(),
            "score": round(score, 4),
            "norm_rent": round(b.norm_rent, 2),
            "norm_crime": round(b.norm_crime, 2),
            "norm_youth": round(b.norm_youth, 2),
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]

@api_view(["POST"])
def recommend_boroughs(request):
    user_preferences = request.data
    recommendations = get_recommendations(user_preferences)
    return Response(recommendations)
