import os
import torch
import torch.nn as nn

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from .serializers import MatchAnswerSerializer
from borough.models import Borough
from .models import MatchQuestion
from .serializers import MatchQuestionSerializer

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

def get_recommendations(user_preferences, top_n=4):
    weights = [
        float(user_preferences.get("budget_weight", 0) or 0),
        float(user_preferences.get("safety_weight", 0) or 0),
        float(user_preferences.get("youth_weight", 0) or 0),
        float(user_preferences.get("centrality_weight", 0) or 0),
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

class SaveUserAnswersView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = MatchAnswerSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response({"status": "answers saved"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MatchQuestionListView(APIView):
    def get(self, request):
        questions = MatchQuestion.objects.prefetch_related('options').all()
        serializer = MatchQuestionSerializer(questions, many=True)
        return Response(serializer.data)
