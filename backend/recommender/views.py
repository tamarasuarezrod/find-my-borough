import os

import torch
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from borough.models import Borough
from ml_model.models.load_latest_model import load_latest_model

from .models import MatchQuestion, UserMatchAnswerSet, UserMatchFeedback
from .serializers import MatchQuestionSerializer

model = load_latest_model()

CURRENT_SITUATION_OPTIONS = ["student", "young_professional", "professional", "other"]
STAY_DURATION_OPTIONS = ["short_term", "medium_term", "long_term"]


def encode_one_hot(value, options):
    return [1.0 if value == opt else 0.0 for opt in options]


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

    situation_encoding = encode_one_hot(
        user_preferences.get("current_situation", ""), CURRENT_SITUATION_OPTIONS
    )
    stay_encoding = encode_one_hot(
        user_preferences.get("stay_duration", ""), STAY_DURATION_OPTIONS
    )

    full_input = weights + situation_encoding + stay_encoding

    recommendations = []
    boroughs = (
        Borough.objects.exclude(norm_rent__isnull=True)
        .exclude(norm_crime__isnull=True)
        .exclude(norm_youth__isnull=True)
    )

    for b in boroughs:
        borough_features = [
            b.norm_rent,
            b.norm_crime,
            b.norm_youth,
            b.norm_centrality,
        ]

        score = predict_score(full_input, borough_features)

        recommendations.append(
            {
                "borough": b.name.lower(),
                "score": round(score, 4),
                "norm_rent": round(b.norm_rent, 2),
                "norm_crime": round(b.norm_crime, 2),
                "norm_youth": round(b.norm_youth, 2),
                "norm_centrality": round(b.norm_centrality, 2),
            }
        )

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]


@api_view(["POST"])
def recommend_boroughs(request):
    user_preferences = request.data
    recommendations = get_recommendations(user_preferences)
    return Response(recommendations)


class SaveUserAnswersOnlyView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        answers = request.data.get("answers")

        if not answers:
            return Response({"error": "Missing answers"}, status=400)

        answer_hash = UserMatchAnswerSet.calculate_hash(answers)
        UserMatchAnswerSet.objects.get_or_create(
            user=user, hash=answer_hash, defaults={"answers": answers}
        )

        return Response({"status": "answers saved"}, status=201)


class SaveUserFeedbackView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        borough_slug = request.data.get("borough")
        feedback_value = request.data.get("feedback")

        if not borough_slug or feedback_value is None:
            return Response({"error": "Missing data"}, status=400)

        try:
            borough = Borough.objects.get(slug=borough_slug)
        except Borough.DoesNotExist:
            return Response({"error": "Borough not found"}, status=404)

        latest_answer_set = (
            UserMatchAnswerSet.objects.filter(user=user).order_by("-created_at").first()
        )

        if not latest_answer_set:
            return Response({"error": "No answer set found"}, status=404)

        feedback, created = UserMatchFeedback.objects.get_or_create(
            answer_set=latest_answer_set,
            borough=borough,
            defaults={"feedback": feedback_value},
        )

        if not created:
            if feedback.feedback == feedback_value:
                return Response({"status": "unchanged"}, status=200)
            feedback.feedback = feedback_value
            feedback.save()
            return Response({"status": "updated"}, status=200)

        return Response({"status": "created"}, status=201)


class MatchQuestionListView(APIView):
    def get(self, request):
        questions = MatchQuestion.objects.prefetch_related("options").all()
        serializer = MatchQuestionSerializer(questions, many=True)
        return Response(serializer.data)
