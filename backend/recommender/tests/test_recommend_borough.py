from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from borough.models import Borough
from recommender.models import UserMatchAnswerSet, UserMatchFeedback

User = get_user_model()


class RecommendBoroughsTests(TestCase):
    def setUp(self):
        self.client = APIClient()

        self.borough = Borough.objects.get(slug="camden")
        self.borough.norm_rent = 0.8
        self.borough.norm_crime = 0.6
        self.borough.norm_youth = 0.7
        self.borough.norm_centrality = 0.9
        self.borough.save()

        islington = Borough.objects.get(slug="islington")
        islington.norm_rent = 0.6
        islington.norm_crime = 0.7
        islington.norm_youth = 0.5
        islington.norm_centrality = 0.8
        islington.save()

    def test_recommend_boroughs_returns_results(self):
        """Should return a list of recommended boroughs based on user preferences."""
        answers = {
            "budget_weight": 1,
            "safety_weight": 0,
            "youth_weight": 0.5,
            "centrality_weight": 0,
            "current_situation": "student",
            "stay_duration": "short_term",
        }
        response = self.client.post("/api/recommendations/", answers, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()), 0)
        self.assertIn("borough", response.json()[0])
        self.assertIn("score", response.json()[0])
