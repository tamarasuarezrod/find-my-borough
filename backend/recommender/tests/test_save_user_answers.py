from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from recommender.models import UserMatchAnswerSet

User = get_user_model()


class SaveUserAnswersTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="testuser", password="pass")
        self.url = "/api/match/answers/"
        self.valid_answers = {
            "budget_weight": 1,
            "safety_weight": 0,
            "youth_weight": 0,
            "centrality_weight": 0,
            "current_situation": "student",
            "stay_duration": "short_term",
        }

    def test_unauthenticated_user_cannot_save_answers(self):
        """Should return 401 if user is not authenticated."""
        response = self.client.post(
            self.url, {"answers": self.valid_answers}, format="json"
        )
        self.assertEqual(response.status_code, 401)

    def test_authenticated_user_can_save_answers(self):
        """Should allow authenticated user to save answer set."""
        self.client.force_authenticate(user=self.user)
        response = self.client.post(
            self.url, {"answers": self.valid_answers}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(UserMatchAnswerSet.objects.count(), 1)

    def test_duplicate_answers_are_not_saved_twice(self):
        """Should not create duplicate answer sets with same content."""
        self.client.force_authenticate(user=self.user)
        self.client.post(self.url, {"answers": self.valid_answers}, format="json")
        self.client.post(self.url, {"answers": self.valid_answers}, format="json")
        self.assertEqual(UserMatchAnswerSet.objects.count(), 1)

    def test_missing_answers_returns_400(self):
        """Should return 400 if answers are missing from payload."""
        self.client.force_authenticate(user=self.user)
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 400)
