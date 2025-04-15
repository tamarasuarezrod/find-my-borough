from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from borough.models import Borough
from recommender.models import UserMatchAnswerSet, UserMatchFeedback

User = get_user_model()


class SaveUserFeedbackTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="testuser", password="pass")
        self.borough, _ = Borough.objects.get_or_create(
            slug="camden", defaults={"name": "Camden"}
        )
        self.url = "/api/match/feedback/"

        self.answers = {
            "budget_weight": 1,
            "safety_weight": 0,
            "youth_weight": 0,
            "centrality_weight": 0,
            "current_situation": "student",
            "stay_duration": "short_term",
        }

        # Save answers to create a valid answer set
        self.client.force_authenticate(user=self.user)
        self.client.post(
            "/api/match/answers/", {"answers": self.answers}, format="json"
        )

    def test_unauthenticated_user_cannot_send_feedback(self):
        """Should return 401 if user is not authenticated."""
        self.client.logout()
        response = self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        self.assertEqual(response.status_code, 401)

    def test_user_can_submit_feedback(self):
        """Should allow submitting feedback if answer set exists."""
        response = self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(UserMatchFeedback.objects.count(), 1)

    def test_duplicate_feedback_is_updated_not_duplicated(self):
        """Should update feedback if already exists instead of duplicating."""
        self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        self.client.post(
            self.url, {"borough": "camden", "feedback": False}, format="json"
        )
        self.assertEqual(UserMatchFeedback.objects.count(), 1)
        self.assertFalse(UserMatchFeedback.objects.first().feedback)

    def test_identical_feedback_returns_unchanged(self):
        """Should return status 200 unchanged if feedback is identical."""
        self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        response = self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unchanged")

    def test_missing_fields_returns_400(self):
        """Should return 400 if borough or feedback is missing."""
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 400)

    def test_invalid_borough_returns_404(self):
        """Should return 404 if borough slug is invalid."""
        response = self.client.post(
            self.url, {"borough": "invalid", "feedback": True}, format="json"
        )
        self.assertEqual(response.status_code, 404)

    def test_no_answers_returns_404(self):
        """Should return 404 if user has not saved any answer set."""
        user2 = User.objects.create_user(username="another", password="pass")
        self.client.force_authenticate(user=user2)
        response = self.client.post(
            self.url, {"borough": "camden", "feedback": True}, format="json"
        )
        self.assertEqual(response.status_code, 404)
