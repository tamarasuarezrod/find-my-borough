from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from recommender.models import MatchOption, MatchQuestion


class MatchQuestionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.question = MatchQuestion.objects.create(
            id="budget_weight",
            title="💰 Rent prices",
            description="How sensitive are you to rent prices?",
            question_type="choice",
        )
        MatchOption.objects.create(
            question=self.question, label="Very sensitive", value="1", order=0
        )
        MatchOption.objects.create(
            question=self.question, label="Moderately", value="0.5", order=1
        )
        MatchOption.objects.create(
            question=self.question, label="Not really", value="0", order=2
        )

    def test_fetch_match_questions(self):
        """Should return a list of match questions with their options."""
        response = self.client.get("/api/match/questions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        question = response.json()[0]
        self.assertEqual(question["id"], "budget_weight")
        self.assertEqual(len(question["options"]), 3)
