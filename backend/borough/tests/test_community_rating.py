from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from rest_framework.test import APIClient

from borough.models import Borough, CommunityFeature, CommunityRating

User = get_user_model()


class CommunityRatingTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user1 = User.objects.create_user(username="user1", password="pass")
        self.user2 = User.objects.create_user(username="user2", password="pass")
        self.borough = Borough.objects.first()
        self.feature = CommunityFeature.objects.create(id="safety", label="Safety")

    def test_anonymous_user_cannot_vote(self):
        """Anonymous users cannot submit ratings (should return 401)."""
        response = self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 3}]},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 401)

    def test_vote_with_zero_score_does_not_create_rating(self):
        """Ratings with a score of 0 should be ignored and not saved."""
        self.client.force_authenticate(user=self.user1)
        response = self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 0}]},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(CommunityRating.objects.count(), 0)

    def test_authenticated_user_can_vote(self):
        """Authenticated users can successfully vote and their rating is saved."""
        self.client.force_authenticate(user=self.user1)
        response = self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 4}]},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        rating = CommunityRating.objects.get()
        self.assertEqual(rating.user, self.user1)
        self.assertEqual(rating.borough, self.borough)
        self.assertEqual(rating.feature, self.feature)
        self.assertEqual(rating.score, 4)

    def test_community_score_is_average(self):
        """Community score endpoint returns the average of all user ratings."""
        self.client.force_authenticate(user=self.user1)
        self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 2}]},
            content_type="application/json",
        )
        self.client.force_authenticate(user=self.user2)
        self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 4}]},
            content_type="application/json",
        )

        response = self.client.get(
            f"/api/boroughs/community/scores/{self.borough.slug}/"
        )
        self.assertEqual(response.status_code, 200)
        data = [item for item in response.json() if item["feature"] == self.feature.id][
            0
        ]
        self.assertEqual(data["score"], 3)

    def test_user_update_vote_only_once(self):
        """Submitting multiple ratings from the same user for the same feature updates the existing one, doesn't create duplicates."""
        self.client.force_authenticate(user=self.user1)
        for _ in range(3):
            self.client.post(
                "/api/boroughs/community/submit/",
                {"borough": self.borough.slug, "ratings": [{self.feature.id: 5}]},
                content_type="application/json",
            )

        ratings = CommunityRating.objects.filter(
            user=self.user1, borough=self.borough, feature=self.feature
        )
        self.assertEqual(ratings.count(), 1)
        self.assertEqual(ratings.first().score, 5)

    def test_user_vote_is_updated_on_change(self):
        """If the user changes their vote, the score is updated accordingly."""
        self.client.force_authenticate(user=self.user1)

        self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 3}]},
            content_type="application/json",
        )

        self.client.post(
            "/api/boroughs/community/submit/",
            {"borough": self.borough.slug, "ratings": [{self.feature.id: 1}]},
            content_type="application/json",
        )

        ratings = CommunityRating.objects.filter(
            user=self.user1, borough=self.borough, feature=self.feature
        )
        self.assertEqual(ratings.count(), 1)
        self.assertEqual(ratings.first().score, 1)
