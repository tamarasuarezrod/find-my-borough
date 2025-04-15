from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

User = get_user_model()


class TokenRefreshTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="testuser", password="pass")
        self.refresh = RefreshToken.for_user(self.user)
        self.url = "/api/auth/token/refresh/"

    def test_refresh_token_returns_new_access(self):
        """Should return a new access token given a valid refresh token"""
        response = self.client.post(
            self.url, {"refresh": str(self.refresh)}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.json())

    def test_invalid_refresh_token_returns_401(self):
        """Should return 401 if the refresh token is invalid"""
        response = self.client.post(
            self.url, {"refresh": "not-a-real-token"}, format="json"
        )
        self.assertEqual(response.status_code, 401)
