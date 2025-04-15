from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

User = get_user_model()


class GoogleLoginTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/api/auth/google/"

    def test_missing_token_returns_400(self):
        """Should return 400 if no token is provided"""
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Token is required")

    @patch("accounts.views.id_token.verify_oauth2_token")
    def test_invalid_token_returns_400(self, mock_verify):
        """Should return 400 if token verification fails"""
        mock_verify.side_effect = Exception("Invalid token")
        response = self.client.post(self.url, {"token": "fake-token"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid token")

    @patch("accounts.views.id_token.verify_oauth2_token")
    def test_valid_token_creates_user_and_returns_tokens(self, mock_verify):
        """Should create user if doesn't exist and return tokens"""
        mock_verify.return_value = {
            "email": "testuser@example.com",
            "name": "Test User",
        }

        response = self.client.post(self.url, {"token": "valid-token"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.json())
        self.assertIn("refresh", response.json())
        self.assertTrue(User.objects.filter(email="testuser@example.com").exists())
