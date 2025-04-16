import google.auth.exceptions
import requests
from decouple import config
from django.contrib.auth import get_user_model
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView

User = get_user_model()


class GoogleLoginAPIView(APIView):
    def post(self, request):
        provider = request.data.get("provider")
        token_id = request.data.get("token")

        if not token_id:
            return Response(
                {"error": "Token is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        if provider == "google":
            try:
                idinfo = id_token.verify_oauth2_token(
                    token_id,
                    google_requests.Request(),
                    config("GOOGLE_CLIENT_ID"),
                    clock_skew_in_seconds=10,
                )

            except Exception:
                return Response(
                    {"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST
                )

        elif provider == "facebook":
            try:
                fb_app_id = config("FACEBOOK_CLIENT_ID")
                fb_url = f"https://graph.facebook.com/debug_token?input_token={token_id}&access_token={fb_app_id}|{config('FACEBOOK_CLIENT_SECRET')}"

                r = requests.get(fb_url)
                if not r.ok or not r.json().get("data", {}).get("is_valid"):
                    return Response(
                        {"error": "Invalid Facebook token"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                user_info = requests.get(
                    f"https://graph.facebook.com/me?fields=id,name,email&access_token={token_id}"
                ).json()

                email = user_info.get("email")
                name = user_info.get("name")

            except Exception:
                return Response(
                    {"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST
                )

        else:
            return Response(
                {"error": "Invalid provider"}, status=status.HTTP_400_BAD_REQUEST
            )

        email = idinfo.get("email")
        name = idinfo.get("name")

        if not email:
            return Response(
                {"error": "Email not found in token"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user, _ = User.objects.get_or_create(
            email=email,
            defaults={
                "username": email,
                "first_name": name or email.split("@")[0],
            },
        )

        # Generar JWT
        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "email": user.email,
                "name": user.first_name,
            }
        )


@method_decorator(csrf_exempt, name="dispatch")
class CustomTokenRefreshView(TokenRefreshView):
    pass
