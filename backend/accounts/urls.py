from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import CustomTokenRefreshView, GoogleLoginAPIView

urlpatterns = [
    path("google/", GoogleLoginAPIView.as_view(), name="google_login"),
    path("token/refresh/", CustomTokenRefreshView.as_view(), name="token_refresh"),
]
