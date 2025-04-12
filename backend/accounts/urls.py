from django.urls import path
from .views import GoogleLoginAPIView
from rest_framework_simplejwt.views import TokenRefreshView
from .views import CustomTokenRefreshView

urlpatterns = [
    path('google/', GoogleLoginAPIView.as_view(), name='google_login'),
    path('token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
]
