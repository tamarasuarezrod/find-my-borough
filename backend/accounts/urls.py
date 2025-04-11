from django.urls import path
from .views import GoogleLoginAPIView
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('google/', GoogleLoginAPIView.as_view(), name='google_login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
