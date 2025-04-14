from django.urls import path

from .views import (
    BoroughDetailView,
    BoroughListView,
    CommunityRatingSubmitView,
    CommunityScoresByBoroughView,
)

urlpatterns = [
    path("", BoroughListView.as_view(), name="borough-list"),
    path("<slug:slug>/", BoroughDetailView.as_view(), name="borough-detail"),
    path("community/submit/", CommunityRatingSubmitView.as_view()),
    path("community/scores/<slug:slug>/", CommunityScoresByBoroughView.as_view()),
]
