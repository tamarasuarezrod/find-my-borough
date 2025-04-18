from django.urls import path

from .views import (
    MatchQuestionListView,
    SaveUserAnswersOnlyView,
    SaveUserFeedbackView,
    recommend_boroughs,
)

urlpatterns = [
    path("recommendations/", recommend_boroughs, name="recommend-boroughs"),
    path("match/questions/", MatchQuestionListView.as_view(), name="match-questions"),
    path("match/answers/", SaveUserAnswersOnlyView.as_view(), name="save-user-answers"),
    path("match/feedback/", SaveUserFeedbackView.as_view(), name="save-user-feedback"),
]
