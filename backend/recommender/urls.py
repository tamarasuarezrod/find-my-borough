from django.urls import path
from .views import recommend_boroughs, SaveUserAnswersView, MatchQuestionListView


urlpatterns = [
    path('recommendations/', recommend_boroughs),
    path('match/answers/', SaveUserAnswersView.as_view(), name='save_user_answers'),
    path('match/questions/', MatchQuestionListView.as_view(), name='match-questions'),
]
