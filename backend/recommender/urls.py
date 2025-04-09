from django.urls import path
from .views import recommend_boroughs

urlpatterns = [
    path('recommendations/', recommend_boroughs),
]
