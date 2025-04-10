from django.urls import path
from .views import BoroughListView, BoroughDetailView

urlpatterns = [
    path('', BoroughListView.as_view(), name='borough-list'),
    path('<slug:slug>/', BoroughDetailView.as_view(), name='borough-detail'),
]
