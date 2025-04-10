from rest_framework import generics
from .models import Borough
from .serializers import BoroughSerializer

class BoroughListView(generics.ListAPIView):
    queryset = Borough.objects.all()
    serializer_class = BoroughSerializer

class BoroughDetailView(generics.RetrieveAPIView):
    queryset = Borough.objects.all()
    serializer_class = BoroughSerializer
    lookup_field = 'slug'
