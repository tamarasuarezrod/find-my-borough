from rest_framework import generics
from .models import Borough
from .serializers import BoroughSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.db.models import Avg

from .models import CommunityFeature, CommunityRating
from .serializers import (
    CommunityFeatureSerializer,
    CommunityRatingPostSerializer,
)
from borough.models import Borough

class BoroughListView(generics.ListAPIView):
    queryset = Borough.objects.all().order_by('-norm_centrality')
    serializer_class = BoroughSerializer

class BoroughDetailView(generics.RetrieveAPIView):
    queryset = Borough.objects.all()
    serializer_class = BoroughSerializer
    lookup_field = 'slug'


class CommunityFeatureListView(APIView):
    def get(self, request):
        features = CommunityFeature.objects.all()
        serializer = CommunityFeatureSerializer(features, many=True)
        return Response(serializer.data)

class CommunityRatingSubmitView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = CommunityRatingPostSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response({'status': 'Ratings saved'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CommunityScoresByBoroughView(APIView):
    def get(self, request, slug):
        borough = Borough.objects.get(slug=slug)
        features = CommunityFeature.objects.all()

        data = []
        for feature in features:
            avg_score = CommunityRating.objects.filter(borough=borough, feature=feature).aggregate(avg=Avg('score'))['avg']
            if avg_score is not None:
                avg_score = round(avg_score)
            data.append({
                'feature': feature.id,
                'label': feature.label,
                'score': avg_score or 0,
            })

        return Response(data)