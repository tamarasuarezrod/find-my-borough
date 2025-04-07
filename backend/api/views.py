import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ml_model.predict import predict

@api_view(['POST'])
def recommend_boroughs(request):
    user_preferences = request.data
    recommendations = predict(user_preferences)
    return Response(recommendations)
