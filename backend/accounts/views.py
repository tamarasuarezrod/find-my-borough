import requests
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token

User = get_user_model()

@api_view(['POST'])
def google_login(request):
    token_id = request.data.get('token')

    if not token_id:
        return Response({'error': 'Token is required'}, status=status.HTTP_400_BAD_REQUEST)

    google_response = requests.get(
        f'https://oauth2.googleapis.com/tokeninfo?id_token={token_id}'
    )

    if google_response.status_code != 200:
        return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)

    data = google_response.json()
    email = data.get('email')
    name = data.get('name')

    if not email:
        return Response({'error': 'Email not found in token'}, status=status.HTTP_400_BAD_REQUEST)

    user, created = User.objects.get_or_create(email=email, defaults={'username': email, 'first_name': name})
    token, _ = Token.objects.get_or_create(user=user)

    return Response({'token': token.key, 'email': user.email, 'name': user.first_name})
