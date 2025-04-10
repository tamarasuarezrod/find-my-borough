from rest_framework import serializers
from .models import Borough

class BoroughSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(read_only=True)

    class Meta:
        model = Borough
        fields = ['name', 'slug', 'image', 'norm_rent', 'norm_crime', 'norm_youth']

