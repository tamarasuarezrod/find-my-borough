from rest_framework import serializers
from .models import Borough, CommunityFeature, CommunityRating

class BoroughSerializer(serializers.ModelSerializer):
    image = serializers.ImageField(read_only=True)

    class Meta:
        model = Borough
        fields = ['name', 'slug', 'image', 'norm_rent', 'norm_crime', 'norm_youth']

class CommunityFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommunityFeature
        fields = ['id', 'label', 'description']

class CommunityRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = CommunityRating
        fields = ['feature', 'score']

class CommunityRatingPostSerializer(serializers.Serializer):
    borough = serializers.CharField()
    ratings = serializers.ListField(
        child=serializers.DictField(child=serializers.IntegerField())
    )

    def create(self, validated_data):
        user = self.context['request'].user
        borough_slug = validated_data['borough']
        ratings = validated_data['ratings']

        borough = Borough.objects.get(slug=borough_slug)
        objects = []

        for item in ratings:
            for feature_id, score in item.items():
                feature = CommunityFeature.objects.get(id=feature_id)
                obj, _ = CommunityRating.objects.update_or_create(
                    user=user, borough=borough, feature=feature,
                    defaults={'score': score}
                )
                objects.append(obj)

        return objects
