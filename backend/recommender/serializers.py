from rest_framework import serializers
from .models import MatchQuestion, MatchOption

class MatchOptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatchOption
        fields = ['label', 'value']

class MatchQuestionSerializer(serializers.ModelSerializer):
    options = MatchOptionSerializer(many=True)

    class Meta:
        model = MatchQuestion
        fields = ['id', 'title', 'description', 'question_type', 'options']
