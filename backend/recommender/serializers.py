from rest_framework import serializers
from .models import MatchQuestion, MatchOption, UserMatchAnswer

class MatchAnswerSerializer(serializers.Serializer):
    answers = serializers.DictField(child=serializers.CharField())

    def create(self, validated_data):
        user = self.context['request'].user
        answers = validated_data['answers']

        created_objects = []
        for qid, val in answers.items():
            try:
                question = MatchQuestion.objects.get(id=qid)
            except MatchQuestion.DoesNotExist:
                continue

            UserMatchAnswer.objects.create(
                user=user,
                question=question,
                selected_value=str(val),
            )
        return created_objects

class MatchOptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatchOption
        fields = ['label', 'value']

class MatchQuestionSerializer(serializers.ModelSerializer):
    options = MatchOptionSerializer(many=True)

    class Meta:
        model = MatchQuestion
        fields = ['id', 'title', 'description', 'question_type', 'options']
