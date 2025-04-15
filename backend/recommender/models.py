import hashlib
import json

from django.contrib.auth import get_user_model
from django.db import models

from borough.models import Borough

User = get_user_model()


class MatchQuestion(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    question_type = models.CharField(
        max_length=20,
        choices=[
            ("choice", "Choice"),
            ("boolean", "Boolean"),
            ("text", "Text"),
            ("number", "Number"),
        ],
    )

    def __str__(self):
        return self.title


class MatchOption(models.Model):
    question = models.ForeignKey(
        MatchQuestion, on_delete=models.CASCADE, related_name="options"
    )
    label = models.CharField(max_length=200)
    value = models.CharField(max_length=100)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"{self.question.id} → {self.label}"


class UserMatchAnswerSet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    answers = models.JSONField()
    hash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "hash")

    def __str__(self):
        return f"{self.user} – {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    @staticmethod
    def calculate_hash(answers: dict) -> str:
        """Deterministic hash based on the answers JSON"""
        return hashlib.sha256(json.dumps(answers, sort_keys=True).encode()).hexdigest()


class UserMatchFeedback(models.Model):
    answer_set = models.ForeignKey(UserMatchAnswerSet, on_delete=models.CASCADE)
    borough = models.ForeignKey(Borough, on_delete=models.CASCADE)
    feedback = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("answer_set", "borough")
