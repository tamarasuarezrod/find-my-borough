from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class MatchQuestion(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    question_type = models.CharField(max_length=20, choices=[
        ('choice', 'Choice'),
        ('boolean', 'Boolean'),
        ('text', 'Text'),
        ('number', 'Number'),
    ])

    def __str__(self):
        return self.title


class MatchOption(models.Model):
    question = models.ForeignKey(MatchQuestion, on_delete=models.CASCADE, related_name='options')
    label = models.CharField(max_length=200)
    value = models.CharField(max_length=100)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.question.id} → {self.label}"


class UserMatchAnswer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.ForeignKey(MatchQuestion, on_delete=models.CASCADE)
    selected_value = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'question', 'created_at')
