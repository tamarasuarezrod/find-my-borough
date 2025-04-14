from django.contrib import admin

from .models import UserMatchAnswerSet, UserMatchFeedback


@admin.register(UserMatchAnswerSet)
class UserMatchAnswerSetAdmin(admin.ModelAdmin):
    list_display = ("user", "hash", "created_at")
    search_fields = ("user__email", "hash")
    readonly_fields = ("hash", "created_at")
    list_filter = ("created_at",)


@admin.register(UserMatchFeedback)
class UserMatchFeedbackAdmin(admin.ModelAdmin):
    list_display = ("answer_set", "borough", "feedback", "created_at")
    list_filter = ("feedback", "created_at", "borough")
    search_fields = ("answer_set__user__email", "borough__name")
