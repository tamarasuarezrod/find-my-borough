from django.contrib import admin
from .models import MatchQuestion, MatchOption, UserMatchAnswer

class MatchOptionInline(admin.TabularInline):
    model = MatchOption
    extra = 1

@admin.register(MatchQuestion)
class MatchQuestionAdmin(admin.ModelAdmin):
    inlines = [MatchOptionInline]
    list_display = ['id', 'title', 'question_type']

admin.site.register(UserMatchAnswer)
