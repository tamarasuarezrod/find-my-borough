from django.contrib import admin

from .models import Borough, CommunityRating


@admin.register(Borough)
class BoroughAdmin(admin.ModelAdmin):
    list_display = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(CommunityRating)
class CommunityRatingAdmin(admin.ModelAdmin):
    list_display = ("user", "borough", "feature", "score", "created_at")
    list_filter = ("borough", "feature", "score")
    search_fields = ("user__email", "borough__name", "feature__label")
