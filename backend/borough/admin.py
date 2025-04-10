from django.contrib import admin
from .models import Borough

@admin.register(Borough)
class BoroughAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug')
    prepopulated_fields = {"slug": ("name",)}
