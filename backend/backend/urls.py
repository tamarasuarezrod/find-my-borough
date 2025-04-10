from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('recommender.urls')),
    path('api/boroughs/', include('borough.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
