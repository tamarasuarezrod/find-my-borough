from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    # Auth
    path("api/auth/", include("dj_rest_auth.urls")),
    path("api/auth/registration/", include("dj_rest_auth.registration.urls")),
    # Social auth
    path("api/auth/", include("accounts.urls")),
    # APIs propias
    path("api/", include("recommender.urls")),
    path("api/boroughs/", include("borough.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
