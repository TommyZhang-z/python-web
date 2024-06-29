from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("ai_app/", include("ai_app.urls")),
    path("admin/", admin.site.urls),
]
