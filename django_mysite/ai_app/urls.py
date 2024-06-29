from django.urls import path

from . import views

urlpatterns = [
    path("deep_learning/", views.deep_learning, name="deep_learning"),
    path("machine_learning/", views.machine_learning, name="machine_learning"),
]
