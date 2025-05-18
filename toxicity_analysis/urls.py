from django.urls import path
from . import views

urlpatterns = [
    path("", views.analyze_toxicity, name="analyze_toxicity"),
]