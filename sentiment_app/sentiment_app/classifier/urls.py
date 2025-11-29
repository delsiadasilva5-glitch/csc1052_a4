from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_review, name='classify_review'),
    path('summary/', views.model_summary, name='model_summary'),
]
