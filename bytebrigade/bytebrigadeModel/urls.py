from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_complaint_view, name='classify_category'),
]
