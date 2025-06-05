from django.urls import path
from .views import process_query

urlpatterns = [
    path('extract/', process_query),
]