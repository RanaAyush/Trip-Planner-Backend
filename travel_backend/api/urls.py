from django.urls import path
from .views import process_query, get_place_image

urlpatterns = [
    path('extract/', process_query),
    path('place-image/', get_place_image, name='place-image')
]