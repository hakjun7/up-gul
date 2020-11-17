from django.urls import path
from . import views

app_name = 'AI'

urlpatterns = [
    path('emotions/', views.emotions, name='emotions'),
    path('emotions/<str:user_name>/', views.emotions_api, name='emotions'),
]
