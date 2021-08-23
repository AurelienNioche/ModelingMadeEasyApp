from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='survey'),
    path('<str:user_id>/', views.index, name='survey'),
]