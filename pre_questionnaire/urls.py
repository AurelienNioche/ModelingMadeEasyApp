from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='pre_questionnaire'),
    path('<str:user_id>/', views.index, name='pre_questionnaire'),
]