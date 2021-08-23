from django.urls import path

from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index, name='group1'),
    path('<str:user_id>/', views.index, name='group1'),
    path('<str:user_id>/validate', views.validate, name="group1_validate"),
]