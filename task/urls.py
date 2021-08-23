from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='task'),
    path('<str:user_id>/<int:group_id>/', views.index, name='task'),
    path('<str:user_id>/<int:group_id>/user_action', views.user_action, name="user_action"),
]