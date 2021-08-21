from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='group0'),
    path('<str:user_id>/<int:group_id>/', views.index, name='group0'),
    path('<str:user_id>/validate', views.validate, name="validate"),
]