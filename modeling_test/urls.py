from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='modeling_test'),
    path('<str:user_id>/<int:after>/', views.index, name='modeling_test'),
]