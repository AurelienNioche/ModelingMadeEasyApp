"""ModelingMadeEasyApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView

urlpatterns = [
    path('', include('intro.urls')),
    path('task/', include('task.urls')),
    path('group0/', include('group0.urls')),
    path('group1/', include('group1.urls')),
    path('group2/', include('group2.urls')),
    path('consent_form/', include('consent_form.urls')),
    path('end/', include('end.urls')),
    path('exp_tutorial/', include('exp_tutorial.urls')),
    path('modeling_test/', include('modeling_test.urls')),
    path('pre_questionnaire/', include('pre_questionnaire.urls')),
    path('survey/', include('survey.urls')),
    path('admin/', admin.site.urls),
    path('favicon.ico', RedirectView.as_view(
        url=staticfiles_storage.url('images/favicon.ico')))
]
