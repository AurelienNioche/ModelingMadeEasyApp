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

urlpatterns = [
    path('group0/', include('group0.urls')),
    path('group1/', include('group1.urls')),
    path('group2/', include('group2.urls')),
    path('constent_form/', include('consent_form.urls')),
    path('end/', include('end.urls')),
    path('exp_tutorial/', include('exp_tutorial.urls')),
    path('mme_intro/', include('mme_intro.urls')),
    path('modeling_test/', include('modeling_test.urls')),
    path('modeling_test_after/', include('modeling_test_after.urls')),
    path('pre_questionnaire/', include('pre_questionnaire.urls')),
    path('survey/', include('survey.urls')),
    path('admin/', admin.site.urls),
]
