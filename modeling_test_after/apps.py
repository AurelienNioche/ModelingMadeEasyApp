from django.apps import AppConfig


class ModelingTestAfterConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'modeling_test_after'
