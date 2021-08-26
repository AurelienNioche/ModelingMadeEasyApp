from django.contrib import admin

from . models import TaskLog, UserModel, UserData


admin.site.register(TaskLog)
admin.site.register(UserModel)
admin.site.register(UserData)
