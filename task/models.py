from django.db import models


class TaskLog(models.Model):

    user_id = models.CharField(max_length=80)
    group_id = models.IntegerField()
    timestamp = models.IntegerField()
    action_type = models.CharField(max_length=80)
    parameters = models.CharField(max_length=80, null=True)
    rec_item = models.CharField(max_length=80, null=True)

    def __str__(self):
        return f"{self.user_id} | group{self.group_id} | ts={self.timestamp} | {self.action_type} | {self.parameters} | rec={self.rec_item}"
