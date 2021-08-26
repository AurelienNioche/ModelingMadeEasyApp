import pickle
import json

from django.db import models


class TaskLog(models.Model):

    user_id = models.CharField(max_length=80)
    group_id = models.IntegerField()
    timestamp = models.IntegerField()
    action_type = models.CharField(max_length=80)
    action_var = models.CharField(max_length=80, null=True)
    rec_item = models.CharField(max_length=80, null=True)
    rec_item_cor = models.CharField(max_length=80, null=True)
    included_vars = models.TextField(null=True)

    def __str__(self):
        return f"{self.user_id} | group{self.group_id} | ts={self.timestamp} | {self.action_type} | {self.action_var} | rec={self.rec_item}"


class UserModel(models.Model):

    user_id = models.CharField(max_length=80)
    group_id = models.IntegerField()
    _value = models.BinaryField()

    def set_data(self, data):
        self._value = pickle.dumps(data)

    def get_data(self):
        return pickle.loads(self._value)

    value = property(get_data, set_data)

    def __str__(self):
        return f"{self.user_id} | group{self.group_id}"


class UserData(models.Model):

    user_id = models.CharField(max_length=80)
    group_id = models.IntegerField()
    value = models.TextField()

    def __str__(self):
        return f"{self.user_id} | group{self.group_id}"

