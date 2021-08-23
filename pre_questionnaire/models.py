from django.db import models


class PreQuestionnaire(models.Model):

    user_id = models.CharField(max_length=80)
    gender = models.CharField(max_length=80)
    age = models.CharField(max_length=80)
    education = models.CharField(max_length=80)
    confidence = models.CharField(max_length=80)

    def __str__(self):
        return f"{self.user_id} | {self.gender} | {self.age} | " \
               f"{self.education} | {self.confidence}"
