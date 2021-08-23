from django.db import models


class ConsentForm(models.Model):

    user_id = models.CharField(max_length=80)
    perm1 = models.CharField(max_length=80)
    perm2 = models.CharField(max_length=80)
    username = models.TextField()
    date = models.DateField(max_length=80)
    city = models.TextField(max_length=80)

    def __str__(self):
        return f"{self.date} | {self.user_id} | {self.username} | " \
               f"{self.city} | perm1={self.perm1} | perm2={self.perm2}"
