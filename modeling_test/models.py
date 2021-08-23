from django.db import models


class ModelingTest(models.Model):

    user_id = models.CharField(max_length=80)
    test_selection = models.CharField(max_length=80)
    test_reason = models.TextField()
    is_after = models.BooleanField()

    def __str__(self):
        return f"{self.user_id} | {self.test_selection} | " \
               f"{'after' if self.is_after else 'before'}"
