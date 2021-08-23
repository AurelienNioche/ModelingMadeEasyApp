from django.db import models


class Survey(models.Model):

    user_id = models.CharField(max_length=80)
    q1 = models.TextField()
    q2 = models.TextField()
    q3 = models.TextField()
    q4 = models.TextField()
    q5 = models.TextField()

    def display_q(self, q_idx, max_char=20):
        q = getattr(self, f"q{q_idx}")
        len_q = len(q)
        text = q[:min(len_q, max_char)]
        if len_q > max_char:
            text += "..."
        return text

    def __str__(self):
        return f"{self.user_id} | " \
               f"{self.display_q(1)} | " \
               f"{self.display_q(2)} | " \
               f"{self.display_q(3)} | " \
               f"{self.display_q(4)} | " \
               f"{self.display_q(5)} | " \

