from django import forms
from django.utils.safestring import mark_safe

# CHOICES = [
#     ("aa", "A. y = a1x1 + b "),
#     ("bb", "B. y = a2x2 + b "),
#     ("cc", "C. y = a3x3 + b "),
#     ("dd", "D. y = a1x1 + a2x2 + b "),
#     ("ee", "E. y = a1x1 + a3x3 + b "),
#     ("ff", "F. y = a2x2 + a3x3 + b "),
#     ("gg", "G. y = a1x1 + a2x2 + a3x3 + b ")
# ]


class Form(forms.Form):

    gender = forms.ChoiceField(
        label="1. What gender do you identify as:",
        widget=forms.RadioSelect,
        required=True,
        choices=[
            ("male", "A. Male"),
            ("female", "B. Female"),
            ("other", "C. Other")
        ]
    )

    age = forms.ChoiceField(
        label="2. What is your age group?",
        widget=forms.RadioSelect,
        required=True,
        choices=[
            ("teen", "A. 16-20"),
            ("young", "B. 21-25"),
            ("adult", "C. 26-30"),
            ("other", "D. Other")
        ]
    )

    education = forms.ChoiceField(
        label="3. What is the highest degree or level of education you have completed?",
        widget=forms.RadioSelect,
        required=True,
        choices=[
            ("highschool", "A. High School"),
            ("bachelor", "B. Bachelor's Degree"),
            ("master", "C. Master's Degree"),
            ("phd", "D. PhD Degree"),
            ("other", "E. Other")
        ]
    )

    confidence = forms.ChoiceField(
        label="4. How well do you know about linear regression?",
        widget=forms.RadioSelect,
        required=True,
        choices=[
            ("very", "A. Very well"),
            ("basic", "B. Basic understanding"),
            ("heard", "C. Only heard about it"),
            ("never", "D. Never heard about it"),
        ]
    )

