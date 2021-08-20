from django import forms
from django.utils.safestring import mark_safe

CHOICES = [
    ("aa", "A. y = a1x1 + b "),
    ("bb", "B. y = a2x2 + b "),
    ("cc", "C. y = a3x3 + b "),
    ("dd", "D. y = a1x1 + a2x2 + b "),
    ("ee", "E. y = a1x1 + a3x3 + b "),
    ("ff", "F. y = a2x2 + a3x3 + b "),
    ("gg", "G. y = a1x1 + a2x2 + a3x3 + b ")
]


class Form(forms.Form):

    test_selection = forms.MultipleChoiceField(
        label="",
        required=True,
        widget=forms.CheckboxSelectMultiple,
        choices=CHOICES)
    test_reason = forms.CharField(
        widget=forms.Textarea,
        label=mark_safe("Reason: <br/>"),
        label_suffix="",
        required=True)
