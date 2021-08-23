from django import forms
from django.utils.safestring import mark_safe


class Form(forms.Form):

    test_selection = forms.MultipleChoiceField(
        label="",
        required=True,
        widget=forms.CheckboxSelectMultiple,
        choices=[("A", "A. y = a1x1 + b "),
                 ("B", "B. y = a2x2 + b "),
                 ("C", "C. y = a3x3 + b "),
                 ("D", "D. y = a1x1 + a2x2 + b "),
                 ("E", "E. y = a1x1 + a3x3 + b "),
                 ("F", "F. y = a2x2 + a3x3 + b "),
                 ("G", "G. y = a1x1 + a2x2 + a3x3 + b ")
        ])
    test_reason = forms.CharField(
        widget=forms.Textarea(
            attrs={'class': 'reason_text',
                   'rows': 2,
                   'cols': 50}),
        label='Reason',
        label_suffix=mark_safe(':<br>'),
        required=True)
