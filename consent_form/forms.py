from django import forms
import datetime

PERM1_CHOICES = [
    ("agree", "I agree to releasing anonymized extracts from my data."),
    ("agree_informed",   "I agree to releasing anonymized extracts from my data only if I am informed "
                        "about the research groups in question. I have been told what that subset will be."),
    ("disagree", "I do not agree to releasing extracts from my data.")]

PERM2_CHOICES = [
    ("agree", "I agree to anonymized quotation/publication of extracts from my interview/questionnaire."),
    ("disagree", "I do not agree to quotation/publication of extracts from my interview/questionnaire.")
]


class Form(forms.Form):
    perm1 = forms.ChoiceField(
        label="",
        required=True,
        widget=forms.RadioSelect,
        choices=PERM1_CHOICES)

    perm2 = forms.ChoiceField(
        label="",
        required=True,
        widget=forms.RadioSelect,
        choices=PERM2_CHOICES)

    username = forms.CharField(
        label="",
        required=True)

    date = forms.CharField(
        widget=forms.DateInput(
            format=('%Y-%m-%d'),
            attrs={
                   'type': 'date',
                   }),
        initial=datetime.datetime.today
    )

    city = forms.CharField(initial="Helsinki")

