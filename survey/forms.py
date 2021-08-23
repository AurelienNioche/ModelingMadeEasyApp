from django import forms


class Form(forms.Form):

    TEXTAREA_ATTRS = {'class': 'survey_text',
                      'rows': 2,
                      'cols': 80}

    q1 = forms.CharField(
        widget=forms.Textarea(attrs=TEXTAREA_ATTRS,),
        label="1. How do you feel about the whole experiment",
        required=True)

    q2 = forms.CharField(
        widget=forms.Textarea(attrs=TEXTAREA_ATTRS,),
        label="2. How do you feel about the first modeling test",
        required=True)

    q3 = forms.CharField(
        widget=forms.Textarea(attrs=TEXTAREA_ATTRS,),
        label="3. How do you feel about the tutorial session",
        required=True)

    q4 = forms.CharField(
        widget=forms.Textarea(attrs=TEXTAREA_ATTRS,),
        label="4. How do you feel about the modeling test after the tutorial",
        required=True)

    q5 = forms.CharField(
        widget=forms.Textarea(attrs=TEXTAREA_ATTRS,),
        label="5. Any other comments",
        required=True)
