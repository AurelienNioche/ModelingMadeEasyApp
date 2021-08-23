from django.shortcuts import render, redirect, reverse
import uuid

from . forms import Form
from .models import ConsentForm


def index(request):
    if request.method == 'POST':
        form = Form(request.POST)
        if form.is_valid():
            user_id = uuid.uuid4()
            print(f"user_id={user_id}; survey={form.cleaned_data}")
            cf = ConsentForm(user_id=user_id,
                             username=form.cleaned_data["username"],
                             perm1=form.cleaned_data["perm1"],
                             perm2=form.cleaned_data["perm2"],
                             date=form.cleaned_data["date"],
                             city=form.cleaned_data["city"])
            cf.save()
            return redirect(reverse('pre_questionnaire',
                                    kwargs={'user_id': user_id}))
    else:
        form = Form()

    return render(request, 'consent_form/index.html', {'form': form})
