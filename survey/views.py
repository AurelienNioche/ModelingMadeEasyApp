from django.shortcuts import render, redirect
from django.urls import reverse

from .forms import Form
from .models import Survey


def index(request, user_id="user_test"):

    if request.method == 'POST':
        form = Form(request.POST)
        if form.is_valid():
            print(f"user_id={user_id}; survey={form.cleaned_data}")
            s = Survey(user_id=user_id,
                       q1=form.cleaned_data["q1"],
                       q2=form.cleaned_data["q2"],
                       q3=form.cleaned_data["q3"],
                       q4=form.cleaned_data["q4"],
                       q5=form.cleaned_data["q5"])
            s.save()
            return redirect(reverse('end'))

    else:
        form = Form()

    return render(request, 'survey/index.html', {'form': form,
                                                 'user_id': user_id})
