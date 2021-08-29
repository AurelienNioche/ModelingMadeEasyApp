from django.shortcuts import render, reverse, redirect

from .forms import Form
from .models import PreQuestionnaire


def index(request, user_id="user_test"):

    if request.method == 'POST':
        form = Form(request.POST)
        if form.is_valid():
            print(f"user_id={user_id}; survey={form.cleaned_data}")
            pqs = PreQuestionnaire.objects.filter(user_id=user_id)
            if len(pqs):
                pqs.objects.all().delete()
                print("Erasing previous form")
            pq = PreQuestionnaire(
                user_id=user_id,
                gender=form.cleaned_data["gender"],
                age=form.cleaned_data["age"],
                education=form.cleaned_data["education"],
                confidence=form.cleaned_data["confidence"])
            pq.save()
            return redirect(reverse('modeling_test',
                                    kwargs={'after': 0,
                                            'user_id': user_id}))

    else:
        form = Form()

    return render(request, 'pre_questionnaire/index.html', {'form': form,
                                                            'user_id': user_id})
