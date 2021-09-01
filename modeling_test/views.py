from django.shortcuts import render, redirect
from django.urls import reverse

from .forms import Form
from .models import ModelingTest


N_CONDITION = 3


def attribute_group():

    n_users = ModelingTest.objects.values_list("user_id", flat=True).distinct().count()
    group_id = (n_users - 1) % N_CONDITION  # -1: ignore current user
    return group_id


def index(request, after=0, user_id="user_test"):

    if request.method == 'POST':
        form = Form(request.POST)
        if form.is_valid():
            print(f"user_id={user_id}; modeling_test={form.cleaned_data}")

            mt = ModelingTest.objects.filter(user_id=user_id,
                                             is_after=bool(after))
            if len(mt):
                mt.objects.all().delete()
                print("Erasing previous form")

            mt = ModelingTest(
                user_id=user_id,
                is_after=bool(after),
                test_selection=",".join(form.cleaned_data["test_selection"]),
                test_reason=form.cleaned_data["test_reason"])
            mt.save()

            if after:
                return redirect(reverse('survey', kwargs={'user_id': user_id}))
            else:
                group_id = attribute_group()
                print(f"user_id={user_id}; group_id={group_id}")

                return redirect(reverse('task', kwargs={'user_id': user_id,
                                                        'group_id': group_id}))

    else:
        form = Form()

    return render(request, 'modeling_test/index.html', {'form': form,
                                                        'user_id': user_id,
                                                        'after': after})
