from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse


def validate(request, user_id):
    action_type = request.POST.get("action_type")
    parameters = request.POST.get("parameters")
    timestamp = request.POST.get("timestamp")
    print(f"user_id={user_id}; action_type={action_type}; "
          f"parameters={parameters}; timestamp={timestamp}")
    return JsonResponse({"valid": True})


def index(request, user_id='user_test'):

    if request.method == 'POST':
        return redirect(reverse(
            'modeling_test',
            kwargs={
                'user_id': user_id,
                'after': 1}))

    return render(request, 'group1/index.html', {'user_id': user_id})