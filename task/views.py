from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse

from .models import TaskLog

import numpy as np


class Action:

    VIS_1 = "vis-1"
    VIS_2_X = "vis-2-x"
    VIS_2_Y = "vis-2-y"
    ADD = "add"
    REMOVE = "remove"
    AI = "accept"
    SUBMIT = "submit"

    @classmethod
    def list(cls):
        return [getattr(cls, v) for v in vars(cls) if not v.startswith("__")]

    @classmethod
    def trigger_rec(cls):
        return cls.ADD, cls.REMOVE, cls.AI

    @classmethod
    def model_action(cls):
        return cls.ADD, cls.REMOVE


def get_recommendation():
    return np.random.randint(8) + 1


def user_action(request, user_id, group_id):

    action_type = request.POST.get("action_type")
    parameters = request.POST.get("parameters")
    timestamp = request.POST.get("timestamp")
    print(f"user_id={user_id}; action_type={action_type}; "
          f"parameters={parameters}; timestamp={timestamp}")

    if action_type not in Action.list():
        raise ValueError

    if group_id == 0 or action_type not in Action.trigger_rec():
        rec_item = None

    elif action_type in Action.model_action() and parameters == "":
        rec_item = None

    else:
        rec_item_idx = get_recommendation()
        rec_item = f"X{rec_item_idx}"
    print(f"user_id={user_id} I recommend", rec_item)

    tl = TaskLog(
        user_id=user_id,
        group_id=group_id,
        action_type=action_type,
        parameters=parameters,
        rec_item=rec_item,
        timestamp=timestamp)
    tl.save()

    return JsonResponse({
        "valid": True,
        "rec_item": rec_item
    })


def index(request, user_id='user_test', group_id=0):

    if request.method == 'POST':
        return redirect(reverse(
            'modeling_test',
            kwargs={
                'user_id': user_id,
                'after': 1}))

    if group_id == 0:
        init_rec_item = None
    else:
        init_rec_item = "X3"

    print("init_rec_item", init_rec_item)

    return render(request, f'task/group{group_id}.html',
                  {'user_id': user_id, 'group_id': group_id,
                   'init_rec_item': init_rec_item})
