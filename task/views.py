from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse

import numpy as np
import pickle

from .models import TaskLog

from .dataset.generate_data import generate_data

from .ai.ai import AiAssistant
from .ai.planning.rollout_one_step_la import rollout_onestep_la
from .ai.planning.no_educate_rollout_one_step_la import no_educate_rollout_one_step_la

DUMMY_VALUE = -1


class Action:

    VIS_1 = "vis1-x"
    VIS_2_Y = "vis2-y"
    ADD = "add"
    REMOVE = "remove"
    AI_ACCEPT = "accept"
    AI_NEW = "new"
    AI_CLOSE_TUTORIAL = "close-tutorial"
    SUBMIT = "submit"

    @classmethod
    def list(cls):
        return [getattr(cls, v) for v in vars(cls) if not v.startswith("__")]

    @classmethod
    def trigger_rec(cls):
        return [
            cls.AI_NEW
        ]

    # @classmethod
    # def model_action(cls):
    #     return cls.ADD, cls.REMOVE


def create_ai_assistant(group_id):

    if group_id == 1:
        planning_function = no_educate_rollout_one_step_la
    elif group_id == 2:
        planning_function = rollout_onestep_la
    else:
        raise ValueError(f"Group id incorrect: {group_id}")

    # difficulty = [(5, 3), (4, 4), (3, 5)]
    # t = 0  # task id

    n_data_points = 100
    n_test_dataset = 10

    n_collinear = 4    # difficulty[t][0]
    n_noncollinear = 4  # difficulty[t][1]

    # FORGET ABOUT THESE ????
    W_typezero = (7.0, 0.0)
    W_typeone = (7.0, -7.0)

    educability = 0.30
    init_cost = 0.05
    init_last_cost = 0.5

    n_interactions = 10000

    training_dataset = generate_data(n_noncollinear=n_noncollinear,
                                     n_collinear=n_collinear,
                                     n=n_data_points)
    test_datasets = [generate_data(n_noncollinear=n_noncollinear,
                                   n_collinear=n_collinear,
                                   n=n_data_points) for
                     _ in range(n_test_dataset)]
    test_datasets.append(training_dataset)

    training_X, training_y, test_X, test_y, _, _ = training_dataset

    ai = AiAssistant(
        planning_function=planning_function,
        training_X=training_X,
        training_y=training_y,
        W_typezero=W_typezero,
        W_typeone=W_typeone,
        educability=educability,
        n_training_collinear=n_collinear,
        n_training_noncollinear=n_noncollinear,
        n_interactions=n_interactions,
        init_cost=init_cost,
        init_last_cost=init_last_cost,
        user_model_file="task/ai/compiled_stan_model/mixture_model_w_ed.pkl")
    return ai


def get_recommendation(user_id, group_id, init=False):

    if init:
        return DUMMY_VALUE, DUMMY_VALUE
    #     ai = create_ai_assistant(group_id=group_id)
    #
    # else:
    #     with open(f'task/ai/user_models/{user_id}.p', 'rb') as f:
    #         ai = pickle.load(f)
    #
    # rec_item = ai.get_recommendation()
    #
    # with open(f'task/ai/user_models/{user_id}.p', 'rb') as f:
    #     pickle.dump(ai, f)
    #
    # if rec_item == ai.EDUCATE:
    #     rec_item = "educate"
    # else:
    #     rec_item = f"X{rec_item}"
    rec_item = np.random.randint(0, 8) + 1 # "educate"
    rec_item_cor = np.random.randint(0, 8) + 1

    # rec_item = AiAssistant.EDUCATE

    if rec_item == AiAssistant.EDUCATE:
        rec_item = "educate"
        rec_item_cor = DUMMY_VALUE

    return rec_item, rec_item_cor


def user_action(request, user_id, group_id):

    action_type = request.POST.get("action_type")
    parameters = request.POST.get("parameters")
    timestamp = request.POST.get("timestamp")
    print(f"user_id={user_id}; action_type={action_type}; "
          f"parameters={parameters}; timestamp={timestamp}")

    if action_type not in Action.list():
        raise ValueError(f"`action_type` not recognized: {action_type}")

    if group_id == 0 or action_type not in Action.trigger_rec():
        rec_item, rec_item_cor = None, None

    # elif action_type in Action.model_action() and parameters in ("", None):
    #     rec_item, rec_item_cor = None, None

    else:
        rec_item, rec_item_cor = get_recommendation(
            user_id=user_id, group_id=group_id)

    print(f"user_id={user_id} I recommend", rec_item, "and for cor", rec_item_cor)

    tl = TaskLog(
        user_id=user_id,
        group_id=group_id,
        action_type=action_type,
        parameters=parameters,
        rec_item=rec_item,
        rec_item_cor=rec_item_cor,
        timestamp=timestamp)
    tl.save()

    return JsonResponse({
        "valid": True,
        "rec_item": rec_item,
        "rec_item_cor": rec_item_cor
    })


def index(request, user_id='user_test', group_id=0):

    if request.method == 'POST':
        return redirect(reverse(
            'modeling_test',
            kwargs={
                'user_id': user_id,
                'after': 1}))

    if group_id == 0:
        init_rec_item, init_rec_item_cor = DUMMY_VALUE, DUMMY_VALUE
    else:
        init_rec_item, init_rec_item_cor = get_recommendation(
            user_id=user_id,
            group_id=group_id,
            init=True)

    print("init_rec_item", init_rec_item, "init_rec_item_cor", init_rec_item_cor)

    if group_id == 0:
        template_name = 'task/group0.html'
    else:
        template_name = 'task/group1_and_2.html'

    return render(request, template_name,
                  {'user_id': user_id,
                   'group_id': group_id,
                   'init_rec_item': init_rec_item,
                   'init_rec_item_cor': init_rec_item_cor})
