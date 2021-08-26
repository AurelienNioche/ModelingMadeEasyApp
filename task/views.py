from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse

import json

from .models import TaskLog, UserModel, UserData

from .dataset.generate_data import generate_data

from .ai.ai import AiAssistant
from .ai.planning.rollout_one_step_la import rollout_onestep_la
from .ai.planning.no_educate_rollout_one_step_la import no_educate_rollout_one_step_la


class Action:

    VIS_1 = "vis1-x"
    VIS_2_Y = "vis2-y"
    ADD = "add"
    REMOVE = "remove"
    AI_ACCEPT = "accept"
    AI_REFUSE = "refuse"
    AI_IGNORE = "ignore"
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

    @classmethod
    def trigger_feedback(cls):
        return [
            cls.AI_ACCEPT, cls.AI_REFUSE, cls.AI_IGNORE, cls.AI_CLOSE_TUTORIAL
        ]


def format_data(training_X, training_y):
    X = training_X.T.tolist()
    y = training_y.tolist()
    data = {}
    for i, x in enumerate(X):
        data[f"X{i+1}"] = x
    data["Y"] = y
    return json.dumps(data)


def init(user_id, group_id):

    uds = UserData.objects.filter(user_id=user_id)
    print(uds)
    if len(uds):
        print("user already exists, I just load the data")
        # return uds[0].value
        UserData.objects.all().delete()
        UserModel.objects.all().delete()

    # ----------- if not already existing ----------------- #
    print("Generating data")

    n_data_points = 100
    n_test_dataset = 10

    n_collinear = 4  # difficulty[t][0]
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

    training_X, training_y, test_X, test_y, _, _ = training_dataset

    data = format_data(training_X=training_X, training_y=training_y)

    print("Creating and saving user data...")
    ud = UserData(user_id=user_id, group_id=group_id, value=data)
    ud.save()
    print("Done")

    if group_id == 0:
        return data

    # ---------------- only for group 1 and 2 ------------------- #
    if group_id == 1:
        planning_function = no_educate_rollout_one_step_la

    elif group_id == 2:
        planning_function = rollout_onestep_la
    else:
        raise ValueError(f"Group id incorrect: {group_id}")

    print("generating test data sets...")
    test_datasets = [generate_data(n_noncollinear=n_noncollinear,
                                   n_collinear=n_collinear,
                                   n=n_data_points) for
                     _ in range(n_test_dataset)]
    test_datasets.append(training_dataset)

    print("creating ai assistant")

    ai = AiAssistant(
        planning_function=planning_function,
        training_X=training_X,
        training_y=training_y,
        test_datasets=test_datasets,
        W_typezero=W_typezero,
        W_typeone=W_typeone,
        educability=educability,
        n_training_collinear=n_collinear,
        n_training_noncollinear=n_noncollinear,
        n_interactions=n_interactions,
        init_var_cost=init_cost,
        init_educ_cost=init_last_cost,
        user_model_file="task/ai/compiled_stan_model/mixture_model_w_ed.pkl")

    print("creating and saving user model")
    um = UserModel(user_id=user_id, group_id=group_id, value=ai)
    um.save()
    print("done")
    return data


def unformat_included_vars(included_vars):
    included_vars = included_vars.replace("X", "")
    included_vars = included_vars.split(",")
    included_vars = [int(x) - 1 for x in included_vars if len(x)]
    return included_vars


def format_rec(rec_item, rec_item_cor):

    if rec_item == AiAssistant.EDUCATE:
        rec_item = "educate"
        rec_item_cor = None
    else:
        rec_item = f"X{rec_item + 1}"
        rec_item_cor = f"X{rec_item_cor + 1}"

    return rec_item, rec_item_cor


def get_recommendation(user_id, included_vars):

    print("getting recommendation")

    included_vars = unformat_included_vars(included_vars)

    um = UserModel.objects.get(user_id=user_id)
    ai = um.value
    rec_item, rec_item_cor = ai.get_recommendation(included_vars)

    print("saving object")

    um.value = ai
    um.save()
    print("returning response")
    return format_rec(rec_item, rec_item_cor)


def user_feedback(user_id, action_type):

    um = UserModel.objects.get(user_id=user_id)

    ai = um.value
    outcome = int(action_type == Action.AI_ACCEPT)
    ai.user_feedback(outcome)
    um.value = ai
    um.save()
    print("I saved User Model")


def user_action(request, user_id, group_id):

    action_type = request.POST.get("action_type")
    action_var = request.POST.get("action_var")
    timestamp = request.POST.get("timestamp")
    included_vars = request.POST.get("included_vars")
    print(f"user_id={user_id}; action_type={action_type}; "
          f"action_var={action_var}; timestamp={timestamp}")

    if action_type not in Action.list():
        raise ValueError(f"`action_type` not recognized: {action_type}")

    if group_id == 0:
        rec_item, rec_item_cor = None, None

    elif action_type in Action.trigger_rec():

        rec_item, rec_item_cor = get_recommendation(
            user_id=user_id,
            included_vars=included_vars)

    elif action_type in Action.trigger_feedback():

        user_feedback(user_id=user_id, action_type=action_type)
        rec_item, rec_item_cor = None, None

    else:
        rec_item, rec_item_cor = None, None

    print(f"user_id={user_id} I recommend", rec_item, " and for cor", rec_item_cor)

    tl = TaskLog(
        user_id=user_id,
        group_id=group_id,
        action_type=action_type,
        action_var=action_var,
        rec_item=rec_item,
        rec_item_cor=rec_item_cor,
        included_vars=included_vars,
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

    # init_rec_item, init_rec_item_cor = 1, 1
    data = init(
        user_id=user_id,
        group_id=group_id)
    # if group_id == 0:
    #     init_rec_item, init_rec_item_cor = 1, 1
    # else:
    #     init_rec_item, init_rec_item_cor = get_recommendation(
    #         user_id=user_id,
    #         group_id=group_id,
    #         init=True)

    if group_id == 0:
        template_name = 'task/group0.html'
    else:
        template_name = 'task/group1_and_2.html'

    return render(request, template_name,
                  {'user_id': user_id,
                   'group_id': group_id,
                   'data': data})
