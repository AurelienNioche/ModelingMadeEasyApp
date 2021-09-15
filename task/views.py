from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse

import json
import numpy as np

from .models import TaskLog, UserModel, UserData

from .dataset.generate_data import generate_data

from .ai.ai import AiAssistant
from .ai.planning.rollout_one_step_la \
    import rollout_one_step_la
from .ai.planning.no_educate_rollout_one_step_la \
    import no_educate_rollout_one_step_la

from .config import config


RANDOM_AI_SELECT = False
RECREATE_AT_RELOAD = True


class Action:

    VIS_1 = "vis1-x"
    VIS_2_Y = "vis2-y"
    ADD = "add"
    REMOVE = "remove"
    AI_YES = "accept"
    AI_NO = "refuse"
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
            cls.AI_YES, cls.AI_NO, cls.AI_IGNORE,
            cls.AI_CLOSE_TUTORIAL,
            cls.ADD, cls.REMOVE
        ]


def format_data(dataset):

    training_X, training_y = dataset

    X = training_X.T.tolist()
    y = training_y.tolist()
    data = {}
    for i, x in enumerate(X):
        data[f"X{i+1}"] = x
    data["Y"] = y
    return json.dumps(data)


def unformat_included_vars(included_vars):

    included_vars = included_vars.replace("X", "")
    included_vars = included_vars.split(",")
    included_vars = [int(x) - 1 for x in included_vars if len(x)]
    return included_vars


def unformat_var(var):

    if var == "educate":
        return AiAssistant.EDUCATE

    try:
        return int(var.replace("X", "")) - 1
    except ValueError:
        return None


def format_rec(rec_item, rec_item_cor):

    if rec_item == AiAssistant.EDUCATE:
        rec_item = "educate"
        rec_item_cor = None
    else:
        rec_item = f"X{rec_item + 1}"
        rec_item_cor = f"X{rec_item_cor + 1}"

    return rec_item, rec_item_cor


def init(user_id, group_id):

    uds = UserData.objects.filter(user_id=user_id)
    if len(uds):
        if RECREATE_AT_RELOAD:
            UserData.objects.all().delete()
            UserModel.objects.all().delete()
            print(f"user_id={user_id}: Recreating data and ai")

        else:
            print(f"user_id={user_id}: "
                  f"user already exists, I just load the data")
            return uds[0].value

    # ----------- if not already existing ----------------- #

    print(f"user_id={user_id}: Generating data...",
          end=" ", flush=True)
    kwargs_data = dict(
        n_noncollinear=config.N_NONCOLLINEAR,
        n_collinear=config.N_COLLINEAR,
        n=config.N_DATA_POINTS,
        std_collinear=config.STD_COLLINEAR,
        std_noncollinear=config.STD_NONCOLLINEAR,
        noise_collinear=config.NOISE_COLLINEAR,
        coeff_collinear=config.COEFF_COLLINEAR,
        coeff_noncollinear=config.COEFF_NONCOLLINEAR,
        coeff_intercept=config.COEFF_INTERCEPT,
        phi=config.PHI)
    training_dataset = generate_data(**kwargs_data,
                                     seed=config.TRAINING_DATASET_SEED)
    data = format_data(training_dataset)
    print("Done")

    print(f"user_id={user_id}: Creating and saving user data...",
          end=" ", flush=True)
    ud = UserData(user_id=user_id, group_id=group_id, value=data)
    ud.save()
    print("Done")

    if group_id == 0:
        return data

    # ---------------- only for group 1 and 2 ------------------- #
    if group_id == 1:
        planning_function = no_educate_rollout_one_step_la

    elif group_id == 2:
        planning_function = rollout_one_step_la
    else:
        raise ValueError(f"Group id incorrect: {group_id}")

    print(f"user_id={user_id}: Generating test data sets...",
          end=" ", flush=True)
    test_datasets = [generate_data(**kwargs_data, seed=seed)
                     for seed in config.TEST_DATASET_SEEDS]
    test_datasets.append(training_dataset)
    print("Done")
    print(f"user_id={user_id}: Creating ai assistant...",
          end=" ", flush=True)

    ai = AiAssistant(
        dataset=training_dataset,
        planning_function=planning_function,
        test_datasets=test_datasets,
        educability=config.EDUCABILITY,
        forgetting=config.FORGETTING,
        n_collinear=config.N_COLLINEAR,
        n_noncollinear=config.N_NONCOLLINEAR,
        n_interactions=config.N_INTERACTIONS,
        cost_var=config.COST_VAR,
        cost_edu=config.COST_EDU,
        theta_1=config.THETA_1,
        theta_2=config.THETA_2,
        heuristic_n_samples=config.HEURISTIC_N_SAMPLES,
        user_switch_sim_a=config.USER_SWITCH_SIM_A,
        terminal_cost_err_mlt=config.TERMINAL_COST_ERR_MLT,
        stan_compiled_model_file="task/ai/stan_model/mixture_model_w_ed.pkl")
    print("Done")
    print(f"user_id={user_id}: "
          "Creating and saving user model...", end=" ", flush=True)
    um = UserModel(user_id=user_id, group_id=group_id, value=ai)
    um.save()
    print("Done")
    return data


def get_recommendation(user_id, included_vars):

    if RANDOM_AI_SELECT:
        rec_item = np.random.randint(8)
        rec_item_cor = np.random.randint(8)

    else:

        print(f"user_id={user_id}: Getting recommendation...",
              end=" ", flush=True)
        included_vars = unformat_included_vars(included_vars)
        um = UserModel.objects.get(user_id=user_id)
        ai = um.value
        rec_item, rec_item_cor = ai.act(included_vars)
        print("Done")

        print(f"user_id={user_id}: Saving object...", end=" ")
        um.value = ai
        um.save()
        print("Done")

    return format_rec(rec_item, rec_item_cor)


def user_feedback(user_id, action_var, included_vars):

    if RANDOM_AI_SELECT:
        return

    action_var = unformat_var(action_var)
    included_vars = unformat_included_vars(included_vars)

    print(f"user_id={user_id}: Updating AI for action_var={action_var}, "
          f"included_vars={included_vars}...", end=" ", flush=True)
    um = UserModel.objects.get(user_id=user_id)
    ai = um.value
    ai.update(action_var, included_vars)
    print("Done")

    print(f"user_id={user_id}: Saving object...", end=" ")
    um.value = ai
    um.save()
    print("Done")


def user_action(request, user_id, group_id):

    action_type = request.POST.get("action_type")
    action_var = request.POST.get("action_var")
    timestamp = request.POST.get("timestamp")
    included_vars = request.POST.get("included_vars")
    print(f"user_id={user_id}: action_type={action_type}; "
          f"action_var={action_var}; timestamp={timestamp}")

    if action_type not in Action.list():
        raise ValueError(f"user_id={user_id}: "
                         f"`action_type` not recognized: {action_type}")

    if group_id == 0:
        rec_item, rec_item_cor = None, None

    elif action_type in Action.trigger_rec():

        rec_item, rec_item_cor = get_recommendation(
            user_id=user_id,
            included_vars=included_vars)

    elif action_type in Action.trigger_feedback():

        user_feedback(user_id=user_id,
                      action_var=action_var,
                      included_vars=included_vars)
        rec_item, rec_item_cor = None, None

    else:
        rec_item, rec_item_cor = None, None

    print(f"user_id={user_id}: I recommend", rec_item, " and for cor", rec_item_cor)

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
        return redirect(reverse('modeling_test',
                                kwargs={
                                    'user_id': user_id,
                                    'after': 1}))
    data = init(
        user_id=user_id,
        group_id=group_id)

    if group_id == 0:
        template_name = 'task/group0.html'
    else:
        template_name = 'task/group1_and_2.html'

    return render(request, template_name,
                  {'user_id': user_id,
                   'group_id': group_id,
                   'data': data})
