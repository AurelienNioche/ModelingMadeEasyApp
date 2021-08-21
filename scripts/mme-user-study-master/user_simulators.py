import arviz
import numpy as np
import pandas
import seaborn as sns
import torch
import torch.distributions as dist
from matplotlib import pyplot

import test_stan
import generate_data

sns.set()


# np.random.seed(1)

def user_simulator_typezero(action, W, a, educability=0.6):
    # action is either a tuple, or -1 for educate.

    # Educate action
    if isinstance(action, int):
        print("Educate!")
        educate_o = dist.Bernoulli(educability).sample()
        return educate_o
    else:
        probs = a + action @ W

        a_o = dist.Bernoulli(logits=probs).sample()
        return int(a_o.item())


def user_simulator_typeone(action, W, a, educability=0.6):
    # action is either a tuple, or -1 for educate.

    # Educate action
    if isinstance(action, int):
        print("Educate!")
        educate_o = dist.Bernoulli(educability).sample()
        return educate_o
    else:
        probs = a + action @ W

        a_o = dist.Bernoulli(logits=probs).sample()
        return int(a_o.item())


def user_simulator_switching(action, W, a, educability=0.1, user_type=0, forgetting=0.0):
    # action is either a tuple, or -1 for educate.
    # W[0] is the type-zero user weights, W[1] type-one.
    # Educate action

    educability_per_type = [educability, 1.0]
    if isinstance(action, int):
        user_type_ = int(dist.Bernoulli(educability_per_type[user_type]).sample().item())
        #if user_type != user_type_:
        #    print("User Type Changed!")
        return user_type_

    else:
        probs = a + action @ W[user_type]

        a_o = dist.Bernoulli(logits=probs).sample()
        return int(a_o.item())


def test_user_typezero():
    training_X, training_y, test_X, test_y, _, _ = generate_data(n_noncollinear=50, n_collinear=100, n=100)
    corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0, 1)))
    W_typezero = [5.0, 0.0]
    W_typeone = [5.0, -5.0]
    n_covars = training_X.shape[1]

    data_dict = {"N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone]}
    aux_data_dict = {"xi": torch.zeros(n_covars + 1, dtype=torch.bool)}
    n_iterations = 100
    teacher_actions = list(np.random.choice(n_covars, n_iterations))

    model_file = None
    for i in range(n_iterations):
        act_in = teacher_actions[i]
        if act_in != -1:
            mask = aux_data_dict["xi"].numpy().copy()
            mask[act_in] = False
            masked = corr_mat[act_in, mask]
            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[act_in, -1], max_cross_corr])
            outcome = user_simulator_typezero(action, torch.tensor(W_typezero, dtype=torch.double), a=1.0)
            if outcome == 1.0:
                aux_data_dict["xi"][act_in] = True
            else:
                aux_data_dict["xi"][act_in] = False

            data_dict["x"].append(action.tolist())
            data_dict["y"].append(outcome)
            data_dict["N"] += 1

            fit, model_file = test_stan.fit_model_w_education(data_dict, model_file)
    arviz.plot_trace(fit)
    pyplot.show()


def test_user_typeone():
    training_X, training_y, test_X, test_y, _, _ = generate_data(n_noncollinear=50, n_collinear=100, n=100)
    corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0, 1)))
    W_typezero = [5.0, 0.0]
    W_typeone = [5.0, -5.0]
    n_covars = training_X.shape[1]

    data_dict = {"N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone]}
    aux_data_dict = {"xi": torch.zeros(n_covars + 1, dtype=torch.bool)}
    n_iterations = 20
    teacher_actions = list(np.random.choice(n_covars, n_iterations))

    model_file = None
    for i in range(n_iterations):
        act_in = teacher_actions[i]
        if act_in != -1:
            mask = aux_data_dict["xi"].numpy().copy()
            mask[act_in] = False
            masked = corr_mat[act_in, mask]
            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[act_in, -1], max_cross_corr])
            outcome = user_simulator_typeone(action, torch.tensor(W_typeone, dtype=torch.double), a=1.0)
            if outcome == 1.0:
                aux_data_dict["xi"][act_in] = True
            else:
                aux_data_dict["xi"][act_in] = False

            data_dict["x"].append(action.tolist())
            data_dict["y"].append(outcome)
            data_dict["N"] += 1

            fit, model_file = test_stan.fit_model_w_education(data_dict, model_file)
    arviz.plot_trace(fit)
    pyplot.show()





def test_user_switching(educability=0.01):
    training_X, training_y, test_X, test_y, _, _ = generate_data(n_noncollinear=50, n_collinear=100, n=100)
    corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0, 1)))
    sns.heatmap(corr_mat)
    pyplot.show()
    W_typezero = [5.0, 0.0]
    W_typeone = [5.0, -5.0]
    n_covars = training_X.shape[1]

    data_dict = {"N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone], "educability": educability,
                 "forgetting": 0.0}
    aux_data_dict = {"xi": torch.zeros(n_covars + 1, dtype=torch.bool)}
    n_iterations = 100
    recommend_actions = list(np.random.choice(n_covars, n_iterations))
    educate_or_recommend = list(np.random.choice(2, n_iterations, p=(0.5, 0.5)))
    educate_or_recommend[0] = 1

    model_file = None
    user_type = 0
    change_point = 0
    for i in range(n_iterations):
        #print("Step: {}".format(i))
        if educate_or_recommend[i] == 0:
            act_in = -1
        else:
            act_in = recommend_actions[i]
        if act_in != -1:
            mask = aux_data_dict["xi"].numpy().copy()
            mask[act_in] = False
            masked = corr_mat[act_in, mask]
            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[act_in, -1], max_cross_corr])
            outcome = user_simulator_switching(action, torch.tensor([W_typezero, W_typeone], dtype=torch.double), a=1.0,
                                               educability=data_dict["educability"], user_type=user_type)
            if outcome == 1:
                aux_data_dict["xi"][act_in] = True
            else:
                aux_data_dict["xi"][act_in] = False

            data_dict["x"].append(action.tolist())
            data_dict["y"].append(outcome)
        else:
            _user_type = 0 + user_type
            user_type = user_simulator_switching(act_in, torch.tensor([W_typezero, W_typeone], dtype=torch.double),
                                                 a=1.0, educability=data_dict["educability"], user_type=user_type)
            action = [-1.0, -1.0]
            outcome = 0
            data_dict["x"].append(action)
            data_dict["y"].append(outcome)
            if user_type == 1 and _user_type == 0:
                print("State Changed to Type 1 at iteration: {}".format(i))
                change_point += i

        data_dict["N"] += 1

        fit, model_file = test_stan.fit_model_w_education(data_dict, model_file)
        # if i % 100 ==0:
    s = fit.summary()
    print(fit)
    arviz.plot_trace(fit)
    pyplot.show()
    summary = pandas.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])

    print(summary.iloc[2:6, :])
    strt = 6 + (3 * n_iterations)
    endn = strt + n_iterations
    print(summary.iloc[strt, :])
    print(summary.iloc[endn, :])
    pyplot.plot(list(summary.iloc[307:407, 0]))
    pyplot.axvline(x=change_point, ymin=0, ymax=1, color='r', linestyle='--')

    pyplot.scatter(x=np.arange(n_iterations), y=np.zeros(n_iterations), c=educate_or_recommend, s=1.5, marker="x",
                   cmap="bone")

    pyplot.savefig("interaction_alpha_e{}_test.png".format(educability), dpi=300)

