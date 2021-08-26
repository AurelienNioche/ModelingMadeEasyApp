import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.special import expit
import tkinter as tk
import time
import subprocess 
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


# from mme_stan_regression import test_stan
# from mme_stan_regression.generate_data import generate_data
# from mme_stan_regression.user_simulators import user_simulator_switching

import test_stan
import generate_data
from user_simulators import user_simulator_switching

sns.set()
plt.interactive(True)

error_multiplier = 5 # Determines how much penalty for collinearity errors 
tutoring_cost = 0.5 # Cost for making a 'tutoring' action
recommend_cost = 0.05 # Cost for making a 'recommend' action

def generate_features_k(n_covars, corr_mat, xi):
    # Generates the feature vectors per covariate, based on current xi.
    phi_k = torch.zeros((n_covars, 2), dtype=torch.double)

    for i in range(n_covars):
        mask = xi.numpy().copy()
        mask[i] = False
        masked = corr_mat[i, mask]
        if masked.size != 0:
            max_cross_corr = np.max(masked)
        else:
            max_cross_corr = 0.0

        phi_k[i, 0] = corr_mat[i, -1]
        phi_k[i, 1] = max_cross_corr

    return phi_k


# Posterior Sampling Machine Education of Switching Learners
def ts_teach_user_switching(dataset, W_typezero=(5.0, 0.0), W_typeone=(5.0, -5.0), educability=0.01, n_interactions=100,
                            user_model_file=None, initial_user_type=0, planning_function=None, test_datasets=None,
                            n_training_collinear=0, n_training_noncollinear=0, theta_1=0.5, theta_2=0.5):
    training_X, training_y, test_X, test_y, _, _ = dataset
    corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0, 1)))
    n_covars = training_X.shape[1]
    data_dict = {"N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone], "educability": educability,
                 "forgetting": 0.0}

    # xi is n_covars + 1 because its size must match the corr_mat which includes the output y.
    aux_data_dict = {"xi": torch.zeros(n_covars + 1, dtype=torch.bool)}

    # cost[0,...,n_covars-1] is recommendation cost per covariate. cost[n_covars] is educate cost.
    cost = torch.zeros(n_covars + 1) + recommend_cost
    cost[-1] = tutoring_cost

    recommend_actions = []
    educate_or_recommend = []
    change_point = -1

    fit = None
    user_type = initial_user_type

    cumulative_cost = np.zeros(n_interactions)

    for i in range(n_interactions):
        # Indices for grabbing the necessary statistics from Stan fit object
        strt = 6 + (3 * i)
        endn = strt + i

        print("Step: {}".format(i))
        if i == 0:
            # First action must be recommend for numerical purposes
            educate_or_recommend.append(1)
            # Uniform choice of covariate
            recommend_actions.append(np.random.choice(n_covars, 1).item())
            # Action index is the index of the latest recommended covariate
            act_in = recommend_actions[-1]

            cumulative_cost[i] += cost[act_in]

        else:

            ### User model estimation part. Uses Certainty Equivalence to replace betas with their expectation, and posterior sampling to sample a user type.
            s = fit.summary()
            summary = pandas.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])
            betas_mean = list(summary.iloc[2:6, 0])
            betas_mean[1], betas_mean[2] = betas_mean[2], betas_mean[1]
            # E[\alpha_0] and E[\alpha_1], posterior expectations of type-0 and type-1 probabilities
            type_probs_mean = list(summary.iloc[[strt, endn], 0])
            sample_user_type = np.random.choice(2, p=type_probs_mean)

            sample_user_model = (
                sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2], betas_mean[3], educability)
            ###

            # Returns educate OR recommend and if recommend, recommended covar's index. IF not, r_i is None.

            e_or_r, r_i = planning_function(n_covars, aux_data_dict["xi"], corr_mat, sample_user_model, cost,
                                            n_interactions - i + 1, n_training_collinear, n_training_noncollinear,
                                            test_datasets)

            # If *educate*
            if e_or_r == 0:
                educate_or_recommend.append(e_or_r)
                act_in = -1
                cumulative_cost[i] = cumulative_cost[i - 1] + cost[-1]
            else:
                # Action indices of recommend actions are the index of the recommended covariate \in {0,...,n_covars-1}
                educate_or_recommend.append(e_or_r)
                recommend_actions.append(r_i)
                act_in = recommend_actions[-1]
                cumulative_cost[i] = cumulative_cost[i - 1] + cost[recommend_actions[-1]]

        # If the chosen action is to recommend
        if act_in != -1:
            # Get the currently selected variables as boolean mask
            mask = aux_data_dict["xi"].numpy().copy()

            # Set the recommended variable to False. Otherwise cross-correlations will include self-correlation.
            mask[act_in] = False

            # Get the cross-correlations between recommended var and included vars.
            masked = corr_mat[act_in, mask]

            # If there are more than one variables included
            if masked.size != 0:
                # The maximum absolute cross-correlation to selected vars.
                max_cross_corr = np.max(masked)
            else:
                # Set to zero since there are no vars to cross-correlate to
                max_cross_corr = 0.0

            # Generate the action's observation vector for the user: (corr, cross_corr).
            action = torch.tensor([corr_mat[act_in, -1], max_cross_corr])

            # Outcome of recommend action from simulator. This is 0 if refused, 1 if accepted.
            # Attention! This is the USER SIMULATOR. This will be replaced with real user feedback in human study.
            # The teacher only sees the outcome variable. Does not have access to W_typezero or W_typeone directly.
            outcome = user_simulator_switching(action, torch.tensor([W_typezero, W_typeone], dtype=torch.double), a=1.0,
                                               educability=data_dict["educability"], user_type=user_type)
            print("Variable: {} Outcome: {}".format(act_in, outcome))

            if outcome == 1:
                # If accepted, set the variable's inclusion indicator to true.
                aux_data_dict["xi"][act_in] = True
            else:
                # If not, set it to false.
                aux_data_dict["xi"][act_in] = False

            # Add the observation to dataset. So this observations are like: (corr, cross_corr), outcome.
            data_dict["x"].append(action.tolist())
            data_dict["y"].append(outcome)

        # This is an educate action!
        else:

            # Outcome of an educate action is the updated user type.
            # This is not observed by the teacher! This is just for book-keeping and tracking. Does not leak to the teacher.
            user_type = user_simulator_switching(act_in, torch.tensor([W_typezero, W_typeone], dtype=torch.double),
                                                 a=1.0, educability=data_dict["educability"], user_type=user_type)

            # The dummy action observation vector for educate actions. This is needed to ignore these in Stan.
            action = [-1.0, -1.0]

            # Dummy outcome. Not used for anything.
            outcome = 0

            data_dict["x"].append(action)
            data_dict["y"].append(outcome)
            if user_type == 1 and change_point == -1:
                print("State Changed to Type 1 at iteration: {}".format(i))
                change_point += (i + 1)

        # After the action execution, and observation of outcome, update our model of the user. The fit gives us the current model.
        data_dict["N"] += 1
        fit, user_model_file = test_stan.fit_model_w_education(data_dict, user_model_file)

    # print(fit)
    # arviz.plot_trace(fit)
    # pyplot.show()

    s_ = fit.summary()
    summary = pandas.DataFrame(s_['summary'], columns=s_['summary_colnames'], index=s_['summary_rownames'])

    print(summary.iloc[2:6, :])

    # pyplot.plot(list(summary.iloc[(7+(3*n_interactions)):(7+(4*n_interactions)), 0]))
    # if change_point != -1:
    #    pyplot.axvline(x=change_point, ymin=0, ymax=1, color='r', linestyle='--')
    # pyplot.scatter(x=np.arange(n_interactions), y=np.zeros(n_interactions), c=educate_or_recommend, s=2.5, marker="x",
    # cmap="bone")
    #
    # pyplot.savefig("changepoint_experiment_e{}.png".format(educability))

    cumulative_cost[-1] = cumulative_cost[-2] + terminal_cost(test_datasets=test_datasets, user_model=sample_user_model,
                                                              xi=aux_data_dict["xi"],
                                                              n_training_collinear=n_training_collinear,
                                                              n_training_noncollinear=n_training_noncollinear,
                                                              theta_1=theta_1, theta_2=theta_2)
    print("Planning Method: {}, The selected variables: {}".format(planning_function.__name__, aux_data_dict["xi"]))
    # pyplot.scatter(x=np.arange(n_interactions+1), y=cumulative_cost)
    # pyplot.show()

    performance_evaluation_sets = [
        generate_data.generate_data(n_noncollinear=n_training_noncollinear, n_collinear=n_training_collinear, n=100) for
        _ in
        range(10)]
    performance = performance_evaluation(test_datasets=performance_evaluation_sets, user_model=sample_user_model,
                                         n_training_noncollinear=n_training_noncollinear)

    return summary.iloc[(7 + (3 * n_interactions)):(7 + (4 * n_interactions)),
           0].to_numpy(), cumulative_cost, performance, change_point, educate_or_recommend


# Terminal cost for variable selection example. Number of collinears must be 1, all noncollinears must be included.
def terminal_cost(test_datasets=None, user_model=None, xi=None,
                  n_training_collinear=None, n_training_noncollinear=None, theta_1=1.0, theta_2=1.0):
    n_testsets = len(test_datasets)
    test_X = []
    test_y = []
    n_test_collinear = []
    n_test_noncollinear = []
    test_regrets = np.zeros(n_testsets)

    for i in range(n_testsets):
        test_X.append(test_datasets[i][0])
        test_y.append(test_datasets[i][1])
        n_test_collinear.append(test_datasets[i][4])
        n_test_noncollinear.append(test_datasets[i][5])

        recommend_actions = list(
            np.random.choice(n_test_collinear[i] + n_test_noncollinear[i], n_test_collinear[i] + n_test_noncollinear[i],
                             replace=False))
        aux_data_dict = {"xi": torch.zeros(n_test_collinear[i] + n_test_noncollinear[i] + 1, dtype=torch.bool)}
        corr_mat = np.abs(
            np.corrcoef(torch.transpose(torch.cat((test_X[i], test_y[i].unsqueeze(dim=1)), dim=1), 0, 1)))

        for action_index in recommend_actions:
            mask = aux_data_dict["xi"].numpy().copy()
            mask[action_index] = False
            masked = corr_mat[action_index, mask]

            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[action_index, -1], max_cross_corr])
            outcome = user_simulator_switching(action, torch.tensor(
                [[user_model[1], user_model[2]], [user_model[3], user_model[4]]], dtype=torch.double), a=1.0,
                                               educability=user_model[5], user_type=user_model[0])
            if outcome == 1:
                aux_data_dict["xi"][action_index] = True
            else:
                aux_data_dict["xi"][action_index] = False

        xi_n_collinear = torch.sum(aux_data_dict["xi"][0:n_test_collinear[i]])
        xi_n_noncollinear = torch.sum(aux_data_dict["xi"][n_test_collinear[i]:])
        test_regrets[i] = error_multiplier * ((np.abs(xi_n_collinear - 1).item()) + np.abs(n_test_noncollinear[i] - xi_n_noncollinear).item())

    xi_n_collinear = torch.sum(xi[0:n_training_collinear])
    xi_n_noncollinear = torch.sum(xi[n_training_collinear:-1])

    total_regret = error_multiplier * ((theta_1 * np.mean(test_regrets)) + theta_2 * ((np.abs(xi_n_collinear - 1)) + np.abs(n_training_noncollinear - xi_n_noncollinear)))
    return total_regret.item()


def performance_evaluation(test_datasets=None, user_model=None, n_training_noncollinear=None):
    n_testsets = len(test_datasets)
    test_X = []
    test_y = []
    n_test_collinear = []
    n_test_noncollinear = []
    test_regrets = np.zeros(n_testsets)

    for i in range(n_testsets):
        test_X.append(test_datasets[i][0])
        test_y.append(test_datasets[i][1])
        n_test_collinear.append(test_datasets[i][4])
        n_test_noncollinear.append(test_datasets[i][5])

        recommend_actions = list(
            np.random.choice(n_test_collinear[i] + n_test_noncollinear[i], n_test_collinear[i] + n_test_noncollinear[i],
                             replace=False))
        aux_data_dict = {"xi": torch.zeros(n_test_collinear[i] + n_test_noncollinear[i] + 1, dtype=torch.bool)}
        corr_mat = np.abs(
            np.corrcoef(torch.transpose(torch.cat((test_X[i], test_y[i].unsqueeze(dim=1)), dim=1), 0, 1)))

        for action_index in recommend_actions:
            mask = aux_data_dict["xi"].numpy().copy()
            mask[action_index] = False
            masked = corr_mat[action_index, mask]

            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[action_index, -1], max_cross_corr])
            outcome = user_simulator_switching(action, torch.tensor(
                [[user_model[1], user_model[2]], [user_model[3], user_model[4]]], dtype=torch.double), a=1.0,
                                               educability=user_model[5], user_type=user_model[0])
            if outcome == 1:
                aux_data_dict["xi"][action_index] = True
            else:
                aux_data_dict["xi"][action_index] = False

        xi_n_collinear = torch.sum(aux_data_dict["xi"][0:n_test_collinear[i]])
        xi_n_noncollinear = torch.sum(aux_data_dict["xi"][n_test_collinear[i]:])
        test_regrets[i] = error_multiplier * (np.abs(xi_n_collinear - 1).item() + np.abs(n_test_noncollinear[i] - xi_n_noncollinear).item())

    total_regret = np.mean(test_regrets)

    return total_regret.item()


# Rollout method's base heuristic: random action selection without tutoring actions
def random_base_heuristic_noeducate(n_covars, xi, corr_mat, user_model, cost, time_to_go, n_training_collinear,
                                    n_training_noncollinear, test_datasets, n_samples=10, theta_1=1.0, theta_2=1.0):
    user_type, w_00, w_01, w_10, w_11, e = user_model
    weights = torch.tensor([[w_00, w_01], [w_10, w_11]], dtype=torch.double)

    cost_sum = 0
    for _ in range(n_samples):
        xi_ = xi.clone()

        for k in range(time_to_go):
            recommend_or_educate_action = np.random.choice(n_covars)

            phi_k = generate_features_k(n_covars, corr_mat, xi_)

            outcome = user_simulator_switching(phi_k[recommend_or_educate_action, :], weights,
                                               a=1.0, educability=e, user_type=user_type)
            if outcome == 1:
                xi_[recommend_or_educate_action] = True
            else:
                xi_[recommend_or_educate_action] = False

            cost_sum += cost[recommend_or_educate_action]

        cost_sum += terminal_cost(test_datasets=test_datasets,
                                  user_model=user_model, xi=xi_,
                                  n_training_collinear=n_training_collinear,
                                  n_training_noncollinear=n_training_noncollinear, theta_1=theta_1, theta_2=theta_2)

    return cost_sum / n_samples


# Rollout method's base heuristic: random action selection with tutoring actions included
def random_base_heuristic_weducate(n_covars, xi, corr_mat, user_model, cost, time_to_go, n_training_collinear,
                                   n_training_noncollinear, test_datasets, n_samples=10, theta_1=1.0, theta_2=1.0):
    user_type, w_00, w_01, w_10, w_11, e = user_model
    weights = torch.tensor([[w_00, w_01], [w_10, w_11]], dtype=torch.double)

    cost_sum = 0
    for _ in range(n_samples):
        xi_ = xi.clone()

        for k in range(time_to_go):
            if user_type == 0:
                recommend_or_educate_action = np.random.choice(n_covars + 1)
            else:
                recommend_or_educate_action = np.random.choice(n_covars)

            if recommend_or_educate_action != n_covars:
                phi_k = generate_features_k(n_covars, corr_mat, xi_)

                outcome = user_simulator_switching(phi_k[recommend_or_educate_action, :], weights,
                                                   a=1.0, educability=e, user_type=user_type)
                if outcome == 1:
                    xi_[recommend_or_educate_action] = True
                else:
                    xi_[recommend_or_educate_action] = False

                cost_sum += cost[recommend_or_educate_action]

            else:

                user_type = user_simulator_switching(-1, weights,
                                                     a=1.0, educability=e, user_type=user_type)

                cost_sum += cost[-1]

        cost_sum += terminal_cost(test_datasets=test_datasets,
                                  user_model=user_model, xi=xi_,
                                  n_training_collinear=n_training_collinear,
                                  n_training_noncollinear=n_training_noncollinear, theta_1=theta_1, theta_2=theta_2)

    return cost_sum / n_samples


# Rollout with one-step look-ahead.
# theta_1 = u_2,  theta_2 = u_1 on the paper.
def rollout_onestep_la(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go, n_training_collinear=None,
                       n_training_noncollinear=None, test_datasets=None, base_heuristic=random_base_heuristic_weducate,
                       theta_1=1.0, theta_2=1.0, prev_action=None):
    user_type, _, _, _, _, educability = sample_user_model
    phi_k = generate_features_k(n_covars, corr_mat, xi)
    weights = torch.tensor([sample_user_model[(user_type * 2) + 1], sample_user_model[(user_type * 2) + 2]],
                           dtype=torch.double)
    logits_per_covar = phi_k @ weights
    probs_per_covar = expit(logits_per_covar)

    # The q(x_k, u_k, w_k) values. One per recommend + one for educate (last one). Initialize them with g_k stage cost.
    q_factors = torch.zeros(n_covars + 1)
    q_factors[-1] += cost[-1]
    q_factors[:n_covars] += cost[:n_covars]

    for action_index in range(n_covars + 1):
        # Recommend action
        if action_index < n_covars:
            # Make sure this is proper copy
            xi_1 = xi.clone()
            xi_1[action_index] = True
            cost_to_go_1 = base_heuristic(n_covars, xi_1, corr_mat, sample_user_model, cost, time_to_go - 1,
                                          n_training_collinear, n_training_noncollinear, test_datasets, theta_1=theta_1,
                                          theta_2=theta_2)

            xi_0 = xi.clone()
            xi_0[action_index] = False
            cost_to_go_0 = base_heuristic(n_covars, xi_1, corr_mat, sample_user_model, cost, time_to_go - 1,
                                          n_training_collinear, n_training_noncollinear, test_datasets, theta_1=theta_1,
                                          theta_2=theta_2)

            q_factors[action_index] += probs_per_covar[action_index] * cost_to_go_1 + (
                    1 - probs_per_covar[action_index]) * cost_to_go_0

        else:
            if user_type == 0:
                user_model_1 = (
                    1, sample_user_model[1], sample_user_model[2], sample_user_model[3], sample_user_model[4],
                    sample_user_model[5])
                cost_to_go_1 = base_heuristic(n_covars, xi, corr_mat, user_model_1, cost, time_to_go - 1,
                                              n_training_collinear, n_training_noncollinear, test_datasets,
                                              theta_1=theta_1, theta_2=theta_2)
                cost_to_go_0 = base_heuristic(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go - 1,
                                              n_training_collinear, n_training_noncollinear, test_datasets,
                                              theta_1=theta_1, theta_2=theta_2)
                q_factors[action_index] += educability * cost_to_go_1 + (1 - educability) * cost_to_go_0

            else:
                cost_to_go = base_heuristic(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go - 1,
                                            n_training_collinear, n_training_noncollinear, test_datasets,
                                            theta_1=theta_1, theta_2=theta_2)
                q_factors[action_index] += cost_to_go

    # Do not repeat same action twice back to back.
    if prev_action is None:
        act = torch.argmin(q_factors).item()
    else:
        acts = torch.topk(q_factors, 2, largest=False)[1]
        if acts[0].item() == prev_action:
            act = acts[1].item()
        else:
            act = acts[0].item()

    if act < n_covars:
        e_or_r = 1
    else:
        e_or_r = 0
        act = None
    return e_or_r, act


# Rollout with one-step look-ahead without tutoring actions. So typical machine teaching with rollout.
def noeducate_rollout_onestep_la(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go, n_training_collinear=None,
                                 n_training_noncollinear=None, test_datasets=None,
                                 base_heuristic=random_base_heuristic_noeducate, theta_1=1.0, theta_2=1.0,
                                 prev_action=None):
    user_type, _, _, _, _, educability = sample_user_model
    phi_k = generate_features_k(n_covars, corr_mat, xi)
    weights = torch.tensor([sample_user_model[(user_type * 2) + 1], sample_user_model[(user_type * 2) + 2]],
                           dtype=torch.double)
    logits_per_covar = phi_k @ weights
    probs_per_covar = expit(logits_per_covar)

    # The q(x_k, u_k, w_k) values. One per recommend + one for educate (last one). Initialize them with g_k stage cost.
    q_factors = torch.zeros(n_covars)
    q_factors[:n_covars] += cost[:n_covars]

    for action_index in range(n_covars):
        # Recommend action
        # Make sure this is proper copy
        xi_1 = xi.clone()
        xi_1[action_index] = True
        cost_to_go_1 = base_heuristic(n_covars, xi_1, corr_mat, sample_user_model, cost, time_to_go - 1,
                                      n_training_collinear, n_training_noncollinear, test_datasets, theta_1=theta_1,
                                      theta_2=theta_2)

        xi_0 = xi.clone()
        xi_0[action_index] = False
        cost_to_go_0 = base_heuristic(n_covars, xi_1, corr_mat, sample_user_model, cost, time_to_go - 1,
                                      n_training_collinear, n_training_noncollinear, test_datasets, theta_1=theta_1,
                                      theta_2=theta_2)

        q_factors[action_index] += probs_per_covar[action_index] * cost_to_go_1 + (
                1 - probs_per_covar[action_index]) * cost_to_go_0

    # Do not repeat same action twice back to back.
    if prev_action is None:
        act = torch.argmin(q_factors).item()
    else:
        acts = torch.topk(q_factors, 2, largest=False)[1]
        if acts[0].item() == prev_action:
            act = acts[1].item()
        else:
            act = acts[0].item()

    if act < n_covars:
        e_or_r = 1
    else:
        e_or_r = 0
    return e_or_r, act


# Random policy with no planning.
def random_no_planning(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go, n_training_collinear=None,
                       n_training_noncollinear=None, test_datasets=None,
                       base_heuristic=random_base_heuristic_noeducate):
    r_i = np.random.choice(n_covars + 1)
    if r_i == n_covars:
        e_or_r = 0
    else:
        e_or_r = 1

    return e_or_r, r_i


# Smoke test for planning methods.
def test_planning(n_covars, sample_user_model, cost, phi_k, time_to_go, test_datasets):
    user_type, _, _, _, _, educability = sample_user_model
    weight_0, weight_1 = sample_user_model[(user_type * 2) + 1], sample_user_model[(user_type * 2) + 2]

    expected_t_absorption = 1 / educability
    education_fraction = expected_t_absorption / time_to_go
    if education_fraction >= 1.0:
        p_educate = 0.1
    else:
        p_educate = 0.5

    if user_type == 0:
        print("Probability of choosing educate: {}".format(p_educate))
        educate_or_recommend = np.random.choice(2, p=(p_educate, 1 - p_educate))
        if educate_or_recommend == 0:
            return educate_or_recommend, None

        else:
            r_i = np.random.choice(n_covars)
            return educate_or_recommend, r_i

    else:
        return 1, np.random.choice(n_covars)


# Posterior Sampling Machine Education of Switching Learners
def ts_teach_user_study(ax1, ax2, dataset, W_typezero=(7.0, 0.0), W_typeone=(7.0, -7.0), educability=0.01, n_interactions=100,
                        user_model_file=None, initial_user_type=0, planning_function=None, test_datasets=None,
                        n_training_collinear=0, n_training_noncollinear=0, theta_1=0.5, theta_2=0.5, user_id=0, group_id=0, task_id=0):

    training_X, training_y, test_X, test_y, _, _ = dataset

    from model import Model
    m = Model(ax1=ax1, ax2=ax2,
              educability=educability,
              training_X=training_X, training_y=training_y,
              test_datasets=test_datasets,
              W_typezero=W_typezero,
              W_typeone = W_typeone,
              planning_function=planning_function,
              n_training_collinear=n_training_collinear,
              n_training_noncollinear=n_training_noncollinear,
              n_interactions=n_interactions)
    for _ in range(m.n_interactions):
        m.do_one_step()

    s_ = m.fit.summary()
    summary = pandas.DataFrame(s_['summary'], columns=s_['summary_colnames'], index=s_['summary_rownames'])

    print(summary.iloc[2:6, :])

    # pyplot.plot(list(summary.iloc[(7+(3*n_interactions)):(7+(4*n_interactions)), 0]))
    # if change_point != -1:
    #    pyplot.axvline(x=change_point, ymin=0, ymax=1, color='r', linestyle='--')
    # pyplot.scatter(x=np.arange(n_interactions), y=np.zeros(n_interactions), c=educate_or_recommend, s=2.5, marker="x",
    # cmap="bone")
    #
    # pyplot.savefig("changepoint_experiment_e{}.png".format(educability))

    sample_user_model = m.sampled_user_models[-1]
    cumulative_cost = m.cumulative_cost
    task_start_time = m.task_start_time
    change_point = -1
    educate_or_recommend = m.educate_or_recommend
    recommend_actions = m.recommend_actions
    user_responses = m.user_responses
    sampled_user_models = m.sampled_user_models
    user_response_times = m.user_response_times
    corr_mat = m.corr_mat
    is_incorrect_action = m.is_incorrect_action
    system_action_string = m.system_action_string
    user_response_string = m.user_response_string
    task_time_list = m.task_time_list


    cumulative_cost[-1] = cumulative_cost[-2] + terminal_cost(test_datasets=test_datasets, user_model=sample_user_model,
                                                              xi=m.aux_data_dict["xi"],
                                                              n_training_collinear=n_training_collinear,
                                                              n_training_noncollinear=n_training_noncollinear,
                                                              theta_1=theta_1, theta_2=theta_2)
    print("Planning Method: {}, The selected variables: {}".format(planning_function.__name__, m.aux_data_dict["xi"]))
    # pyplot.scatter(x=np.arange(n_interactions+1), y=cumulative_cost)
    # pyplot.show()

    performance_evaluation_sets = [
        generate_data.generate_data(n_noncollinear=n_training_noncollinear, n_collinear=n_training_collinear, n=100) for
        _ in
        range(10)]
    performance = performance_evaluation(test_datasets=performance_evaluation_sets, user_model=sample_user_model,
                                         n_training_noncollinear=n_training_noncollinear)

    task_elapsed_time = time.time() - task_start_time

    return_dict = {"important_summary_stats": summary.iloc[(7 + (3 * n_interactions)):(7 + (4 * n_interactions)),
           0].to_numpy(), "cumulative_cost": cumulative_cost, "performance": performance,
                   "change_point": change_point, "educate_or_recommend": educate_or_recommend, "recommend_actions": recommend_actions,
                   "sampled_user_models": sampled_user_models, "user_responses":user_responses,
                   "user_response_times": user_response_times, "entire_summary_stats": summary, "corr_mat": corr_mat}

    user_id_list = [user_id for _ in range(1,n_interactions + 1)]
    group_id_list = [group_id for _ in range(1,n_interactions + 1)]
    task_id_list = [task_id for _ in range(1,n_interactions + 1)]
    num_collinear_list = [n_training_collinear for _ in range(1,n_interactions + 1)]
    num_noncollinear_list = [n_training_noncollinear for _ in range(1,n_interactions + 1)]
    

    
    return_csv_task = pandas.DataFrame({"userID": user_id_list, "groupID": group_id_list, "taskID":task_id_list, "numCollinear":num_collinear_list, "numNoncollinear":num_noncollinear_list,"modelBuiltSoFar": model_built_sofar, 
    "isIncorrectAction": is_incorrect_action, "systemAction": system_action_string,
                                        "userResponse": user_response_string, "cumulativeCost": cumulative_cost, "userResponseTime": user_response_times, "taskElapsedTime":task_time_list})

    return return_dict, return_csv_task


#################################""""


if __name__ == "__main__":
    e = .25
    n_nncl = 4
    n_cl = 6
    theta_1 = .5

    training_dataset = generate_data.generate_data(n_noncollinear=n_nncl, n_collinear=n_cl, n=100)
    test_datasets = [generate_data.generate_data(n_noncollinear=n_nncl, n_collinear=n_cl, n=100) for _ in range(10)]
    test_datasets.append(training_dataset)

    ts_teach_user_study(educability=e, dataset=training_dataset,
                        planning_function=rollout_onestep_la,
                        n_interactions=20,
                        test_datasets=test_datasets,
                        n_training_noncollinear=n_nncl,
                        n_training_collinear=n_cl, theta_1=theta_1,
                        theta_2=1 - theta_1)