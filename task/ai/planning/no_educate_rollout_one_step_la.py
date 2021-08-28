import torch
from scipy.special import expit
import numpy as np

from .utils import \
    generate_features_k, terminal_cost, user_simulator_switching


def random_base_heuristic_noeducate(
        xi,
        corr_mat,
        user_model,
        cost,
        time_to_go,
        n_collinear,
        n_noncollinear,
        test_datasets,
        theta_1,
        theta_2,
        user_switch_sim_a,
        terminal_cost_err_mlt,
        n_samples):

    """
    Rollout method's base heuristic:
    random action selection without tutoring actions
    """

    n_covars = n_collinear + n_noncollinear

    user_type, w_00, w_01, w_10, w_11, e = user_model
    weights = torch.tensor([[w_00, w_01], [w_10, w_11]], dtype=torch.double)

    cost_sum = 0
    for _ in range(n_samples):
        xi_ = xi.clone()

        for k in range(time_to_go):
            recommend_or_educate_action = np.random.choice(n_covars)

            phi_k = generate_features_k(n_covars, corr_mat, xi_)

            outcome = user_simulator_switching(
                action=phi_k[recommend_or_educate_action, :],
                W=weights,
                a=user_switch_sim_a,
                educability=e,
                user_type=user_type)

            if outcome == 1:
                xi_[recommend_or_educate_action] = True
            else:
                xi_[recommend_or_educate_action] = False

            cost_sum += cost[recommend_or_educate_action]

        cost_sum += terminal_cost(test_datasets=test_datasets,
                                  user_model=user_model, xi=xi_,
                                  n_collinear=n_collinear,
                                  n_noncollinear=n_noncollinear,
                                  theta_1=theta_1,
                                  theta_2=theta_2,
                                  user_switch_a=user_switch_sim_a,
                                  error_multiplier=terminal_cost_err_mlt)

    return cost_sum / n_samples


def no_educate_rollout_one_step_la(
        xi,
        corr_mat,
        sample_user_model,
        cost,
        time_to_go,
        n_collinear,
        n_noncollinear,
        test_datasets,
        prev_action,
        theta_1,
        theta_2,
        heuristic_n_samples,
        user_switch_sim_a,
        terminal_cost_err_mlt):

    """
    Rollout with one-step look-ahead without tutoring actions. So typical machine teaching with rollout.
    """

    n_covars = n_collinear + n_noncollinear

    user_type, _, _, _, _, educability = sample_user_model
    phi_k = generate_features_k(n_covars, corr_mat, xi)
    weights = torch.tensor([sample_user_model[(user_type * 2) + 1],
                            sample_user_model[(user_type * 2) + 2]],
                           dtype=torch.double)
    logits_per_covar = phi_k @ weights
    probs_per_covar = expit(logits_per_covar)

    # The q(x_k, u_k, w_k) values.
    # One per recommend + one for educate (last one).
    # Initialize them with g_k stage cost.
    q_factors = torch.zeros(n_covars)
    q_factors[:n_covars] += cost[:n_covars]

    for action_index in range(n_covars):
        # Recommend action
        # Make sure this is proper copy
        xi_1 = xi.clone()
        xi_1[action_index] = True
        cost_to_go_1 = random_base_heuristic_noeducate(
            xi=xi_1,
            corr_mat=corr_mat,
            user_model=sample_user_model,
            cost=cost,
            time_to_go=time_to_go - 1,
            n_collinear=n_collinear,
            n_noncollinear=n_noncollinear,
            test_datasets=test_datasets,
            theta_1=theta_1,
            theta_2=theta_2,
            n_samples=heuristic_n_samples,
            user_switch_sim_a=user_switch_sim_a,
            terminal_cost_err_mlt=terminal_cost_err_mlt)

        xi_0 = xi.clone()
        xi_0[action_index] = False
        cost_to_go_0 = random_base_heuristic_noeducate(
            xi=xi_1,
            corr_mat=corr_mat,
            user_model=sample_user_model,
            cost=cost,
            time_to_go=time_to_go - 1,
            n_collinear=n_collinear,
            n_noncollinear=n_noncollinear,
            test_datasets=test_datasets,
            theta_1=theta_1,
            theta_2=theta_2,
            n_samples=heuristic_n_samples,
            user_switch_sim_a=user_switch_sim_a,
            terminal_cost_err_mlt=terminal_cost_err_mlt)

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