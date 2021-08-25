import torch
from scipy.special import expit

from .utils import generate_features_k
from .baselines.baselines import random_base_heuristic_noeducate


def no_educate_rollout_one_step_la(n_covars, xi, corr_mat, sample_user_model, cost, time_to_go, n_training_collinear=None,
                                 n_training_noncollinear=None, test_datasets=None,
                                 base_heuristic=random_base_heuristic_noeducate, theta_1=1.0, theta_2=1.0,
                                 prev_action=None):
    """
    Rollout with one-step look-ahead without tutoring actions. So typical machine teaching with rollout.
    """
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