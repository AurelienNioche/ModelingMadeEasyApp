import torch
import numpy as np

from task.ai.planning.utils import generate_features_k
from .utils import user_simulator_switching
from .utils import terminal_cost


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