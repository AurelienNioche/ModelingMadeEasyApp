import torch.distributions as dist

import numpy as np
import torch


error_multiplier = 5   # Determines how much penalty for collinearity errors
tutoring_cost = 0.5    # Cost for making a 'tutoring' action
recommend_cost = 0.05  # Cost for making a 'recommend' action


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