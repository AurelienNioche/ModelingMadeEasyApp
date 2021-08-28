import torch
import numpy as np
import torch.distributions as dist


def generate_features_k(n_covars, corr_mat, xi):

    """
    Generates the feature vectors per covariate, based on current xi.
    """

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


def user_simulator_switching(action, W, a, educability, user_type=0):

    """
    action is either a tuple, or -1 for educate.
    W[0] is the type-zero user weights, W[1] type-one.
    """

    educability_per_type = [educability, 1.0]
    if isinstance(action, int):
        user_type_ = int(
            dist.Bernoulli(educability_per_type[user_type]).sample().item())
        # if user_type != user_type_:
        #    print("User Type Changed!")
        return user_type_

    else:
        probs = a + action @ W[user_type]

        a_o = dist.Bernoulli(logits=probs).sample()
        return int(a_o.item())


def terminal_cost(test_datasets,
                  n_collinear,
                  n_noncollinear,
                  user_model,
                  xi,
                  theta_1,
                  theta_2,
                  error_multiplier,
                  user_switch_a):

    """
    Terminal cost for variable selection example.
    Number of collinears must be 1, all noncollinears must be included. <???>
    error_multiplier: Determines how much penalty for collinearity errors
    """

    n_testsets = len(test_datasets)
    test_X = []
    test_y = []
    n_covars = n_collinear + n_noncollinear
    test_regrets = np.zeros(n_testsets)

    for i in range(n_testsets):
        test_X.append(test_datasets[i][0])
        test_y.append(test_datasets[i][1])

        recommend_actions = np.random.choice(n_covars, n_covars,
                                             replace=False)
        aux_data = torch.zeros(n_covars + 1, dtype=torch.bool)
        corr_mat = np.abs(
            np.corrcoef(torch.transpose(
                torch.cat((test_X[i], test_y[i].unsqueeze(dim=1)), dim=1),
                0, 1)))

        for action_index in recommend_actions:
            mask = aux_data.numpy().copy()
            mask[action_index] = False
            masked = corr_mat[action_index, mask]

            if masked.size != 0:
                max_cross_corr = np.max(masked)
            else:
                max_cross_corr = 0.0

            action = torch.tensor([corr_mat[action_index, -1], max_cross_corr])
            outcome = user_simulator_switching(
                action=action,
                W=torch.tensor([[user_model[1], user_model[2]],
                                [user_model[3], user_model[4]]],
                               dtype=torch.double),
                a=1.0,
                educability=user_model[5],
                user_type=user_model[0])

            if outcome == 1:
                aux_data[action_index] = True
            else:
                aux_data[action_index] = False

        xi_n_collinear = torch.sum(aux_data[0:n_collinear])
        xi_n_noncollinear = torch.sum(aux_data[n_collinear:])
        test_regrets[i] = \
            error_multiplier \
            * ((np.abs(xi_n_collinear - 1).item())
               + np.abs(n_noncollinear - xi_n_noncollinear).item())

    xi_n_collinear = torch.sum(xi[0:n_collinear])
    xi_n_noncollinear = torch.sum(xi[n_collinear:-1])

    total_regret = \
        error_multiplier \
        * ((theta_1 * np.mean(test_regrets))
           + theta_2 * ((np.abs(xi_n_collinear - 1))
                        + np.abs(n_noncollinear - xi_n_noncollinear)))
    return total_regret.item()
