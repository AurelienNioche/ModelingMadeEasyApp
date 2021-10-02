import torch
from scipy.special import expit
import numpy as np

from . utils import \
    generate_features_k, user_simulator_switching, terminal_cost


def random_base_heuristic_weducate(
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
    random action selection with tutoring actions included
    """

    n_covars = n_collinear + n_noncollinear

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

            else:

                user_type = user_simulator_switching(
                    action=-1,
                    W=weights,
                    a=user_switch_sim_a,
                    educability=e,
                    user_type=user_type)

                cost_sum += cost[-1]

        cost_sum += terminal_cost(test_datasets=test_datasets,
                                  user_model=user_model,
                                  xi=xi_,
                                  n_collinear=n_collinear,
                                  n_noncollinear=n_noncollinear,
                                  theta_1=theta_1,
                                  theta_2=theta_2,
                                  error_multiplier=terminal_cost_err_mlt,
                                  user_switch_a=user_switch_sim_a)

    return cost_sum / n_samples


def rollout_one_step_TS_la(
        xi, corr_mat,
        type_probs_mean, betas_mean, educability,
        cost, time_to_go,
        n_collinear,
        n_noncollinear, test_datasets,
        prev_action,
        theta_1,
        theta_2,
        heuristic_n_samples,
        user_switch_sim_a,
        terminal_cost_err_mlt):

    """
    Rollout with one-step look-ahead, for a known user type
    Using Thompson sampling
    theta_1 = u_2,  theta_2 = u_1 on the paper.
    """

    # Moving the Thompson-sampling directly into the planning
    
    sample_user_type = np.random.choice(2, p=type_probs_mean)

    sample_user_model = (
            sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2],
            betas_mean[3], educability)


    ###

    n_covars = n_collinear + n_noncollinear

    user_type = sample_user_type
    phi_k = generate_features_k(n_covars, corr_mat, xi)
    weights = torch.tensor([sample_user_model[(user_type * 2) + 1],
                            sample_user_model[(user_type * 2) + 2]],
                           dtype=torch.double)
    logits_per_covar = phi_k @ weights
    probs_per_covar = expit(logits_per_covar)

    # The q(x_k, u_k, w_k) values.
    # One per recommend + one for educate (last one).
    # Initialize them with g_k stage cost.
    q_factors = torch.zeros(n_covars + 1)
    q_factors[-1] += cost[-1]
    q_factors[:n_covars] += cost[:n_covars]

    for action_index in range(n_covars + 1):
        # Recommend action
        if action_index < n_covars:
            kwargs_rollout = dict(
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
            
            
            # Make sure this is proper copy
            xi_1 = xi.clone()
            xi_1[action_index] = True
            # Rollout if variable is included
            cost_to_go_1 = random_base_heuristic_weducate(xi=xi_1, **kwargs_rollout)


            xi_0 = xi.clone()
            xi_0[action_index] = False
            # Rollout if variable is not included
            cost_to_go_0 = random_base_heuristic_weducate(xi=xi_0, **kwargs_rollout)

            q_factors[action_index] += \
                probs_per_covar[action_index] \
                * cost_to_go_1 \
                + (1 - probs_per_covar[action_index]) \
                * cost_to_go_0

        # Tutoring action
        else:
            kwargs_rollout = dict(
                    xi=xi,
                    corr_mat=corr_mat,
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
            
            
            user_model_1 = (
                    1, sample_user_model[1], sample_user_model[2], sample_user_model[3], sample_user_model[4],
                    sample_user_model[5])
            
            # Rollout if user transitioned to type 1
            cost_to_go_1 = random_base_heuristic_weducate(user_model=user_model_1, **kwargs_rollout)
            
            
            if user_type == 0:
                # Rollout if user stayed at type 0
                cost_to_go_0 = random_base_heuristic_weducate(user_model=sample_user_model, **kwargs_rollout)

                q_factors[action_index] += (educability * cost_to_go_1 + (1 - educability) * cost_to_go_0) * type_probs_mean[user_type]

            else:
                q_factors[action_index] += cost_to_go_1 * type_probs_mean[user_type]

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







def rollout_one_step_la(
        xi, corr_mat,
        type_probs_mean, betas_mean, educability,
        cost, time_to_go,
        n_collinear,
        n_noncollinear, test_datasets,
        prev_action,
        theta_1,
        theta_2,
        heuristic_n_samples,
        user_switch_sim_a,
        terminal_cost_err_mlt):

    """
    Rollout with one-step look-ahead, for a known user type
    theta_1 = u_2,  theta_2 = u_1 on the paper.
    """

    # Moving the Thompson-sampling directly into the planning
    
    sample_user_type = np.random.choice(2, p=type_probs_mean)

    sample_user_model = (
            sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2],
            betas_mean[3], educability)


    ###

    n_covars = n_collinear + n_noncollinear

    user_type, _, _, _, _, educability = sample_user_model
    phi_k = generate_features_k(n_covars, corr_mat, xi)


    # The q(x_k, u_k, w_k) values.
    # One per recommend + one for educate (last one).
    # Initialize them with g_k stage cost.
    q_factors = torch.zeros(n_covars + 1)
    q_factors[-1] += cost[-1]
    q_factors[:n_covars] += cost[:n_covars]
    
    

    for user_type in range(len(type_probs_mean)):
        
        weights = torch.tensor([sample_user_model[(user_type * 2) + 1],
                                                  sample_user_model[(user_type * 2) + 2]], dtype=torch.double)
        logits_per_covar = phi_k @ weights
        probs_per_covar = expit(logits_per_covar)
        
        for action_index in range(n_covars + 1):
            # Recommend action
            if action_index < n_covars:
                kwargs_rollout = dict(
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
                
                
                # Make sure this is proper copy
                xi_1 = xi.clone()
                xi_1[action_index] = True
                # Rollout if variable is included
                cost_to_go_1 = random_base_heuristic_weducate(xi=xi_1, **kwargs_rollout)
    
    
                xi_0 = xi.clone()
                xi_0[action_index] = False
                # Rollout if variable is not included
                cost_to_go_0 = random_base_heuristic_weducate(xi=xi_0, **kwargs_rollout)
    
                q_factors[action_index] += \
                    (probs_per_covar[action_index] \
                    * cost_to_go_1 \
                    + (1 - probs_per_covar[action_index]) \
                    * cost_to_go_0) \
                    * type_probs_mean[user_type]
    
            # Tutoring action
            else:
                kwargs_rollout = dict(
                        xi=xi,
                        corr_mat=corr_mat,
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
                
                if user_type == 0:
                    user_model_1 = (
                        1, sample_user_model[1], sample_user_model[2], sample_user_model[3], sample_user_model[4],
                        sample_user_model[5])
                    
                    # Rollout if user transitioned to type 1
                    cost_to_go_1 = random_base_heuristic_weducate(user_model=user_model_1, **kwargs_rollout)
                    
                    # Rollout if user stayed at type 0
                    cost_to_go_0 = random_base_heuristic_weducate(user_model=sample_user_model, **kwargs_rollout)
    
                    q_factors[action_index] += (educability * cost_to_go_1 + (1 - educability) * cost_to_go_0) * type_probs_mean[user_type]
    
                else:
                    # Rollout if user stayed at type 1 (no other option)
                    cost_to_go = random_base_heuristic_weducate(user_model=sample_user_model, **kwargs_rollout)
    
                    q_factors[action_index] += cost_to_go * type_probs_mean[user_type]

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