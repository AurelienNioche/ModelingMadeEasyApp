import os
import pandas
import torch
import numpy as np

import pystan
import pickle

from torch.distributions import Normal, MultivariateNormal
import torch.distributions as dist

from scipy.special import expit

model_file = "task/ai/stan_model/mixture_model_w_ed.pkl"

error_multiplier = 5 # Determines how much penalty for collinearity errors


def generate_data(n_collinear=4, n_noncollinear=4, n=100):
    coeffs = torch.cat((torch.ones(n_collinear), torch.ones(n_noncollinear)))
    coeffs[0:n_collinear] = coeffs[0:n_collinear] - 0.9

    z = Normal(torch.tensor([0.0]), torch.tensor([10.0])).sample(torch.tensor([n]))
    x_colin = Normal(0, torch.tensor([0.1])).sample((n,torch.tensor([n_collinear])))
    x_colin = x_colin.squeeze()

    x_colin_test = Normal(0, torch.tensor([0.1])).sample((n,torch.tensor([n_collinear])))
    x_colin_test = x_colin_test.squeeze()


    for i in range(n_collinear):
        x_colin[:,i] = x_colin[:,i] + z.squeeze()
        x_colin_test[:,i] = x_colin_test[:,i] + z.squeeze()




    #Generate non co-linear covariates by sampling them independently
    x_noncolin = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample((n,torch.tensor([n_noncollinear])))
    x_noncolin = x_noncolin.squeeze()

    x_noncolin_test = Normal(torch.tensor([0.0]), torch.tensor([10.0])).sample((n,torch.tensor([n_noncollinear])))
    x_noncolin_test = x_noncolin_test.squeeze()


    design_matrix = torch.cat((x_colin,x_noncolin),dim=1)
    design_matrix_test = torch.cat((x_colin_test,x_noncolin_test),dim=1)

    #Generate outputs

    coeff_intercept = 1
    phi = 0.10
    covar_matrix = phi * torch.eye(n)

    reg_term = design_matrix @ coeffs
    reg_term_test = design_matrix_test @ coeffs


    output_rv = MultivariateNormal((torch.ones(n) * coeff_intercept) + reg_term, covar_matrix).sample()
    output_rv_test = MultivariateNormal((torch.ones(n) * coeff_intercept) + reg_term_test, covar_matrix).sample()



    return design_matrix, output_rv, design_matrix_test, output_rv_test, n_collinear, n_noncollinear


def fit_model_w_education(data):
    # data must be a dict containing --> N: number of datapoints, x: 2-d list of xs, y: 1-d list of ys, beta: 2-d list of weight vectors
    # If model_file is None, will compile. If not, use pre-compiled model from the file.
    mixture_with_tseries_model = """
    data {
        int<lower=0> N; // number of total interactions
        int<lower=0, upper=1> y[N]; // user responses
        row_vector[2] x[N]; // 
        real<lower=0, upper=1> educability;
        real<lower=0, upper=1> forgetting;

    }
    parameters {
        //real<lower=0, upper=1> pp_init;
        vector<lower=0.0, upper=1.0>[2] beta;
        //real<lower=0, upper=1> educability;
    }

    transformed parameters{
        vector[2] beta_constrained[2];
        real pp_init;
        matrix[N,2] mxs;
        matrix<lower=0, upper=1>[N,2] pp;
        vector[N] f;

        pp_init = 0.5;
        beta_constrained[1][2] = 0.0;
        beta_constrained[1][1] = 1.0 + (10.0) * beta[1];  
        beta_constrained[2][1] = 1.0 + (10.0) * beta[1];  
        beta_constrained[2][2] = -11.0 + (10.0) * beta[2];  


        for (n in 1:N){
            if (x[n][1] != -1.0){
                mxs[n,1] = exp(bernoulli_logit_lpmf( y[n] | 1.0 +  (x[n] *  beta_constrained[1])));
                mxs[n,2] = exp(bernoulli_logit_lpmf( y[n] | 1.0 +  (x[n] *  beta_constrained[2])));
            }

            else{
                mxs[n,1] = mxs[n-1,1];
                mxs[n,2] = mxs[n-1,2];
            }

        }        
        for (n in 1:N){

            if (n==1) {
                f[n] = (1-pp_init) * mxs[n,1] + pp_init * mxs[n,2]; 
                pp[n,1] = (1-pp_init) * mxs[n,1] / f[n];
                pp[n,2] = 1.0 - pp[n,1];
            }
            else {

                if (x[n][1] != -1.0) {

                    if(x[n-1][1] == -1.0){
                        f[n] = (1-educability) * pp[n-1,1] * mxs[n,1] + (educability) * pp[n-1,1]  * mxs[n,2] + pp[n-1,2] * mxs[n,2]; 
                        pp[n,1] = (1-educability) * pp[n-1,1] * mxs[n,1] / f[n];
                        pp[n,2] = 1.0 - pp[n,1];
                    }
                    else{
                        f[n] =  pp[n-1,1] * mxs[n,1] + pp[n-1,2] * (1-forgetting) * mxs[n,2] + pp[n-1,2] * forgetting * mxs[n,1]; 
                        pp[n,1] = (pp[n-1,1] * mxs[n,1] + pp[n-1,2] * forgetting * mxs[n,1]) / f[n];
                        pp[n,2] = 1.0 - pp[n,1];
                    }
                }

                else {
                    f[n] = 1.0;
                    pp[n,1] = (1-educability) * pp[n-1,1];
                    pp[n,2] = 1- pp[n,1];
                    //pp[n,1] = pp[n-1,1];
                    //pp[n,2] = pp[n-1,2];
                }
            }
        }
    }

    model {

        //Put more informative priors for the parameters
        target += sum(log(f));
        }

    """

    if not os.path.exists(model_file):
        sm = pystan.StanModel(model_code=mixture_with_tseries_model)
        fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)
        with open(model_file, "wb") as f:
            pickle.dump(sm, f)

        return fit, model_file
    else:

        with open(model_file, "rb") as f:
            sm = pickle.load(f)
            fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)

        return fit, model_file


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


# Posterior Sampling Machine Education of Switching Learners
def ts_teach_user_study(dataset, planning_function,
                        test_datasets,
                        init_var_cost,
                        init_edu_cost,
                        W_typezero,
                        W_typeone,
                        educability,
                        n_interactions,
                        n_training_collinear,
                        n_training_noncollinear):

    training_X, training_y, test_X, test_y, _, _ = dataset

    corr_mat = np.abs(np.corrcoef(torch.transpose(
        torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0, 1)))
    n_covars = training_X.shape[1]
    data_dict = {"N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone],
                 "educability": educability,
                 "forgetting": 0.0}

    included_vars = []

    # cost[0,...,n_covars-1] is recommendation cost per covariate. cost[n_covars] is educate cost.
    cost = torch.zeros(n_covars + 1) + init_var_cost
    cost[-1] = init_edu_cost

    fit = None
    prev_action = None

    for i in range(n_interactions):

        print("Step {} out of {}.".format(i + 1, n_interactions))

        print("Starting step", i + 1, "out of", n_interactions, "steps")

        # Indices for grabbing the necessary statistics from Stan fit object
        strt = 6 + (3 * i)
        endn = strt + i

        if len(included_vars) == 0:
            print("No variables currently included in the model")
        else:
            print("You already included the following variables:",
                  ", ".join(["X{}".format(i) for i in included_vars]))

        if i == 0:
            e_or_r = 1
            act_in = np.random.choice(n_covars, 1).item()

        else:

            ### User model estimation part. Uses Certainty Equivalence to replace betas with their expectation, and posterior sampling to sample a user type.
            s = fit.summary()
            summary = pandas.DataFrame(s['summary'],
                                       columns=s['summary_colnames'],
                                       index=s['summary_rownames'])
            betas_mean = list(summary.iloc[2:6, 0])
            betas_mean[1], betas_mean[2] = betas_mean[2], betas_mean[1]
            # print(summary.iloc[[strt, endn], 0])
            # E[\alpha_0] and E[\alpha_1], posterior expectations of type-0 and type-1 probabilities
            type_probs_mean = list(summary.iloc[[strt, endn], 0])
            sample_user_type = np.random.choice(2, p=type_probs_mean)

            sample_user_model = (
                sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2],
                betas_mean[3], educability)
            ###

            # Returns educate OR recommend and if recommend, recommended covar's index. IF not, r_i is None.
            # xi is n_covars + 1 because its size must match the corr_mat which includes the output y.
            aux_data_dict = torch.zeros(n_covars + 1, dtype=torch.bool)
            aux_data_dict[included_vars] = 1
            e_or_r, r_i = planning_function(n_covars, aux_data_dict,
                                            corr_mat, sample_user_model, cost,
                                            n_interactions - i + 1,
                                            n_training_collinear,
                                            n_training_noncollinear,
                                            test_datasets,
                                            prev_action=prev_action)
            prev_action = r_i

            # If *educate*
            if e_or_r == 0:
                act_in = -1

            else:
                act_in = r_i

        # If the chosen action is to recommend
        if act_in != -1:
            if act_in in included_vars:
                print(
                    "Would you like to keep variable X{} in the model?".format(
                        act_in))
            else:
                print(
                    "Would you like to include variable X{} in the model?".format(
                        act_in))

            # Get the currently selected variables as boolean mask
            mask = torch.zeros(n_covars+1, dtype=bool)
            mask[included_vars] = True

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
            action = [corr_mat[act_in, -1], max_cross_corr]

            # Logs the user's response time
            outcome = np.random.randint(2)

            if outcome == 1:
                # If accepted, set the variable's inclusion indicator to true.
                if act_in not in included_vars:
                    included_vars.append(act_in)
                included_vars = sorted(included_vars)

            elif outcome == 0:
                # If not, set it to false.
                if act_in in included_vars:
                    included_vars.remove(act_in)
            else:
                raise ValueError

            # Add the observation to dataset. So this observations are like: (corr, cross_corr), outcome.
            data_dict["x"].append(action)
            data_dict["y"].append(outcome)

        # This is an educate action!
        else:
            # The dummy action observation vector for educate actions. This is needed to ignore these in Stan.
            action = [-1.0, -1.0]

            # Dummy outcome. Not used for anything.
            outcome = 0

            data_dict["x"].append(action)
            data_dict["y"].append(outcome)

        # After the action execution, and observation of outcome, update our model of the user. The fit gives us the current model.
        data_dict["N"] += 1
        fit, user_model_file = fit_model_w_education(data_dict)


def main(group_id=2):


    n_data_points = 100
    n_test_dataset = 10

    n_collinear = 2  # difficulty[t][0]
    n_noncollinear = 6  # difficulty[t][1]

    # FORGET ABOUT THESE ????
    W_typezero = (7.0, 0.0)
    W_typeone = (7.0, -7.0)

    educability = 0.30
    init_var_cost = 0.05
    init_edu_cost = 0.5

    n_interactions = 20

    training_dataset = generate_data(n_noncollinear=n_noncollinear,
                                     n_collinear=n_collinear,
                                     n=n_data_points)

    training_X, training_y, test_X, test_y, _, _ = training_dataset

    if group_id == 0:
        return

    # ---------------- only for group 1 and 2 ------------------- #
    if group_id == 1:
        planning_function = noeducate_rollout_onestep_la

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

    ts_teach_user_study(
        dataset=training_dataset,
        test_datasets=test_datasets,
        planning_function=planning_function,
        educability=educability,
        W_typezero=W_typezero,
        W_typeone=W_typeone,
        init_var_cost=init_var_cost,
        init_edu_cost=init_edu_cost,
        n_interactions=n_interactions,
        n_training_collinear=n_collinear,
        n_training_noncollinear=n_noncollinear)


if __name__ == "__main__":
    main()