import pandas
import torch
import numpy as np
import time
import os
import logging
# logger = logging.getLogger("pystan")
#
# # add root logger (logger Level always Warning)
# # not needed if PyStan already imported
# logger.addHandler(logging.NullHandler())
#
# logger_path = "pystan.log"
# fh = logging.FileHandler(logger_path, encoding="utf-8")
# fh.setLevel(logging.INFO)
# # optional step
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

import pystan
from matplotlib import pyplot
import arviz
import pickle

from task.dataset.generate_data import generate_data

from task.ai.planning.rollout_one_step_la import rollout_onestep_la
from task.ai.planning import no_educate_rollout_one_step_la
from task.ai.planning.baselines.utils import terminal_cost


def fit_model_w_education(data, model_file=None):
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

    if model_file is None:
        model_file = "mixture_model_w_ed.pkl"
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


# Posterior Sampling Machine Education of Switching Learners
def ts_teach_user_study(dataset, planning_function,
                        test_datasets,
                        user_model_file,
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
        fit, user_model_file = fit_model_w_education(data_dict, user_model_file)


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
        planning_function = no_educate_rollout_one_step_la

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
        user_model_file="mixture_model_w_ed.pkl",
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