import os
import numpy as np
import torch
import pandas
import pystan
import pickle


class AiAssistant:

    EDUCATE = -1
    RECOMMEND = 1

    def __init__(self,
                 planning_function,
                 training_X, training_y, test_datasets,
                 W_typezero, W_typeone, educability,
                 n_training_collinear,
                 n_training_noncollinear,
                 n_interactions,
                 init_var_cost,
                 init_educ_cost,
                 user_model_file="mixture_model_w_ed.pkl"):

        self.user_model_file = user_model_file

        self.educability = educability
        self.n_interactions = n_interactions

        self.n_covars = training_X.shape[1]
        self.training_X = training_X
        self.training_y = training_y

        self.test_datasets = test_datasets

        self.n_training_collinear = n_training_collinear
        self.n_training_noncollinear = n_training_noncollinear

        self.data_dict = {
            "N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone],
            "educability": educability,
            "forgetting": 0.0}

        # xi is n_covars + 1 because its size must match the corr_mat which includes the output y.
        self.aux_data_dict = {"xi": torch.zeros(self.n_covars + 1, dtype=torch.bool)}

        self.corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((self.training_X, self.training_y.unsqueeze(dim=1)), dim=1), 0, 1)))

        self.i = 0

        # cost[0,...,n_covars-1] is recommendation cost per covariate. cost[n_covars] is educate cost.
        self.cost = torch.zeros(self.n_covars + 1) + init_var_cost
        self.cost[-1] = init_educ_cost

        self.fit = None
        self.prev_action = None
        self.prev_max_cor = None

        self.planning_function = planning_function

    def get_recommendation(self, included_vars):

        """
        User model estimation part. Uses Certainty Equivalence to replace betas with their expectation, and posterior sampling to sample a user type.
        """

        # Special case for first action
        if self.i == 0:
            # First action must be recommend for numerical purposes
            is_var_rec = True
            # Uniform choice of covariate
            r_i = np.random.randint(self.n_covars)

        else:
            # Indices for grabbing the necessary statistics
            # from Stan fit object
            strt = 6 + (3 * self.i)
            endn = strt + self.i

            s = self.fit.summary()
            summary = pandas.DataFrame(s['summary'],
                                       columns=s['summary_colnames'],
                                       index=s['summary_rownames'])
            betas_mean = list(summary.iloc[2:6, 0])
            betas_mean[1], betas_mean[2] = betas_mean[2], betas_mean[1]
            # print(summary.iloc[[strt, endn], 0])
            # E[\alpha_0] and E[\alpha_1],
            # posterior expectations of type-0 and type-1 probabilities
            type_probs_mean = list(summary.iloc[[strt, endn], 0])
            sample_user_type = np.random.choice(2, p=type_probs_mean)

            sample_user_model = (
                sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2],
                betas_mean[3], self.educability)

            # Returns educate OR recommend and if recommend,
            # recommended covar's index. IF not, r_i is None.
            is_var_rec, r_i = self.planning_function(
                self.n_covars,
                self.aux_data_dict["xi"],
                self.corr_mat,
                sample_user_model,
                self.cost,
                self.n_interactions - self.i + 1,
                self.n_training_collinear,
                self.n_training_noncollinear,
                self.test_datasets,
                prev_action=self.prev_action)

        if is_var_rec:
            # Action indices of recommend actions are the index of
            # the recommended covariate \in {0,...,n_covars-1}
            act_in = r_i

            rec_item_cor, max_cross_corr = self.select_var_to_cor(
                act_in=act_in, included_vars=included_vars)

            self.prev_action = r_i
            self.prev_max_cor = max_cross_corr

        else:
            act_in = AiAssistant.EDUCATE
            rec_item_cor = None

        return act_in, rec_item_cor

    def select_var_to_cor(self, act_in, included_vars):

        if len(included_vars) == 0:
            # by default, correlate with itself
            return act_in, 0.0

        # Get the currently selected variables as boolean mask
        mask = np.ones(len(included_vars))

        # Set the recommended variable to False.
        # Otherwise cross-correlations will include self-correlation.
        mask[act_in] = False

        # Get the cross-correlations between recommended var and included vars.
        masked = np.arange(self.n_covars)[mask]
        values = self.corr_mat[act_in, mask]
        to_cor = masked[np.argmax(values)]
        return to_cor, np.max(values)

    def user_feedback(self, outcome: int = 0):

        # If the chosen action was to recommend
        if self.prev_action != AiAssistant.EDUCATE:
            # Generate the action's observation vector for the user:
            # (corr, cross_corr).
            print("prev action", self.prev_action)
            print("cor dim", self.corr_mat.shape)
            print("cor", self.corr_mat[self.prev_action, -1])
            print("prev_max_cor", self.prev_max_cor)
            action = [self.corr_mat[self.prev_action, -1], self.prev_max_cor]

        # This is an educate action!
        else:
            # Dummy action and outcome. Not used for anything.
            action = [-1.0, -1.0]

        print("i", self.i)
        print(action[0])
        print(action[1])

        # Add the observation to dataset.
        # So this observations are like: (corr, cross_corr), outcome.
        self.data_dict["x"].append([float(x) for x in action])
        self.data_dict["y"].append(outcome)

        self.data_dict["N"] += 1
        self.fit = self.fit_model_w_education(self.data_dict)

        self.i += 1

    def fit_model_w_education(self, data):
        """
        data must be a dict containing --> N: number of datapoints, x: 2-d list of xs, y: 1-d list of ys, beta: 2-d list of weight vectors
        If model_file is None, will compile. If not, use pre-compiled model from the file.
        """
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
        file_path = os.path.abspath(self.user_model_file)
        if not os.path.exists(file_path):
            sm = pystan.StanModel(model_code=mixture_with_tseries_model)
            fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)
            with open(file_path, "wb") as f:
                pickle.dump(sm, f)

            return fit
        else:

            with open(file_path, "rb") as f:
                sm = pickle.load(f)
                fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)

            return fit
