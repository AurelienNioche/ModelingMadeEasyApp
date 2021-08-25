import numpy as np
import torch
import pandas
import time
import pystan
import pickle


class AiAssistant:

    EDUCATE = -1

    def __init__(self,
                 planning_function,
                 training_X, training_y, test_datasets,
                 W_typezero, W_typeone, educability,
                 n_training_collinear,
                 n_training_noncollinear,
                 n_interactions,
                 init_cost,
                 init_last_cost,
                 user_model_file="mixture_model_w_ed.pkl"):

        self.user_model_file = user_model_file

        self.educability = educability
        self.n_interactions = n_interactions

        self.n_covars = training_X.shape[1]
        self.training_X = training_X
        self.training_y = training_y

        self.test_datasets = test_datasets

        self.data_dict = {
            "N": 0, "x": [], "y": [], "beta": [W_typezero, W_typeone],
            "educability": educability,
            "forgetting": 0.0}

        # xi is n_covars + 1 because its size must match the corr_mat which includes the output y.
        self.aux_data_dict = {"xi": torch.zeros(self.n_covars + 1, dtype=torch.bool)}

        self.corr_mat = np.abs(np.corrcoef(torch.transpose(torch.cat((self.training_X, self.training_y.unsqueeze(dim=1)), dim=1), 0, 1)))

        self.i = 0
        self.task_start_time = time.time()

        self.educate_or_recommend = []
        self.recommend_actions = []

        # cost[0,...,n_covars-1] is recommendation cost per covariate. cost[n_covars] is educate cost.
        self.cost = torch.zeros(self.n_covars + 1) + init_cost
        self.cost[-1] = init_last_cost
        self.cumulative_cost = np.zeros(self.n_interactions)

        self.fit = None
        self.prev_action = None

        self.planning_function = planning_function

        self.n_training_collinear = n_training_collinear
        self.n_training_noncollinear = n_training_noncollinear

        self.sampled_user_models = []
        self.user_response_times = []
        self.user_responses = []
        self.user_response_string = []

        self.user_response_legend = ["exclude", "include"]

        # For CSV
        self.model_built_sofar = []
        self.is_incorrect_action = []
        self.user_response_string = []
        self.system_action_string = []
        self.task_time_list = []

    def get_first_recommendation(self):
        # First action must be recommend for numerical purposes
        self.educate_or_recommend.append(1)
        # Uniform choice of covariate
        self.recommend_actions.append(np.random.choice(self.n_covars, 1).item())
        # Action index is the index of the latest recommended covariate
        act_in = self.recommend_actions[-1]

        self.system_action_string.append("Recommend: {}".format(act_in))
        self.cumulative_cost[self.i] += self.cost[act_in]
        return act_in

    def get_recommendation(self):

        """
        User model estimation part. Uses Certainty Equivalence to replace betas with their expectation, and posterior sampling to sample a user type.
        """

        # Special case for first action
        if self.i == 0:
            return self.get_first_recommendation()

        # Indices for grabbing the necessary statistics from Stan fit object
        strt = 6 + (3 * self.i)
        endn = strt + self.i

        s = self.fit.summary()
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
            betas_mean[3], self.educability)

        self.sampled_user_models.append(sample_user_model)

        # Returns educate OR recommend and if recommend, recommended covar's index. IF not, r_i is None.
        e_or_r, r_i = self.planning_function(
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
        self.prev_action = r_i

        # If *educate*
        if e_or_r == 0:
            self.educate_or_recommend.append(e_or_r)
            act_in = self.EDUCATE
            self.cumulative_cost[self.i] = self.cumulative_cost[self.i - 1] + self.cost[-1]
            self.system_action_string.append("Tutor")

        else:
            # Action indices of recommend actions are the index of the recommended covariate \in {0,...,n_covars-1}
            self.educate_or_recommend.append(e_or_r)
            self.recommend_actions.append(r_i)
            act_in = self.recommend_actions[-1]
            self.system_action_string.append("Recommend: {}".format(act_in))
            self.cumulative_cost[self.i] = self.cumulative_cost[self.i - 1] + self.cost[
                self.recommend_actions[-1]]
        return act_in

    def post_var_recommend(self, accept):

        # Indices for grabbing the necessary statistics from Stan fit object
        included_vars = torch.where(self.aux_data_dict["xi"])[0].tolist()

        act_in = self.prev_action

        # if act_in in included_vars:
        #     print(
        #         "Would you like to keep variable X{} in the model?".format(
        #             act_in))
        # else:
        #     print(
        #         "Would you like to include variable X{} in the model?".format(
        #             act_in))

        # X_in = self.training_X[:, act_in]
        # X_name = 'X' + str(act_in)
        #
        # d = {X_name: X_in, 'y': self.training_y}
        # df = pandas.DataFrame(data=d)
        #
        # self.ax1.cla()
        # sns.scatterplot(ax=self.ax1, x=X_name, y="y", data=df)

        # Get the currently selected variables as boolean mask
        mask = self.aux_data_dict["xi"].numpy().copy()

        # Set the recommended variable to False. Otherwise cross-correlations will include self-correlation.
        mask[act_in] = False

        # Get the cross-correlations between recommended var and included vars.
        masked = self.corr_mat[act_in, mask]

        # If there are more than one variables included
        if masked.size != 0:
            # The maximum absolute cross-correlation to selected vars.
            max_cross_corr = np.max(masked)
            to_corr = [x for x in np.argsort(self.corr_mat[act_in, :]) if mask[x]][-1]

            # d = {X_name: X_in, 'X' + str(j): self.training_X[:, j]}
            # df = pandas.DataFrame(data=d)
            #
            # sns.scatterplot(ax=self.ax2, x=X_name, y='X' + str(j), data=df)

        else:
            # Set to zero since there are no vars to cross-correlate to
            max_cross_corr = 0.0
            to_corr = None

        # Generate the action's observation vector for the user: (corr, cross_corr).
        action = torch.tensor([self.corr_mat[act_in, -1], max_cross_corr])

        # Logs the user's response time
        start_time = time.time()
        # is_valid_response = False
        outcome = int(accept)
        # # Get user input. Repeat if user enters invalid value.
        # while not is_valid_response:
        #     outcome = input("Type 1 for yes, 0 for no: ")
        #     if str(outcome).lower() in ['y', 'yes']: outcome = '1'
        #     if str(outcome).lower() in ['n', 'no']: outcome = '0'
        #     if outcome in ['0', '1']:
        #         outcome = int(outcome)
        #         is_valid_response = True
        #     else:
        #         print("Error! Invalid input.")

        self.user_response_times.append(time.time() - start_time)
        self.user_responses.append(outcome)
        self.user_response_string.append(self.user_response_legend[outcome])

        previous_xi_n_collinear = torch.sum(
            self.aux_data_dict["xi"][0:self.n_training_collinear])
        previous_xi_n_noncollinear = torch.sum(
            self.aux_data_dict["xi"][self.n_training_collinear:-1])
        previous_terminal_cost = (np.abs(previous_xi_n_collinear - 1)) + (
            np.abs(self.n_training_noncollinear - previous_xi_n_noncollinear))

        if outcome == 1:
            # If accepted, set the variable's inclusion indicator to true.
            self.aux_data_dict["xi"][act_in] = True
            new_xi_n_collinear = torch.sum(
                self.aux_data_dict["xi"][0:self.n_training_collinear])
            new_xi_n_noncollinear = torch.sum(
                self.aux_data_dict["xi"][self.n_training_collinear:-1])
            new_terminal_cost = (np.abs(new_xi_n_collinear - 1)) + (np.abs(
                self.n_training_noncollinear - new_xi_n_noncollinear))

            if (new_terminal_cost - previous_terminal_cost) <= 0:
                self.is_incorrect_action.append(False)
            else:
                self.is_incorrect_action.append(True)
        else:
            # If not, set it to false.
            self.aux_data_dict["xi"][act_in] = False
            new_xi_n_collinear = torch.sum(
                self.aux_data_dict["xi"][0:self.n_training_collinear])
            new_xi_n_noncollinear = torch.sum(
                self.aux_data_dict["xi"][self.n_training_collinear:-1])
            new_terminal_cost = (np.abs(new_xi_n_collinear - 1)) + (np.abs(
                self.n_training_noncollinear - new_xi_n_noncollinear))

            if (new_terminal_cost - previous_terminal_cost) <= 0:
                self.is_incorrect_action.append(False)
            else:
                self.is_incorrect_action.append(True)

        # Add the observation to dataset. So this observations are like: (corr, cross_corr), outcome.
        self.data_dict["x"].append(action.tolist())
        self.data_dict["y"].append(outcome)

    def post_educate(self):

        # input(
        #     "I would read up on Collinearity if I were you. Go ahead, I will wait. Press ENTER to show the tutorial.")
        #
        # os.system("start Material/Full_Tutorial.pdf")
        # start_time = time.time()
        # input("Press ENTER once you have completed the tutorial.")

        self.user_response_times.append(None)

        # The dummy action observation vector for educate actions. This is needed to ignore these in Stan.
        action = [-1.0, -1.0]

        # Dummy outcome. Not used for anything.
        outcome = 0

        self.is_incorrect_action.append(False)
        self.user_response_string.append(self.user_response_legend[outcome])

        self.data_dict["x"].append(action)
        self.data_dict["y"].append(outcome)

    def update(self):

        # After the action execution, and observation of outcome, update our model of the user. The fit gives us the current model.
        self.data_dict["N"] += 1
        self.fit, self.user_model_file = self.fit_model_w_education(
            self.data_dict,
            self.user_model_file)
        task_elapsed_time = time.time() - self.task_start_time
        self.task_time_list.append(task_elapsed_time)
        self.model_built_sofar.append(
            str(torch.where(self.aux_data_dict["xi"])[0].tolist()))

        self.i += 1

    # def get_recommendation(self):
    #
    #     # cls()   // REMOVE
    #     # plt.figure("X-Y")
    #     # plt.clf()
    #     # self.ax1.cla()
    #     # # plt.figure("X-X")
    #     # # plt.clf()
    #     # self.ax2.cla()
    #     # print("Step {} out of {}.".format(self.i + 1, self.n_interactions))
    #
    #     # print("Starting step", self.i + 1, "out of", self.n_interactions, "steps")
    #
    #     # if len(included_vars) == 0:
    #     #     print("No variables currently included in the model")
    #     # else:
    #     #     print("You already included the following variables:",
    #     #           ", ".join(["X{}".format(i) for i in included_vars]))
    #
    #     if self.i == 0:
    #         act_in = self.first_action()
    #
    #     else:
    #         act_in = self.action()
    #
    #     return act_in

    def feed_back_user(self, accept=None):

        # If the chosen action is to recommend
        if self.prev_action != self.EDUCATE:
            self.post_var_recommend(accept)

        # This is an educate action!
        else:
            self.post_educate()

        self.update()

    @staticmethod
    def fit_model_w_education(data, model_file=None):
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

        print(data)

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
