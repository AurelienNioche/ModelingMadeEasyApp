import numpy as np
import torch
import pandas

from task.ai.stan_model import stan_model


class AiAssistant:

    EDUCATE = -1

    def __init__(
            self,
            dataset,
            planning_function,
            test_datasets,
            stan_compiled_model_file,
            cost_var,
            cost_edu,
            theta_1,
            theta_2,
            heuristic_n_samples,
            user_switch_sim_a,
            terminal_cost_err_mlt,
            educability,
            forgetting,
            n_interactions,
            n_collinear,
            n_noncollinear):

        training_X, training_y = dataset

        self.corr_mat = np.abs(np.corrcoef(torch.transpose(
            torch.cat((training_X, training_y.unsqueeze(dim=1)), dim=1), 0,
            1)))
        self.n_covars = training_X.shape[1]
        self.data_dict = {
            "N": 0, "x": [], "y": [],
            "educability": educability,
            "forgetting": forgetting}

        # cost[0,...,n_covars-1] is recommendation cost per covariate.
        # cost[n_covars] is educate cost.
        self.cost = torch.zeros(self.n_covars + 1) + cost_var
        self.cost[-1] = cost_edu

        self.prev_action = None

        self.n_interactions = n_interactions
        self.educability = educability
        self.n_collinear = n_collinear
        self.n_noncollinear = n_noncollinear
        self.test_datasets = test_datasets
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.heuristic_n_samples = heuristic_n_samples
        self.terminal_cost_err_mlt = terminal_cost_err_mlt
        self.user_switch_sim_a = user_switch_sim_a
        self.planning_function = planning_function

        self.stan_compiled_model_file = stan_compiled_model_file

        self.fit_summary = None

        self.i = 0

    def act(self, included_vars):

        # Indices for grabbing the necessary statistics from Stan fit object
        strt = 6 + (3 * self.i)
        endn = strt + self.i

        if self.i == 0:
            is_rec_var = 1
            r_i = np.random.choice(self.n_covars, 1).item()

        else:
            betas_mean = list(self.fit_summary.iloc[2:6, 0])
            betas_mean[1], betas_mean[2] = betas_mean[2], betas_mean[1]
            # print(summary.iloc[[strt, endn], 0])
            # E[\alpha_0] and E[\alpha_1],
            # posterior expectations of type-0 and type-1 probabilities
            type_probs_mean = list(self.fit_summary.iloc[[strt, endn], 0])
            sample_user_type = np.random.choice(2, p=type_probs_mean)

            sample_user_model = (
                sample_user_type, betas_mean[0], betas_mean[1], betas_mean[2],
                betas_mean[3], self.educability)

            # Returns educate OR recommend and if recommend,
            # recommended covar's index. IF not, r_i is None.
            # xi is n_covars + 1 because its size must
            # match the corr_mat which includes the output y.
            aux_data_dict = torch.zeros(self.n_covars + 1, dtype=torch.bool)
            aux_data_dict[included_vars] = 1
            is_rec_var, r_i = self.planning_function(
                xi=aux_data_dict,
                corr_mat=self.corr_mat,
                sample_user_model=sample_user_model,
                cost=self.cost,
                time_to_go=self.n_interactions - self.i + 1,
                n_collinear=self.n_collinear,
                n_noncollinear=self.n_noncollinear,
                test_datasets=self.test_datasets,
                prev_action=self.prev_action,
                theta_1=self.theta_1,
                theta_2=self.theta_2,
                heuristic_n_samples=self.heuristic_n_samples,
                user_switch_sim_a=self.user_switch_sim_a,
                terminal_cost_err_mlt=self.terminal_cost_err_mlt)

        # if recommendation a variable
        if is_rec_var:
            act_in = r_i
            to_cor, _ = self._select_to_cor_and_get_max_cor(
                act_in, included_vars)

        # if *educate*
        else:
            act_in = AiAssistant.EDUCATE
            to_cor = None

        return act_in, to_cor

    def update(self, act_in, included_vars):

        # If that was an educate action
        if act_in == AiAssistant.EDUCATE:
            # The dummy action observation vector for educate actions.
            # This is needed to ignore these in Stan.
            action = [-1.0, -1.0]

            # Dummy outcome. Not used for anything.
            outcome = 0

        else:
            _, max_cross_cor = self._select_to_cor_and_get_max_cor(
                act_in, included_vars)

            # Generate the action's observation vector for the user:
            # (corr, cross_corr).
            action = [self.corr_mat[act_in, -1], max_cross_cor]

            # Logs the user's response time
            outcome = int(act_in in included_vars)  # np.random.randint(2)

            # we update prev_action only if not educate
            self.prev_action = act_in

        # Add the observation to dataset.
        # So this observations are like: (corr, cross_corr), outcome.
        self.data_dict["x"].append(action)
        self.data_dict["y"].append(outcome)

        # After the action execution, and observation of outcome,
        # update our model of the user. The fit gives us the current model.
        self.data_dict["N"] += 1
        fit = stan_model.fit_model_w_education(
            self.data_dict,
            self.stan_compiled_model_file)

        # User model estimation part.
        # Uses Certainty Equivalence to replace betas with their expectation,
        # and posterior sampling to sample a user type.
        s = fit.summary()
        self.fit_summary = pandas.DataFrame(
            s['summary'],
            columns=s['summary_colnames'],
            index=s['summary_rownames'])

        self.i += 1

    def _select_to_cor_and_get_max_cor(self, act_in, included_vars):
        # Get the currently selected variables as boolean mask
        mask = torch.zeros(self.n_covars + 1, dtype=bool)
        mask[included_vars] = True

        # Set the recommended variable to False.
        # Otherwise cross-correlations will include self-correlation.
        mask[act_in] = False

        # Get the cross-correlations between recommended var and included vars.
        masked = self.corr_mat[act_in, mask]

        # If there are more than one variables included
        if masked.size != 0:
            # The maximum absolute cross-correlation to selected vars.
            max_cross_corr = np.max(masked)
            to_cor = np.arange(self.n_covars + 1)[mask][np.argmax(masked)]

        else:
            # Set to zero since there are no vars to cross-correlate to
            max_cross_corr = 0.0
            to_cor = act_in

        return to_cor, max_cross_corr
