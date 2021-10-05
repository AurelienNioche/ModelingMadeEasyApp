import numpy as np

# Create your tests here.
from task.ai.ai import AiAssistant
from task.dataset.generate_data import generate_data
from task.ai.planning.rollout_one_step_la import rollout_one_step_la
from task.ai.planning.no_educate_rollout_one_step_la import no_educate_rollout_one_step_la


def main():

    group_id = 2

    n_data_points = 100
    n_test_dataset = 10

    n_collinear = 2
    n_noncollinear = 6

    std_collinear = 1.0
    std_noncollinear = 1.0
    noise_collinear = 0.01
    coeff_intercept = 1.0
    coeff_collinear = 0.1
    coeff_noncollinear = 1.0
    phi = 0.10

    theta_1 = 1.0
    theta_2 = 1.0

    educability = 0.30
    forgetting = 0.00

    init_var_cost = 0.05
    init_edu_cost = 0.5

    terminal_cost_err_mlt = 5
    user_switch_sim_a = 1.0
    heuristic_n_samples = 10

    n_interactions = 20
    
    base_seed = 1000

    kwargs_data = dict(
        n_noncollinear=n_noncollinear,
        n_collinear=n_collinear,
        n=n_data_points,
        std_collinear=std_collinear,
        std_noncollinear=std_noncollinear,
        noise_collinear=noise_collinear,
        coeff_collinear=coeff_collinear,
        coeff_noncollinear=coeff_noncollinear,
        coeff_intercept=coeff_intercept,
        phi=phi)
    training_dataset = generate_data(seed=base_seed, **kwargs_data)

    if group_id == 0:
        return

    # ---------------- only for group 1 and 2 ------------------- #

    if group_id == 1:
        planning_function = no_educate_rollout_one_step_la

    elif group_id == 2:
        planning_function = rollout_one_step_la
    else:
        raise ValueError(f"Group id incorrect: {group_id}")

    print("generating test data sets...")
    test_datasets = [generate_data(seed=base_seed + i, **kwargs_data)
                     for i in range(n_test_dataset)]
    test_datasets.append(training_dataset)

    ai = AiAssistant(
        dataset=training_dataset,
        test_datasets=test_datasets,
        planning_function=planning_function,
        stan_compiled_model_file="task/ai/stan_model/mixture_model_w_ed.pkl",
        educability=educability,
        forgetting=forgetting,
        cost_var=init_var_cost,
        cost_edu=init_edu_cost,
        n_interactions=n_interactions,
        n_collinear=n_collinear,
        n_noncollinear=n_noncollinear,
        theta_1=theta_1,
        theta_2=theta_2,
        heuristic_n_samples=heuristic_n_samples,
        user_switch_sim_a=user_switch_sim_a,
        terminal_cost_err_mlt=terminal_cost_err_mlt)

    included_vars = []
    previous_var = None
    previous_include = None
    for i in range(n_interactions):

        print("previous var", previous_var, "previous include", previous_include)

        var, _ = ai.act(included_vars)

        if var == AiAssistant.EDUCATE:
            include = None

        else:
            include = np.random.randint(2)
            if include == 1:
                if var not in included_vars:
                    included_vars.append(var)
                included_vars = sorted(included_vars)

            elif include == 0:
                if var in included_vars:
                    included_vars.remove(var)

        ai.update(var, included_vars)

        previous_include = include
        previous_var = var


if __name__ == "__main__":
    main()
