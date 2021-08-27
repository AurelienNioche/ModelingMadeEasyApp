import numpy as np

# Create your tests here.
from task.ai.ai import AiAssistant
from task.dataset.generate_data import generate_data
from task.ai.planning.rollout_one_step_la import rollout_onestep_la
from task.ai.planning.no_educate_rollout_one_step_la import no_educate_rollout_one_step_la

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

    ai = AiAssistant(
        dataset=training_dataset,
        test_datasets=test_datasets,
        planning_function=planning_function,
        stan_compiled_model_file="task/ai/stan_model/mixture_model_w_ed.pkl",
        educability=educability,
        W_typezero=W_typezero,
        W_typeone=W_typeone,
        init_var_cost=init_var_cost,
        init_edu_cost=init_edu_cost,
        n_interactions=n_interactions,
        n_training_collinear=n_collinear,
        n_training_noncollinear=n_noncollinear)

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
