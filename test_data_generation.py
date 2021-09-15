from task.dataset.generate_data import generate_data
from task.config import config

kwargs_data = dict(
        n_noncollinear=config.N_NONCOLLINEAR,
        n_collinear=config.N_COLLINEAR,
        n=config.N_DATA_POINTS,
        std_collinear=config.STD_COLLINEAR,
        std_noncollinear=config.STD_NONCOLLINEAR,
        noise_collinear=config.NOISE_COLLINEAR,
        coeff_collinear=config.COEFF_COLLINEAR,
        coeff_noncollinear=config.COEFF_NONCOLLINEAR,
        coeff_intercept=config.COEFF_INTERCEPT,
        phi=config.PHI,
        seed=config.TRAINING_DATASET_SEED)

training_dataset = generate_data(**kwargs_data)