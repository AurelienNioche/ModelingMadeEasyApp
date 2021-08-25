import torch
import numpy as np


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
