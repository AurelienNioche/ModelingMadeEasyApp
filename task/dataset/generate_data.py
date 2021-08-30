import torch
from torch.distributions import Normal, MultivariateNormal


def generate_data(n_collinear, n_noncollinear, n,
                  std_collinear, std_noncollinear,
                  noise_collinear,
                  coeff_collinear,
                  coeff_noncollinear,
                  coeff_intercept,
                  phi):

    coeffs = torch.zeros(n_collinear + n_noncollinear)
    coeffs[:n_collinear] = coeff_collinear
    coeffs[n_collinear:] = coeff_noncollinear

    x_colin = Normal(0, std_collinear).sample((n, n_collinear))
    x_colin = x_colin.squeeze()

    x_colin_test = Normal(0, std_collinear).sample((n, n_collinear))
    x_colin_test = x_colin_test.squeeze()

    for i in range(n_collinear):
        z = Normal(0.0, noise_collinear).sample((n, ))
        x_colin[:, i] += z.squeeze()
        x_colin_test[:, i] += z.squeeze()

    # Generate non co-linear covariates by sampling them independently
    x_noncolin = Normal(0.0, std_noncollinear).sample((n, n_noncollinear))
    x_noncolin = x_noncolin.squeeze()

    design_matrix = torch.cat((x_colin, x_noncolin), dim=1)

    # Generate outputs
    covar_matrix = phi * torch.eye(n)

    reg_term = design_matrix @ coeffs

    output_rv = MultivariateNormal(
        torch.ones(n) * coeff_intercept + reg_term, covar_matrix).sample()

    return design_matrix, output_rv
