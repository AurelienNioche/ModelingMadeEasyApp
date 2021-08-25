import torch
from torch.distributions import Normal, MultivariateNormal


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
