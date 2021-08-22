import logging
logger = logging.getLogger("pystan")

# add root logger (logger Level always Warning)
# not needed if PyStan already imported
logger.addHandler(logging.NullHandler())

logger_path = "pystan.log"
fh = logging.FileHandler(logger_path, encoding="utf-8")
fh.setLevel(logging.INFO)
# optional step
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

import stan
from matplotlib import pyplot
import arviz
import pickle


def fit_model_w_education(data):

    #data must be a dict containing --> N: number of datapoints, x: 2-d list of xs, y: 1-d list of ys, beta: 2-d list of weight vectors
    #If model_file is None, will compile. If not, use pre-compiled model from the file.
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

    # if model_file is None:
    sm = stan.build(program_code=mixture_with_tseries_model, data=data)
    fit = sm.sample(num_samples=1000, num_chains=4)
    return fit, None
    # with open(model_file, "wb") as f:
    #     pickle.dump(sm, f)
    #
    # return fit, model_file
    # else:
    #
    #     with open(model_file, "rb") as f:
    #         sm = pickle.load(f)
    #         fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)
    #
    #     return fit, model_file





