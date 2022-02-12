import sys
sys.path.append("..")

import numpy as np
import pandas as pd

import Data.generate_data as generate_data
import models.BSIM_SF as BSIM
import models.util as util

# This code does the following:
# Generate a syntetic dataset.
# Fit the model using Variational inference approach
# We use checkpoints in this and saved to folder name tf_ckpts_sf. make sure to delete this folder before you run a new experiment

# n_stores = number of stores.  Must be positive.
# n_individuals = number of customers. Must be positive.
# truncate = radius of the Gaussian to truncate (cordinates are between 0-10)
# sigma_y_actual = actual value for error term
# beta = actual beta parameters
# n_s_features = number of store features
# n_features = number of customer features. Must be positive and less than 5 as theres 5 different spatial correlation structures
# seed = random seed for selecting the location of stores
# epochs = A scalar total number of epochs to be optimized for. Epochs are complete passes over the data.
# learning_rate = A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
# decay_steps =	A scalar int32 or int64 Tensor or a Python number. Must be positive.

n_stores =  10
n_individuals = 1000
truncate =  4.
s_features = 2
sigma_y_act = 0.5
beta_act = np.array([-0.2,0.4])[:,np.newaxis]
lambda_act = np.array([0.1, 0.5]).reshape(s_features,1)


seed = 49

##### Generating Data
Y,locations_N,locations_S,features,var_s_actual,sigma_y_actual,store_features,act_lambda,act_pdf_array = generate_data.generate_synthetic_data(n_individuals, 2, n_stores, 2, sigma_y_act,truncate,beta_act,lambda_act,seed)

n_features = features.shape[0]

print('Ready to train')

# var set need to be at least one with epochs > 0!!!
epochs = 10000
learning_rate =  0.05
decay_steps = 100


######### TRAINING
model = BSIM.BSIM(n_individuals,n_stores,n_features, s_features, truncate)

nelbo_vector, time_tensor, beta_mu_vec, beta_var_vec,lambda_mu_vec,lambda_var_vec, lambda_s_mu_vec,lambda_s_var_vec,sigma_y_vec,y_gamma_vec,\
opt_beta_mu, opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_lambda_s_mu,opt_lambda_s_var, opt_beta_gamma_a,opt_beta_gamma_b,opt_y_gamma_a,opt_y_gamma_b,beta_gamma_vec = model.fit(locations_N,locations_S, features,store_features,Y,epochs=epochs,display_step_nelbo =100, learning_rate = learning_rate, decay_steps = decay_steps)

######### Predictions
y_est, n_rev_array = model.predict(locations_N,locations_S,features,store_features,opt_beta_mu,opt_beta_var, opt_lambda_mu,opt_lambda_var,opt_lambda_s_mu,opt_lambda_s_var,opt_y_gamma_a,opt_y_gamma_b)

### Process results

post_estimates = util.post_process_estimates(opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_y_gamma_a, opt_y_gamma_b,beta_act, lambda_act, sigma_y_act)
pd.DataFrame(post_estimates)

perfomance_matrix = util.post_process_results(y_est,Y)
pd.DataFrame(perfomance_matrix)


#PLOTS
# %matplotlib inline

fig = util.plot_nelbo(nelbo_vector)
fig.savefig( 'epochs.png')

fig = util.plot_lambda_synth(lambda_mu_vec,lambda_var_vec,store_features, opt_lambda_mu, var_s_actual,s_features,lambda_act)
fig.savefig('lambda.png')

fig = util.plot_sumary(Y,y_est)
fig.savefig('y_results.png')

fig = util.plot_beta(beta_mu_vec,beta_var_vec,n_features,beta_act)
fig.savefig('beta.png')

fig = util.plot_simga(sigma_y_vec,sigma_y_actual)
fig.savefig('sigma.png')
