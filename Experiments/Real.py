import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

import Data.generate_data as generate_data
import models.util as util

# This code does the following:
# Load the preprocessed real data
# Fit the model using Variational inference approach
# We use checkpoints in this and saved to folder name tf_ckpts

# n_stores = number of stores to sample.  Must be positive or use 'All' to use all full dataset
# n_individuals = number of customers to sample.  Must be positive or use 'All' to use all full dataset
# truncate = radius of the Gaussian to truncate. Possible values = {5:25km; 4:20Km; 3: 15Km;2 :10Km;1: 5Km}
# edge_correct = Use revenue after edge corrections
# n_s_features = number of store features
# n_features = number of customer features. Must be positive and less than 5 as theres 5 different spatial correlation structures
# ind_features = Individual features to use in the model. possible values = ['population', 'male_prop', 'income', 'employment','education','health', 'crime','housing', 'living']
# seed = random seed for selecting the location of stores
# epochs = A scalar total number of epochs to be optimized for. Epochs are complete passes over the data.
# learning_rate = A scalar float32 or float64 Tensor or a Python number. The initial learning rate.
# decay_steps =	A scalar int32 or int64 Tensor or a Python number. Must be positive.

# Select to metod to run the BSIM. Possible values:
# -- BSIM_SF_SC :  Store variace as a function of the Store Features plus store specific unobserverd factor
# -- BSIM_SC :  Store variace as a store specific coeficient


method = 'BSIM_SC'

if method == 'BSIM_SF_SC':
    import models.BSIM_SF_SC as BSIM
elif method == 'BSIM_SC':
    import models.BSIM_SC as BSIM

n_stores =  10 # Use 'All' to use all the pubs
n_individuals = 1000 # Use 'All' to use all the customer regions 
truncate =  4.
edge_correct = True
ind_features = ['population', 'male_prop', 'income', 'employment','education','health', 'crime','housing', 'living']

##### Generating Data
Y,locations_N,locations_S,features, store_features = generate_data.real_Data(n_stores, n_individuals,ind_features,edge_correct,truncate)

# Define the inputs for training
n_individuals = len(locations_N)
n_stores = len(locations_S)
s_features =  store_features.shape[1]
n_features = features.shape[0]

epochs = 10000
learning_rate =  0.05
decay_steps = 100

print('Ready to train')

######### Training
model= BSIM.BSIM(n_individuals,n_stores,n_features, s_features, truncate)

nelbo_vector, time_tensor, beta_mu_vec, beta_var_vec,lambda_mu_vec,lambda_var_vec, lambda_s_mu_vec,lambda_s_var_vec,sigma_y_vec,y_gamma_vec,\
opt_beta_mu, opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_lambda_s_mu,opt_lambda_s_var, opt_beta_gamma_a,opt_beta_gamma_b,opt_y_gamma_a,opt_y_gamma_b,beta_gamma_vec = model.fit(locations_N,locations_S, features,store_features,Y,
                            epochs=epochs,display_step_nelbo =100, learning_rate = learning_rate, decay_steps = decay_steps)

# Save Optimum results
opt_results = {'Y':Y,'locations_N':locations_N, 'locations_S':locations_S,'features':features,'store_features': store_features,'opt_beta_mu':opt_beta_mu,
    'opt_beta_var' : opt_beta_var,'opt_lambda_mu':opt_lambda_mu,'opt_lambda_var':opt_lambda_var,'opt_lambda_s_mu':opt_lambda_s_mu,
    'opt_lambda_s_var':opt_lambda_s_var,'opt_y_gamma_a':opt_y_gamma_a,'opt_y_gamma_b':opt_y_gamma_b}
np.save('opt_results_s10_SF.npy', opt_results)

######### Predictions
y_est, n_rev_array = model.predict(locations_N,locations_S,features,store_features,opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_lambda_s_mu,opt_lambda_s_var,opt_y_gamma_a,opt_y_gamma_b)


#Collecting all the stats of the optimal paramerters
post_estimates = util.post_process_estimates(opt_beta_mu,opt_beta_var,opt_lambda_mu,opt_lambda_var,opt_y_gamma_a, opt_y_gamma_b)
pd.DataFrame(post_estimates)

perfomance_matrix = util.post_process_results(y_est,Y)
pd.DataFrame(perfomance_matrix)


#Plots to show training and optimum results
fig = util.plot_nelbo(nelbo_vector)
fig.savefig('epochs.png')

fig = util.plot_lambda_real(lambda_mu_vec,lambda_var_vec,opt_lambda_mu,lambda_s_mu_vec,opt_lambda_s_mu,store_features,s_features)
fig.savefig('lambda.png')

fig = util.plot_sumary(Y,y_est)
fig.savefig('y_results.png')


fig = util.plot_beta(beta_mu_vec,beta_var_vec,n_features )
fig.savefig('beta.png')

fig =  util.plot_simga(sigma_y_vec)
fig.savefig('sigma.png')
