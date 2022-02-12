'
This file provides the sythetic data to run the MCMC.
You can create the synthetic data using the generate_data.py similar to the example dataset saved in this folder.  
'

#Data source directory 
data<-read.csv('data_10Stores.csv')

# Set the parameters to create synthetic data
opt_beta_mu = matrix(c(-0.2,0.4), nrow = 2, ncol = 1)
opt_lambda_mu <- matrix(c(0.1,0.5), nrow = 2, ncol = 1)

# er<-data[1:S,c("error")]

features<-data[1:N,c("features1","features2")]
features<-data.matrix(features)
locations_S<-data[1:S,c("locations_S1","locations_S2")]
locations_S<-data.matrix(locations_S)

locations_N<-data[1:N,c("locations_N1","locations_N2")]
locations_N<-data.matrix(locations_N)

store_features<-data[1:S,c("store_features1","store_features2")] #c(0.8, 0.5,1.0)
store_features<-data.matrix(store_features)

sigma_y_actual<- 0.5

y_array<-data[1:S,"Y"]
